# =============================================================================
# 模型定义文件-----自定义 BERT_ASC 模型 + GRoIE 模块（融合 top-4 层特征）
# =============================================================================
import torch
import torch.nn as nn
#从自定义建模层导入所需组件（BERT基础模型、预训练模型基类、BERT层、池化层）
from layers.modeling_custom import BertModel, BertPreTrainedModel, BertLayer, BertPooler


class GRoIE(nn.Module):
    """
    GRoIE模块：用于对BERT顶层多层输出特征进行再处理与融合
    核心作用：对BERT的top-N层特征分别优化后，融合得到更全面的语义表征
    """
    def __init__(self, count, config, num_labels):
        """
        初始化函数
        :param count: 要融合的BERT层数（此处对应top-4，即count=4）
        :param config: BERT模型配置对象（包含隐藏层维度、注意力头数等参数）
        :param num_labels: 分类任务的标签数量（情感类别数）
        """
        super(GRoIE, self).__init__()   #调用父类nn.Module的初始化方法
        self.count = count    #记录要融合的层数
        self.num_labels = num_labels    #记录标签数量
        self.pooler = BertPooler(config)   #BERT池化层（将序列特征转为单向量，默认使用[CLS] token）
        self.pre_layers = torch.nn.ModuleList()  #存储每层特征的预处理BERT层
        self.loss_fct = torch.nn.ModuleList()   #存储每层对应的损失函数
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)   #最终分类器：将融合后的隐藏层特征映射到标签空间
        #初始化对应的预处理层和损失函数
        for i in range(count):
            self.pre_layers.append(BertLayer(config))   #每层特征的独立预处理BERT层（微调特征）
            self.loss_fct.append(torch.nn.CrossEntropyLoss(ignore_index=-1))   #交叉熵损失函数（忽略标签为-1的样本，-1通常表示无效样本）


    def forward(self, layers, attention_mask, labels):
        """
        前向传播函数：处理BERT多层输出，完成特征融合与分类
        :param layers: BERT输出的多层特征列表（每层形状为[batch_size, seq_len, hidden_size]）
        :param attention_mask: 注意力掩码（区分有效token和填充token，形状为[batch_size, seq_len]）
        :param labels: 情感标签（训练时传入，形状为[batch_size]；预测时为None）
        :return: total_loss - 总损失（训练时）；avg_logits - 融合后的分类logits（未归一化预测分数）
        """
        losses = []     #存储每层的损失
        logitses = []   #存储每层的分类logits
        #遍历每个要融合的BERT层（从顶层开始：layers[-1]是最后一层，layers[-2]是倒数第二层，以此类推）
        for i in range(self.count):
            # 1. 取第i个目标层特征（layers[-i-1]：i=0→最后一层，i=1→倒数第二层...）
            # 2. 经过预处理BERT层微调特征（适配当前层的语义特征）
            layer = self.pre_layers[i](layers[-i-1], attention_mask)
            
            # 3. 池化：将序列特征转为单向量
            layer = self.pooler(layer)
            
            # 4. 分类：将池化后的向量映射到标签空间，得到logits
            logits = self.classifier(layer)

            #若传入标签，则计算当前层的损失；否则记录logits
            if labels is not None:
                # 计算当前层损失
                loss = self.loss_fct[i](logits.view(-1, self.num_labels), labels.view(-1))
                losses.append(loss)
            logitses.append(logits)   #记录当前层logits
            
        #计算总损失（训练时）：将所有层的损失求和
        if labels is not None:
            total_loss = torch.sum(torch.stack(losses), dim=0)
        else:
            total_loss = torch.Tensor(0)    #预测时返回空张量
        avg_logits = torch.sum(torch.stack(logitses), dim=0)/self.count   #融合多层logits：取所有层logits的平均值
        return total_loss, avg_logits   #返回总损失和融合后的logits



class BERT_ASC(BertPreTrainedModel):
    """
    BERT-ASC模型：基于BERT的情感分类模型（Aspect-based Sentiment Classification）
    核心特点：融合BERT顶层4层特征，充分利用深层语义信息，提升分类性能
    """
    def __init__(self, config, num_labels=None):
        """
        初始化函数
        :param config: BERT模型配置对象
        :param num_labels: 标签数量
        """
        super(BERT_ASC, self).__init__(config)    #调用父类BertPreTrainedModel的初始化方法
        
        # use external num_labels or config
        if num_labels is None:
            num_labels = config.num_labels
        config.num_labels = num_labels
        
        self.num_labels = num_labels
        self.bert = BertModel(config, output_attentions=True)   #初始化BERT基础模型（output_attentions=True：输出注意力权重）
        self.groie = GRoIE(count=4, config=config, num_labels=num_labels)   #初始化GRoIE模块（融合top-4层特征，count=4）
        self.post_init()

    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, return_attention=False):
        """
        前向传播函数：整合BERT特征提取与GRoIE特征融合，完成情感分类
        :param input_ids: token对应的索引ID序列（形状：[batch_size, seq_len]）
        :param token_type_ids: 句子分段ID（区分输入中的不同句子，默认None）
        :param attention_mask: 注意力掩码（区分有效token和填充token，默认None）
        :param labels: 情感标签（训练时传入，默认None）
        :param return_attention: 是否返回注意力权重（默认False，用于可视化或分析）,设置 return_attention=True 可输出注意力矩阵
        :return: 训练时返回总损失；预测时返回logits（或注意力权重+logits）
        """
        #调用BERT模型提取特征
        #输出：attention-注意力权重，layers-所有层编码特征，_（预留输出），mask-注意力掩码
        attention, layers, _, mask = self.bert(
            input_ids, 
            token_type_ids,
            attention_mask=attention_mask,
            output_all_encoded_layers=True   #输出所有层的编码特征
        )
        
        loss, logits = self.groie(layers, mask, labels)   #调用GRoIE模块融合top-4层特征，计算损失和logits
        
        #根据是否传入标签和是否需要返回注意力权重，返回不同结果
        if labels is not None:
            return loss   #训练时：仅返回总损失
        else:
            if return_attention:
                return attention, logits   #预测时：返回注意力权重和logits
            else:
                return logits   #预测时：仅返回logits

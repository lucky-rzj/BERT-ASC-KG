# =============================================================================
# 模型定义文件-----基于预训练 BERT 的 “基础版” 情感分析模型
# 直接使用 AutoModelForSequenceClassification（仅最后一层 [CLS] 向量分类）
# =============================================================================

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification   #导入Transformers库的配置类和序列分类模型类


#自定义模块
class BERT_ASC_vanila(nn.Module):
    def __init__(self, args, hidden_size=256):
        """
        初始化函数
        :param args: 配置参数对象（包含预训练BERT名称、标签维度等）
        :param hidden_size: 隐藏层维度（默认256，本模型未直接使用，保留为兼容参数）
        """
        super(BERT_ASC_vanila, self).__init__()   #调用父类nn.Module的初始化方法
        config = AutoConfig.from_pretrained(args.pretrained_bert_name)    #从预训练BERT模型加载配置信息
        config.num_labels = args.label_dim   #设置分类任务的标签维度（即情感类别数量，如正向/中性/负向对应3）
        #加载预训练BERT序列分类模型，并传入自定义配置
        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            args.pretrained_bert_name,   #预训练BERT模型名称（如"bert-base-uncased"）
            config=config   #自定义配置（含标签维度）
        )

    
    def forward(self,  input_ids, token_type_ids=None, attention_mask=None, labels=None, return_attention=False):
        """
        前向传播函数（模型核心计算逻辑）
        :param input_ids: token对应的索引ID序列（模型输入的核心数据）
        :param token_type_ids: 句子分段ID（区分输入中的不同句子，默认None）
        :param attention_mask: 注意力掩码（区分有效token和填充token，默认None）
        :param labels: 情感标签（训练时传入，用于计算损失；预测时可省略，默认None）
        :param return_attention: 是否返回注意力权重（本模型未使用，默认False）
        :return: 分类logits（未经过softmax的原始预测分数，维度为[batch_size, num_labels]）
        """
        #调用BERT编码器进行前向计算，传入所有输入参数
        outputs = self.encoder(
            input_ids,   #token索引序列
            token_type_ids=token_type_ids,   #句子分段ID
            attention_mask=attention_mask   #注意力掩码
        )
        return outputs['logits']

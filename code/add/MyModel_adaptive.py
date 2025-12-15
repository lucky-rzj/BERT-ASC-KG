# =================================================================================================
# 自适应层级融合（Adaptive Layer Fusion） + 任务向量（e_task） + 先验偏置（p_semantic/p_syntax）
# =================================================================================================
import math
import torch
import torch.nn as nn
#导入自定义BERT模块
from layers.modeling_custom import BertModel, BertPreTrainedModel, BertLayer,BertPooler


class AdaptiveLayerFusion(nn.Module):
    """
    自适应层级融合模块 (Adaptive Layer Fusion, ALF)
    - 取 top-K 层 hidden states
    - 通过任务向量 e_task 做跨层注意力
    - 门控 g_i 过滤冗余层
    - 加入语义/句法先验，控制底层/顶层偏好
    """
    def __init__(
        self,
        count: int,
        config,
        num_labels: int,
        task_mode: str = "sentiment_polarity",     # 'sentiment_polarity' or 'implicit_aspect'
        use_mean_pool: bool = False,     #池化：True: mean-pooling, False: CLS
        dataset_type: str = "semeval",   # 新增：数据集类型，用于选择任务向量
    ):
        super().__init__()
        self.count = count   #记录要融合的层数
        self.num_labels = num_labels
        self.task_mode = task_mode
        self.use_mean_pool = use_mean_pool
        self.hidden_size = config.hidden_size
        self.dataset_type = dataset_type  # 新增：保存数据集类型

        # 对每一层做一次轻量 BertLayer 预处理
        self.pre_layers = nn.ModuleList([BertLayer(config) for _ in range(count)])

        # 用于 CLS 池化
        self.pooler = BertPooler(config)

        # 任务向量：根据任务模式和数据集类型（模板类型）分别定义
        # 新增：每种任务模式下为不同模板定义独立的任务向量
        self.e_implicit_template1 = nn.Parameter(torch.randn(self.hidden_size))  # What is the sentiment of...
        self.e_implicit_template2 = nn.Parameter(torch.randn(self.hidden_size))  # The sentiment of... in the sentence is?
        self.e_sentiment_template1 = nn.Parameter(torch.randn(self.hidden_size))
        self.e_sentiment_template2 = nn.Parameter(torch.randn(self.hidden_size))

        # 跨层注意力：Q(任务向量) · K(每层表示)
        self.W_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_k = nn.Linear(self.hidden_size, self.hidden_size)

        # 门控：g_i = σ(W_g [h_i ; e_task])
        self.W_g = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.sigmoid = nn.Sigmoid()

        # 分类头
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        # 注册语义/句法先验（针对 top-K 层）
        # 约定：我们按照 “从底到顶” 的顺序取 top-K： [L9, L10, L11, L12]
        # p_semantic：越顶层权重越大；p_syntax：越底层权重越大
        p_semantic = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
        p_syntax = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
        p_semantic = p_semantic / p_semantic.sum()
        p_syntax = p_syntax / p_syntax.sum()

        self.register_buffer("p_semantic", p_semantic)
        self.register_buffer("p_syntax", p_syntax)

        
    # ----------------------- 工具：mask mean pooling -----------------------
    def masked_mean_pool(self, hidden_states, attention_mask):
        """
        hidden_states: [B, L, H]
        attention_mask: [B, L]
        """
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        masked = hidden_states * mask
        summed = masked.sum(dim=1)  # [B, H]
        denom = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
        return summed / denom

        
    # ----------------------------- 前向传播 ------------------------------
    def forward(self, layers, attention_mask, labels=None):
        """
        layers: list of 12 layer outputs, each [B, L, H]
        attention_mask: [B, L]
        labels: [B]
        返回：total_loss, logits
        """
        batch_size = attention_mask.size(0)

        # 1) 取 top-K 层（从底到顶：L9,L10,L11,L12）
        start = len(layers) - self.count
        selected_layers = layers[start:]    # list 长度 = count

        # 2) 每层 -> 预处理 BertLayer -> 池化候选表示 h_i
        pooled_list = []
        for i, h in enumerate(selected_layers):
            h = self.pre_layers[i](h, attention_mask)  # [B, L, H]
            if self.use_mean_pool:
                h_pooled = self.masked_mean_pool(h, attention_mask)    # [B, H], 使用mean-pooling
            else:
                h_pooled = self.pooler(h)      # [B, H], 使用 CLS
            pooled_list.append(h_pooled)

        # [B, K, H]
        H = torch.stack(pooled_list, dim=1)

        # 3) 选取任务向量 e_task：根据任务模式和数据集类型（模板类型）选择
        if self.task_mode.lower() == "implicit_aspect":
            # 隐式方面任务：根据数据集类型选择对应的模板向量
            if self.dataset_type == "sentihood":
                e_task = self.e_implicit_template2  # 使用template2
            else:
                e_task = self.e_implicit_template1  # 使用template1
            lambda_sem = 0.8     #语义偏置更强
        else:
            # 情感极性任务：根据数据集类型选择对应的模板向量
            if self.dataset_type == "sentihood":
                e_task = self.e_sentiment_template2  # 使用template2
            else:
                e_task = self.e_sentiment_template1  # 使用template1
            lambda_sem = 0.5      #句法/语义平衡

        e_task = e_task.unsqueeze(0).expand(batch_size, -1)  # [B, H]

        # 4) 门控：g_i = σ(W_g [h_i ; e_task])
        e_rep = e_task.unsqueeze(1).expand(-1, self.count, -1)  # [B, K, H]
        gate_in = torch.cat([H, e_rep], dim=-1)  # [B, K, 2H]
        g = self.sigmoid(self.W_g(gate_in))      # [B, K, H]
        H_gated = H * g                          # [B, K, H]

        # 5) 任务向量驱动的跨层注意力
        q = self.W_q(e_task)          # [B, H]
        k = self.W_k(H_gated)         # [B, K, H]

        # scaled dot-product: score_i = (q · k_i)/sqrt(H)
        score = (q.unsqueeze(1) * k).sum(dim=-1)  # [B, K]
        score = score / math.sqrt(self.hidden_size)

        # 6) 加入语义/句法先验偏置
        # p = λ p_semantic + (1-λ) p_syntax
        p = lambda_sem * self.p_semantic + (1.0 - lambda_sem) * self.p_syntax  # [K]
        p = p / p.sum()
        score = score + torch.log(p.unsqueeze(0) + 1e-8)    # broadcast

        alpha = torch.softmax(score, dim=-1)  # [B, K]

        # 7) 融合：Σ α_i · (g_i ⊙ h_i) = Σ α_i · H_gated
        fused = torch.sum(H_gated * alpha.unsqueeze(-1), dim=1)  # [B, H]

        logits = self.classifier(fused)   # [B, num_labels]
        
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            zero = torch.tensor(0.0, device=logits.device)
            return zero, logits



class BERT_ASC_adaptive(BertPreTrainedModel):
    """
    自适应层级融合版本的 BERT-ASC：
    - 基于 BERT-PT_rest backbone :contentReference[oaicite:7]{index=7}
    - 顶层 4 层做 AdaptiveLayerFusion
    """
    def __init__(self, config, num_labels=None, task_mode: str = "polarity", dataset_type: str = "semeval"):
        super().__init__(config)

        if num_labels is None:
            num_labels = config.num_labels
        config.num_labels = num_labels

        self.num_labels = num_labels
        self.task_mode = task_mode
        self.dataset_type = dataset_type  # 新增：保存数据集类型

        # BERT backbone
        self.bert = BertModel(config, output_attentions=True)

        # 自适应层级融合模块：传入数据集类型
        self.layer_fusion = AdaptiveLayerFusion(
            count=4,
            config=config,
            num_labels=num_labels,
            task_mode=task_mode,
            use_mean_pool=False,    #默认用 CLS
            dataset_type=dataset_type,  # 新增：传递数据集类型
        )

        self.post_init()

    
    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        return_attention=False,
    ):
        attention, layers, _, mask = self.bert(
            input_ids,
            token_type_ids,
            attention_mask=attention_mask,
            output_all_encoded_layers=True,    #取出 BERT 的全部 12 层 hidden states  [L1, L2, ..., L12]
        )

        loss, logits = self.layer_fusion(layers, mask, labels)

        if labels is not None:
            return loss   # 训练时返回 loss，测试时返回logits
        else:
            if return_attention:
                return attention, logits
            return logits
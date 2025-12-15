# ================================================================
# 兼容 bart-base, bart-large, roberta-base, roberta-large
# 输入只需：input_ids, attention_mask, labels
# ================================================================
import math
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    BartModel,
    RobertaModel
)


# ---------------------------------------------------------------
#  模块：CLS Pooler（RoBERTa / BART 没有 pooler）
# ---------------------------------------------------------------
class CLS_Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.Tanh()

    def forward(self, hs):
        x = hs[:, 0]
        return self.act(self.dense(x))


# ---------------------------------------------------------------
#  Adaptive Layer Fusion（保留你 baseline 的核心结构）
# ---------------------------------------------------------------
class AdaptiveLayerFusion(nn.Module):
    def __init__(self, count, hidden_size, num_labels, task_mode="sentiment_polarity"):
        super().__init__()

        self.count = count
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.task_mode = task_mode

        # task vectors
        self.e_implicit = nn.Parameter(torch.randn(hidden_size))
        self.e_sentiment = nn.Parameter(torch.randn(hidden_size))

        # projections
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)

        # gate
        self.W_g = nn.Linear(hidden_size * 2, hidden_size)
        self.sigmoid = nn.Sigmoid()

        # classifier
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        # priors
        p_semantic = torch.tensor([0.1, 0.2, 0.3, 0.4])
        p_syntax   = torch.tensor([0.4, 0.3, 0.2, 0.1])
        self.register_buffer("p_semantic", p_semantic / p_semantic.sum())
        self.register_buffer("p_syntax", p_syntax / p_syntax.sum())


    def forward(self, hidden_states, attention_mask, labels=None):
        B = attention_mask.size(0)

        # pick top-K layers
        layers = list(hidden_states[-self.count:])

        # pool each layer
        pooled = [h[:, 0] for h in layers]      # CLS
        H = torch.stack(pooled, dim=1)          # [B, K, H]

        # choose task vector
        if self.task_mode == "implicit_aspect":
            e_task = self.e_implicit
            lambda_sem = 0.8
        else:
            e_task = self.e_sentiment
            lambda_sem = 0.5

        e_task = e_task.unsqueeze(0).expand(B, -1)  # [B, H]

        # gate
        B, K, Hdim = H.size()                       # Hdim 就是 hidden_size
        e_expanded = e_task.unsqueeze(1).expand(B, K, -1)  # [B, K, H]
        gate_in = torch.cat([H, e_expanded], dim=-1)       # [B, K, 2H]
        g = self.sigmoid(self.W_g(gate_in))                # [B, K, H]
        H = H * g
       

        # attention: score = q·k
        q = self.W_q(e_task).unsqueeze(1)        # [B, 1, H]
        k = self.W_k(H)                           # [B, K, H]
        score = (q * k).sum(-1) / math.sqrt(self.hidden_size)  # [B, K]

        # priors
        p = lambda_sem * self.p_semantic + (1 - lambda_sem) * self.p_syntax
        score = score + torch.log(p.unsqueeze(0) + 1e-12)

        alpha = torch.softmax(score, dim=-1)      # [B, K]
        fused = torch.sum(H * alpha.unsqueeze(-1), dim=1)

        logits = self.classifier(fused)

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            return loss, logits

        return torch.tensor(0.0, device=logits.device), logits


# ---------------------------------------------------------------
#  PLM ASC 模型 —— 核心统一模型
# ---------------------------------------------------------------
class PLM_ASC(nn.Module):
    """
    替代 BERT_ASC_adaptive 的 PLM 统一版本
    支持：
        - bart-base / bart-large
        - roberta-base / roberta-large
    forward(input_ids, attention_mask, labels)
    """
    def __init__(self, config, model_type, num_labels, task_mode):
        super().__init__()

        self.model_type = model_type
        self.hidden_size = config.hidden_size
        self.task_mode = task_mode
        self.num_labels = num_labels

        # backbone
        if model_type == "bart":
            self.plm = BartModel(config)
        else:
            self.plm = RobertaModel(config)

        # no pooler inside PLM
        self.pooler = CLS_Pooler(self.hidden_size)

        # adaptive fusion
        self.fusion = AdaptiveLayerFusion(
            count=4,
            hidden_size=self.hidden_size,
            num_labels=num_labels,
            task_mode=task_mode
        )

    @classmethod
    def from_pretrained(cls, pretrained_model, num_labels, task_mode):
        config = AutoConfig.from_pretrained(pretrained_model)
        config.output_hidden_states = True
        config.return_dict = True

        name = pretrained_model.lower()
        if "bart" in name:
            model_type = "bart"
            model = cls(config, model_type, num_labels, task_mode)
            model.plm = BartModel.from_pretrained(pretrained_model, config=config)
        else:
            model_type = "roberta"
            model = cls(config, model_type, num_labels, task_mode)
            model.plm = RobertaModel.from_pretrained(pretrained_model, config=config)

        return model

    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Get hidden states for both BART and RoBERTa/BERT
        if self.model_type == "bart":
            # BART returns hidden states separately for encoder/decoder
            if outputs.encoder_hidden_states is not None:
                hidden_states = outputs.encoder_hidden_states   # list of encoder layers
            else:
                raise ValueError("BART did not return encoder hidden states. Ensure output_hidden_states=True.")
        else:
            # BERT / RoBERTa
            hidden_states = outputs.hidden_states

        loss, logits = self.fusion(hidden_states, attention_mask, labels)

        return loss if labels is not None else logits

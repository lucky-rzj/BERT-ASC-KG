# =====================================================
# PLM-ASC 模型： Unified Model for RoBERTa / BART
# =====================================================
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class MeanPooling(nn.Module):
    """ 对最后一层隐藏状态做均值池化 """
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        masked = last_hidden_state * mask
        sum_hidden = masked.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-9)
        return sum_hidden / lengths



class PLM_ASC(nn.Module):
    """
    Unified ABSA Model for:
      - roberta-base / roberta-large
      - bart-base / bart-large
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        # Auto load config + model
        self.config = AutoConfig.from_pretrained(opt.pretrained_model)
        self.encoder = AutoModel.from_pretrained(opt.pretrained_model, config=self.config)
        
        # ⭐⭐⭐ 开启 Gradient Checkpointing（显存减少 30–50%） ⭐⭐⭐
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        

        hidden_size = self.config.hidden_size
        
        # pooling 层
        self.pooler = MeanPooling()
        
        # dropout
        self.dropout = nn.Dropout(opt.dropout)
        
        # 分类层
        self.classifier = nn.Linear(hidden_size, opt.label_dim)

    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Training: return loss
        Inference: return logits
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # ------------- 处理 RoBERTa / BART 统一输出 ----------------
        if hasattr(outputs, "encoder_last_hidden_state"):
            # BART: use encoder output
            last_hidden = outputs.encoder_last_hidden_state
        else:
            # RoBERTa: use last hidden state
            last_hidden = outputs.last_hidden_state

        # Mean pooling
        pooled = self.pooler(last_hidden, attention_mask)
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        # ----------------- Training -----------------
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            return loss_fct(logits, labels)

        # ----------------- evaluation -----------------
        return logits
 

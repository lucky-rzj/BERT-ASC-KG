# ================================================================
#  modeling_custom.py —— 适配 BERT_ASC + GRoIE + BERT-PT_rest
#  --------------------------------------------------------------
#  1. 直接继承 HuggingFace BertPreTrainedModel / BertModel
#  2. 只做最小封装，使返回值满足 BERT_ASC.forward 的调用格式
#  3. 支持 activebus/BERT-PT_rest 完整加载
# ================================================================


import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import (
    BertModel as HF_BertModel,
    BertPreTrainedModel as HF_BertPreTrainedModel,
    BertLayer as HF_BertLayer,
    BertPooler as HF_BertPooler,
)



# ----------------------------------------------------------
# 重新导出（供 MyModel_pt.py import）
# ----------------------------------------------------------
BertPreTrainedModel = HF_BertPreTrainedModel


# ----------------------------------------------------------
# GRoIE 使用的轻量 BertLayer（自定义）
# ----------------------------------------------------------
class BertLayer(nn.Module):
    """
    改进版 GRoIE Layer —— 更接近 BERT 的 encoder block
    （不会影响加载 BERT-PT_rest 预训练权重）
    """
    def __init__(self, config):
        super().__init__()
        
        # === 修改 1：加入 Q/K/V 全连接层，模仿 BERT 的 Self-Attention ===
        # -------------------------------------------------------------
        self.q = nn.Linear(config.hidden_size, config.hidden_size)   
        self.k = nn.Linear(config.hidden_size, config.hidden_size)   
        self.v = nn.Linear(config.hidden_size, config.hidden_size)   

        self.att = nn.MultiheadAttention(
            embed_dim=config.hidden_size, 
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        
        self.dropout_att = nn.Dropout(config.hidden_dropout_prob) 
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.dropout_ff = nn.Dropout(config.hidden_dropout_prob)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
                     

    def forward(self, hidden, attn_mask=None):
        
        # Q/K/V
        q = self.q(hidden)   
        k = self.k(hidden) 
        v = self.v(hidden)
        
        key_mask = (attn_mask == 0) if attn_mask is not None else None
        
        # === Multi-head Attention ===
        attn_out, _ = self.att(q, k, v, key_padding_mask=key_mask)
        hidden = self.norm1(hidden + self.dropout_att(attn_out))

        # === FFN ===
        ff_out = self.ff(hidden)
        hidden = self.norm2(hidden + self.dropout_ff(ff_out))

        return hidden


# ----------------------------------------------------------
# Pooler —— [CLS]
# ----------------------------------------------------------
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.Tanh()

    def forward(self, hs):
        cls = hs[:, 0]
        return self.act(self.dense(cls))



# ------- 核心：继承 HF_BertModel，而不是套一层 self.bert ------- 
class BertModel(HF_BertModel):
    def __init__(self, config, output_attentions=False, output_hidden_states=True, **kwargs):
        super().__init__(config, add_pooling_layer=True)
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

    
    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=True,
        **kwargs,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=self.output_attentions,
            output_hidden_states=True,
            return_dict=False,
        )

        # HF: (last_hidden_state, pooled_output, all_hidden_states, all_attentions)
        seq_output, pooled, hidden_states, attentions = outputs

        # hidden_states: [emb, layer1, ..., layer12]
        all_layers = list(hidden_states[1:])  # 去掉 embedding，只保留 12 层 encoder

        # 返回格式满足 BERT_ASC 的预期
        return attentions, all_layers, pooled, attention_mask

# =======================================================
# 数据预处理----将 ABSA JSON 转为输入张量（PLM版本）
# 兼容：BART-base/large, RoBERTa-base/large
# =======================================================
import json
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


# =======================================================
# Tokenizer 封装
# =======================================================
class ABSATokenizer:
    """ 通用 ABSA tokenizer：自动支持 RoBERTa / BART """
    def __init__(self, pretrained_name):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name, use_fast=False)

    @classmethod
    def from_pretrained(cls, pretrained_name):
        return cls(pretrained_name)

    def encode_pair(self, auxiliary, text, max_len):
        """ 编码辅助句 + 原句为模型输入 """
        encoded = self.tokenizer(
            auxiliary,
            text,
            max_length = max_len,
            truncation = True,
            padding = "max_length",
            return_tensors = "np"
        )
        return (
            encoded["input_ids"][0],
            encoded["attention_mask"][0]
        )

        

# =================================================
# 工具函数：排序辅助词
# =================================================
def sort_auxiliary(text, auxiliary):
    """
    根据 auxiliary 在原文本中的顺序排序。
    - text: 原始句子
    - auxiliary: list 或 str
    返回排序后的 list
    """
    if isinstance(auxiliary, str):
        auxiliary = auxiliary.split()

    #统一 lowercase
    text_words = text.lower().split()
    auxiliary = [w.lower() for w in auxiliary]

    pos_map = {w: i for i, w in enumerate(text_words)}

    #排序：在 text 出现的按顺序；不出现的排后，同时保持辅助词原有顺序
    sorted_idx = sorted(
        range(len(auxiliary)),
        key=lambda i: pos_map.get(auxiliary[i], 1e9)
    )

    sorted_aux = [auxiliary[i] for i in sorted_idx]
    
    #返回list
    return sorted_aux        



# =======================================================
# SEMEVAL Dataset
# =======================================================
class ABSADataset_absa_bert_semeval_json(Dataset):
    def __init__(self, fname, tokenizer, opt):
        self.opt = opt
        self.data = []
        self.tokenizer = tokenizer

        with open(fname) as f:
            raw_data = json.load(f)
            
        for instance in tqdm(raw_data):
            text_raw = instance["text"]
            
            for asp in instance["aspect"]:
                category = asp["category"]
                auxiliary = asp["auxiliary"]
                opinions = asp["opinions"]

                # ---- 保证 auxiliary 和 opinions 均为 list ----
                if isinstance(auxiliary, str):
                    auxiliary = auxiliary.split()
                if isinstance(opinions, str):
                    opinions = opinions.split()
                    
                #辅助句格式：sorted (category + auxiliary + opinions) 
                aux_list = [category] + auxiliary + opinions         #构建辅助信息：类别 + 辅助词 + 观点词 
                aux_list = list(set(aux_list))                       #去重
                aux_list = sort_auxiliary(text_raw, aux_list)        #按原句顺序排序
                auxiliary = " ".join(aux_list)                       #拼成辅助句

                #label映射
                label_map = {
                    "positive": 0, 
                    "neutral": 1, 
                    "negative": 2,
                    "conflict": 3, 
                    "none": 4
                }
                label = label_map[asp["polarity"]]

                #Encode
                input_ids, att_mask = tokenizer.encode_pair(auxiliary,text_raw, opt.max_seq_len)
                self.data.append({
                    "text": text_raw,
                    "text_bert_indices": input_ids.astype("int64"),
                    "input_mask": att_mask.astype("int64"),
                    "label": label
                })

    
    def __getitem__(self, index):
        """按索引获取单条数据"""
        return self.data[index]

        
    def __len__(self):
        """返回数据集总长度"""
        return len(self.data)



# =======================================================
# SENTIHOOD Dataset
# =======================================================
class ABSADataset_absa_bert_sentihood_json(Dataset):
    def __init__(self, fname, tokenizer, opt):
        self.opt = opt
        self.data = []
        self.tokenizer = tokenizer
        
        with open(fname) as f:
            raw_data = json.load(f)

        for instance in tqdm(raw_data):
            text_raw = instance["text"]
            for asp in instance["aspect"]:
                category = asp["category"]
                auxiliary = asp["auxiliary"]
                target = asp["target"]    #目标实体
                opinions = asp["opinions"]

                # ---- 保证 auxiliary & opinions 为 list ----
                if isinstance(auxiliary, str):
                    auxiliary = auxiliary.split()
                if isinstance(opinions, str):
                    opinions = opinions.split()

                #辅助句格式：target + category + sorted (auxiliary + opinions)
                aux_list = auxiliary + opinions                       #构建辅助信息：辅助词 + 观点词
                aux_list = sort_auxiliary(text_raw, aux_list)         #按原句顺序排序
                auxiliary = " ".join([target, category] + aux_list)   #拼成辅助句
                

                #label映射
                label_map = {"None": 0, "Positive": 1, "Negative": 2}
                label = label_map[asp["polarity"]]

                #Encode
                input_ids, att_mask = tokenizer.encode_pair(auxiliary, text_raw, opt.max_seq_len)
                self.data.append({
                    "text": text_raw,
                    "text_bert_indices": input_ids.astype("int64"),
                    "input_mask": att_mask.astype("int64"),
                    "label": label
                })

    
    def __getitem__(self, index):
        return self.data[index]

    
    def __len__(self):
        return len(self.data)
    
 
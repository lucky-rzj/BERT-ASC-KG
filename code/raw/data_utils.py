# ==============================================================================
# BERT 数据预处理-----使用 transformers 接口（tokenizer.encode_plus）自动构建输入
# ==============================================================================
import json
import numpy as np
from torch.utils.data import Dataset   #PyTorch数据集基类
import pandas as pd
from tqdm import tqdm
import unicodedata
import six


class ABSADataset_absa_bert_semeval_json(Dataset):
    def __init__(self, fname, tokenizer, opt):
        self.opt = opt
        all_data = []
        with open(fname) as f :
            dataa= json.load(f)
        f.close()
        
        #遍历每条原始数据
        for i in tqdm(range(len(dataa))):
            text = dataa[i]['text']
            aspects = dataa[i]['aspect']
            text_=text   #保存原始文本
            #遍历每个方面
            for current_aspect in aspects:
                category = current_aspect['category']
                auxiliary = current_aspect['auxiliary']
                opinions = current_aspect['opinions']

                auxiliary.append(category)
                auxiliary.extend(opinions)
                auxiliary= list(set(auxiliary))
                auxiliary=self.sort_auxiliary(text_, auxiliary)

                #处理情感标签：将文本标签映射为数字ID
                label = current_aspect['polarity']
                label = {a: _ for _, a in enumerate(['positive', 'neutral', 'negative', 'conflict', 'none'])}.get(label)

                #构建完整辅助句（疑问句式："XX的情感是什么？"）
                auxiliary ='what is the sentiment of ' + auxiliary

                #使用BERT分词器处理原始文本和辅助句，生成模型输入格式
                example = tokenizer.encode_plus(
                    text, 
                    auxiliary,
                    add_special_tokens=True,   #添加BERT专用特殊标记（[CLS]、[SEP]）
                    truncation = True,    #自动截断过长序列（超过max_length时）
                    padding = 'max_length',   #自动填充至最大序列长度
                    max_length=self.opt.max_seq_len,    #序列最大长度（从配置参数获取）
                    return_token_type_ids=True   #返回句子分段ID（区分两句文本）
                )
                
                #构建单条数据字典（包含模型输入和标签）
                data = {
                    'text': text_,
                    'text_bert_indices': np.asarray(example['input_ids'], dtype='int64'),
                    'bert_segments_ids': np.asarray(example['token_type_ids'], dtype='int64'),
                    'input_mask': np.asarray(example['attention_mask'], dtype='int64'),
                    'label': label,
                }

                all_data.append(data)
        self.data = all_data

    def sort_auxiliary(self, text_a, text_b):
        """
        按 text_a 中的词序对 text_b 排序。
        - text_b 可为字符串或列表
        - 未在 text_a 出现的词排在末尾并保持原顺序
        """
        text_a_list = text_a.split()

        # ---- 兼容 text_b 为 list 或 str ----
        if isinstance(text_b, str):
            text_b_list = text_b.split()
        elif isinstance(text_b, list):
            text_b_list = text_b
        else:
            raise ValueError(f"text_b 既不是 str 也不是 list，而是 {type(text_b)}")

        # 记录 text_a 中每个词的索引
        pos_map = {w: i for i, w in enumerate(text_a_list)}

        # 排序规则：
        #   若单词在 text_a 中 → 依据其位置排序
        #   若不在 → 排到最后，同时保持 text_b 原顺序
        sorted_b = sorted(
            text_b_list,
            key=lambda w: pos_map[w] if w in pos_map else len(text_a_list) + text_b_list.index(w)
        )

        return " ".join(sorted_b)



    def __getitem__(self, index):
        """按索引获取单条数据"""
        return self.data[index]

    
    def __len__(self):
        """返回数据集总长度"""
        return len(self.data)


        
class ABSADataset_absa_bert_sentihood_json(Dataset):
    def __init__(self, fname, tokenizer, opt):
        self.opt = opt
        all_data = []
        with open(fname) as f :
            dataa= json.load(f)
        f.close()
        
        for i in tqdm(range(len(dataa))):
            text = dataa[i]['text']
            aspects = dataa[i]['aspect']
            text_= text
            for current_aspect in aspects:
                category = current_aspect['category']
                target = current_aspect['target']    #目标实体
                auxiliary = current_aspect['auxiliary']
                opinions = current_aspect['opinions']
                
                auxiliary.extend(opinions)
                auxiliary= self.sort_auxiliary(text_,auxiliary)
                #构建基础辅助句：目标实体 + 方面类别 + 排序后的辅助信息
                auxiliary=target + ' ' + category + ' ' + ' '.join(auxiliary)
                
                #处理情感标签：将文本标签映射为数字ID
                label = current_aspect['polarity']
                label = {a: _ for _, a in enumerate(['None', 'Positive', 'Negative'])}.get(label)
                if label is None or not isinstance(label, int):
                    continue


                auxiliary = 'what is the sentiment of ' + auxiliary
                #使用BERT分词器处理原始文本和辅助句，生成模型输入格式
                example = tokenizer.encode_plus(
                    text, 
                    auxiliary, 
                    add_special_tokens=True, 
                    truncation=True,                       
                    padding='max_length', 
                    max_length=self.opt.max_seq_len,              
                    return_token_type_ids=True
                )
                #构建单条数据字典
                data = {
                    'text_bert_indices': np.asarray(example['input_ids'], dtype='int64'),
                    'bert_segments_ids': np.asarray(example['token_type_ids'], dtype='int64'),
                    'input_mask': np.asarray(example['attention_mask'], dtype='int64'),
                    'label': label,
                }
                all_data.append(data)
        self.data = all_data

    
    def sort_auxiliary(self, text_a, text_b):
        """
        按 text_a 中的词序对 text_b 排序。
        - text_b 可为字符串或列表
        - 未在 text_a 出现的词排在末尾并保持原顺序
        """
        text_a_list = text_a.split()

        # ---- 兼容 text_b 为 list 或 str ----
        if isinstance(text_b, str):
            text_b_list = text_b.split()
        elif isinstance(text_b, list):
            text_b_list = text_b
        else:
            raise ValueError(f"text_b 既不是 str 也不是 list，而是 {type(text_b)}")

        # 记录 text_a 中每个词的索引
        pos_map = {w: i for i, w in enumerate(text_a_list)}

        # 排序规则：
        #   若单词在 text_a 中 → 依据其位置排序
        #   若不在 → 排到最后，同时保持 text_b 原顺序
        sorted_b = sorted(
            text_b_list,
            key=lambda w: pos_map[w] if w in pos_map else len(text_a_list) + text_b_list.index(w)
        )

        return " ".join(sorted_b)

    
    def __getitem__(self, index):
        return self.data[index]

    
    def __len__(self):
        return len(self.data)
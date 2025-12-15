# =============================================================================================================================================
# BERT-PT 数据预处理
# 功能：将 ABSA JSON 转为 BERT 输入张量：text_bert_indices(token索引),bert_segments_ids（句子分段）, attention_mask(注意力掩码), label(情感标签 ID)
# =============================================================================================================================================
import json
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer     #BERT分词器
import pandas as pd
from tqdm import tqdm
import unicodedata
import six


# ----------------新增辅助句模板和生成函数----------------
def generate_auxiliary_sentence(candidate, dataset_type):
    """生成带指令模板的辅助句，根据数据集类型选择不同模板"""
    template1 = "What is the sentiment of [候选词]?"
    template2 = "The sentiment of [候选词] in the sentence is?"
    
    if dataset_type == "sentihood":
        return template2.replace("[候选词]", candidate)
    else:
        return template1.replace("[候选词]", candidate)


class ABSATokenizer(BertTokenizer):
    """ABSA专用分词器（继承自BERT分词器）"""
    # ----------------------- 子词分词的方法 -------------------------------
    def subword_tokenize(self, tokens, labels):   #用于AE（方面提取）任务,{标签通常是 BIO 格式：B = 方面开始词，I = 方面内部词，O = 非方面词}
        split_tokens, split_labels = [], []   #存储拆分后的子词、子词对应的标签
        idx_map = []    #存储「子词」对应「原始词」的索引映射
        for ix, token in enumerate(tokens):   #遍历每个原始词（ix是原始词的索引）
            #第一步：对当前原始词做WordPiece子词拆分（BERT的核心分词逻辑）
            sub_tokens = self.wordpiece_tokenizer.tokenize(token)   
            #第二步：为每个子词分配标签，保证标签逻辑一致
            for jx, sub_token in enumerate(sub_tokens):  #遍历当前原始词拆分出的所有子词（jx是子词在当前词中的索引）
                split_tokens.append(sub_token)    #记录子词
                #处理标签:关键规则----如果原始词的标签是"B"（方面开始），且当前子词不是第一个子词（jx>0），则子词标签改为"I"
                if labels[ix] == "B" and jx > 0:
                    split_labels.append("I")
                else:   #其他情况：标签继承原始词的标签（O保持O，I保持I）
                    split_labels.append(labels[ix])
                #第三步：记录子词对应的原始词索引
                idx_map.append(ix)    
        return split_tokens, split_labels, idx_map   #返回拆分后的token、标签和索引映射


class ABSADataset_absa_bert_semeval_json(Dataset):
    """SEMEVAL数据集专用的ABSA数据集类（JSON格式，适配BERT输入）"""
    def __init__(self, fname, tokenizer, opt):
        self.opt = opt 
        all_data = []   #存储所有处理后的数据
        with open(fname) as f :  
            dataa= json.load(f)
        f.close()
   
        #遍历每条数据
        for i in tqdm(range(len(dataa))):   
            text = dataa[i]['text']
            aspects = dataa[i]['aspect']
            text = self.convert_to_unicode(text)
            text_ = text     #保存原始文本 
            text=tokenizer.tokenize(text.strip().lower())    #原始文本去首尾空白 + 转小写后分词
            
            #遍历每个方面
            for current_aspect in aspects:    
                category = current_aspect['category']
                auxiliary = current_aspect['auxiliary']
                opinions = current_aspect['opinions']
                
                # ----------- 新增：获取情感分数（这里假设数据中有sentiment_scores字段） -----------
                sentiment_scores = current_aspect.get('sentiment_scores', {})  # 情感分数字典 {词: 分数}
                
                # 辅助句子格式：sorted (category + auxiliary + opinions)，带情感权重
                auxiliary_with_scores = []
                for item in auxiliary + [category] + opinions:
                    if item in sentiment_scores:
                        # 拼接句法关系和情感权重，如 "sweet_amod (-0.6)"
                        auxiliary_with_scores.append(f"{item} ({sentiment_scores[item]:.1f})")
                    else:
                        auxiliary_with_scores.append(item)
                auxiliary_with_scores = list(set(auxiliary_with_scores))
                
                # 按原始文本中的词序排序辅助信息
                sorted_aux = self.sort_auxiliary(text_, auxiliary_with_scores)
                
                # 使用新的辅助句生成函数
                candidate = category.split('#')[0]  # 假设category格式为"实体#属性"
                auxiliary_sentence = generate_auxiliary_sentence(candidate, "semeval")
                # 将排序后的辅助信息添加到指令模板后
                full_auxiliary = f"{auxiliary_sentence} {' '.join(sorted_aux)}"
                
                #处理情感标签（转为数字ID）
                label = current_aspect['polarity']
                #标签映射：正向→0，中性→1，负向→2，冲突→3，无→4
                label = {a: _ for _, a in enumerate(['positive', 'neutral', 'negative', 'conflict', 'none'])}.get(label)

                full_auxiliary= self.convert_to_unicode(full_auxiliary)
                full_auxiliary = tokenizer.tokenize(full_auxiliary.strip().lower())   #辅助句子分词

                
                #构建BERT输入格式：[CLS] 辅助句 [SEP] 原始文本 [SEP]
                tokens = []
                segment_ids = []    #句子分段ID（0表示辅助句，1表示原始文本）
                tokens.append("[CLS]")   # BERT起始标记
                segment_ids.append(0)
                #添加辅助句token
                for token in full_auxiliary:   
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")    #辅助句结束标记
                segment_ids.append(0)
                #添加原始文本token
                for token in text:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")    #原始文本结束标记
                segment_ids.append(1)
                
                input_ids = tokenizer.convert_tokens_to_ids(tokens)    #将token转为索引ID
                input_mask = [1] * len(input_ids)    #构建注意力掩码（1表示有效token，0表示填充token）
                
                #截断过长序列（不超过最大序列长度）
                input_ids = input_ids[:self.opt.max_seq_len]
                input_mask = input_mask[:self.opt.max_seq_len]
                segment_ids = segment_ids[:self.opt.max_seq_len]
                #填充短序列（不足最大长度时补0）
                while len(input_ids) < self.opt.max_seq_len:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                #转为numpy数组
                input_ids = np.asarray(input_ids, dtype='int64')
                input_mask = np.asarray(input_mask, dtype='int64')
                segment_ids = np.asarray(segment_ids, dtype='int64')

                #构建单条数据字典
                data = {
                    'text': text_,   #原始文本
                    'text_bert_indices': input_ids,  #BERT输入token索引
                    'bert_segments_ids': segment_ids,   #句子分段ID
                    'input_mask': input_mask,   #注意力掩码
                    'label': label,   #情感标签ID
                }
                all_data.append(data)   
        self.data = all_data    
        
    
    def sort_auxiliary(self, text_a, text_b):
        """
        按原始文本（text_a）中的词序对构建的辅助信息（text_b）排序，未在原始文本中出现的词保留其在text_b中的原始顺序，放在最后
        :param text_a: 原始文本
        :param text_b: 已经构建好的辅助信息列表 (category + auxiliary + opinions)
        :return: 排序后的辅助信息（字符串）
        """
        # -------------------------------
        # >>>: 1.统一转为 word-level排序
        # -------------------------------
        #处理text_b：若为字符串则按空格分词，若为列表则直接使用
        if isinstance(text_b, str):
            text_b = text_b.split()
        
        text_a_words = text_a.strip().lower().split()   #原文词序列表（去空格 + 小写 + 分词）
        aux_words = [w.lower() for w in text_b]         #辅助词列表 (word-level)

        # -------------------------------
        # >>>: 2.计算辅助词在原文中的位置
        # -------------------------------
        positions = []
        for w in aux_words:
            # 忽略情感权重部分，只匹配原始词
            clean_w = w.split()[0] if '(' in w else w
            if clean_w in text_a_words:
                positions.append(text_a_words.index(clean_w))     #在原文中的位置
            else:
                positions.append(10**9)                     #不在原文 → 排在最后
        
        # ------------------------------------------
        # >>>: 3.稳定排序（保持不在原文词的原顺序）
        # ------------------------------------------
        indexed = list(zip(aux_words, positions))
        indexed_sorted = sorted(indexed, key=lambda x: x[1])
        sorted_aux_words = [w for w, _ in indexed_sorted]

        #转换为字符串返回
        return ' '.join(sorted_aux_words)

    

    def convert_to_unicode(self,text):
        """将文本转换为Unicode编码（若尚未是Unicode），假设输入为UTF-8格式"""
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")

            
    def __getitem__(self, index):
        """按索引获取单条数据"""
        return self.data[index]

    
    def __len__(self):
        """返回数据集总长度"""
        return len(self.data)



class ABSADataset_absa_bert_sentihood_json(Dataset):
    """Sentihood数据集专用的ABSA数据集类（JSON格式，适配BERT输入）"""
    def __init__(self, fname, tokenizer, opt):
        self.opt = opt
        all_data = []
        with open(fname) as f :
            dataa= json.load(f)
        f.close()
        
        #遍历每条数据
        for i in tqdm(range(len(dataa))):
            text = dataa[i]['text']
            aspects = dataa[i]['aspect']
            text = self.convert_to_unicode(text)
            text_= text   #保存原始文本
            text=tokenizer.tokenize(text.strip().lower())   #文本分词
            #遍历每个方面
            for current_aspect in aspects:    
                category = current_aspect['category']
                auxiliary = current_aspect['auxiliary']
                opinions = current_aspect['opinions']
                
                # ----------- 新增：获取情感分数（这里假设数据中有sentiment_scores字段） -----------
                sentiment_scores = current_aspect.get('sentiment_scores', {})  # 情感分数字典 {词: 分数}
                
                # 辅助句子格式：sorted (category + auxiliary + opinions)，带情感权重
                auxiliary_with_scores = []
                for item in auxiliary + [category] + opinions:
                    if item in sentiment_scores:
                        # 拼接句法关系和情感权重，如 "sweet_amod (-0.6)"
                        auxiliary_with_scores.append(f"{item} ({sentiment_scores[item]:.1f})")
                    else:
                        auxiliary_with_scores.append(item)
                auxiliary_with_scores = list(set(auxiliary_with_scores))
                
                # 按原始文本中的词序排序辅助信息
                sorted_aux = self.sort_auxiliary(text_, auxiliary_with_scores)
                
                # 使用新的辅助句生成函数
                candidate = category.split('#')[0]  # 假设category格式为"实体#属性"
                auxiliary_sentence = generate_auxiliary_sentence(candidate, "sentihood")
                # 将排序后的辅助信息添加到指令模板后
                full_auxiliary = f"{auxiliary_sentence} {' '.join(sorted_aux)}"
                
                #处理情感标签（转为数字ID）
                label = current_aspect['polarity']
                label = {a: _ for _, a in enumerate(['positive', 'neutral', 'negative', 'conflict', 'none'])}.get(label)

                full_auxiliary= self.convert_to_unicode(full_auxiliary)
                full_auxiliary = tokenizer.tokenize(full_auxiliary.strip().lower())   #辅助句子分词

                #构建BERT输入格式：[CLS] 辅助句 [SEP] 原始文本 [SEP]
                tokens = []
                segment_ids = []    
                tokens.append("[CLS]")   
                segment_ids.append(0)
                #添加辅助句token
                for token in full_auxiliary:   
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")    
                segment_ids.append(0)
                #添加原始文本token
                for token in text:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")    
                segment_ids.append(1)
                
                input_ids = tokenizer.convert_tokens_to_ids(tokens)    
                input_mask = [1] * len(input_ids)    
                
                #截断过长序列
                input_ids = input_ids[:self.opt.max_seq_len]
                input_mask = input_mask[:self.opt.max_seq_len]
                segment_ids = segment_ids[:self.opt.max_seq_len]
                #填充短序列
                while len(input_ids) < self.opt.max_seq_len:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                input_ids = np.asarray(input_ids, dtype='int64')
                input_mask = np.asarray(input_mask, dtype='int64')
                segment_ids = np.asarray(segment_ids, dtype='int64')

                data = {
                    'text': text_,
                    'text_bert_indices': input_ids,
                    'bert_segments_ids': segment_ids,
                    'input_mask': input_mask,
                    'label': label,
                }
                all_data.append(data)   
        self.data = all_data    
        
    # 复用sort_auxiliary和convert_to_unicode方法
    def sort_auxiliary(self, text_a, text_b):
        # 实现与ABSADataset_absa_bert_semeval_json相同的逻辑
        if isinstance(text_b, str):
            text_b = text_b.split()
        
        text_a_words = text_a.strip().lower().split()
        aux_words = [w.lower() for w in text_b]

        positions = []
        for w in aux_words:
            clean_w = w.split()[0] if '(' in w else w
            if clean_w in text_a_words:
                positions.append(text_a_words.index(clean_w))
            else:
                positions.append(10**9)
        
        indexed = list(zip(aux_words, positions))
        indexed_sorted = sorted(indexed, key=lambda x: x[1])
        sorted_aux_words = [w for w, _ in indexed_sorted]

        return ' '.join(sorted_aux_words)
    
    def convert_to_unicode(self, text):
        # 与ABSADataset_absa_bert_semeval_json相同的实现
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")
            
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
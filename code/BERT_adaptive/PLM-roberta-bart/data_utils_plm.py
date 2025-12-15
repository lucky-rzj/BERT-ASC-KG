# =====================================================================================================================
# BERT-PT 数据预处理（已适配 BERT / RoBERTa / BART）
# 功能：将 ABSA JSON 转为 BERT 输入张量：text_bert_indices(token索引),attention_mask(注意力掩码), label(情感标签 ID)
# =====================================================================================================================
import json
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import six



# ==========================
# SEMEVAL
# ==========================
class ABSADataset_absa_bert_semeval_json(Dataset):
    """SEMEVAL数据集专用的ABSA数据集类（JSON格式，适配PLM输入）"""
    
    def __init__(self, fname, tokenizer, opt):
        self.opt = opt 
        self.tokenizer = tokenizer

        #是否使用 token_type_ids（BERT 才使用）
        self.use_token_type = "token_type_ids" in tokenizer.model_input_names

        #读取特殊 token
        self.CLS = tokenizer.cls_token
        self.SEP = tokenizer.sep_token
        self.pad_id = tokenizer.pad_token_id
        
        all_data = []     #存储所有处理后的数据
        with open(fname) as f :  
            dataa= json.load(f)
        f.close()

   
        #遍历每条数据
        for i in tqdm(range(len(dataa))):   
            text = dataa[i]['text']
            aspects = dataa[i]['aspect']
            text = self.convert_to_unicode(text)
            text_ = text     #保存原始文本 
            text_tokens = tokenizer.tokenize(text.strip())    #原始文本去首尾空白后分词
            
            #遍历每个方面
            for current_aspect in aspects:    
                category = current_aspect['category']
                auxiliary = current_aspect['auxiliary']
                opinions = current_aspect['opinions']
                
                #辅助句子格式：sorted (category + auxiliary + opinions)
                auxiliary.append(category)    #构建辅助信息：合并类别、观点词，去重
                auxiliary.extend(opinions)
                auxiliary = list(set(auxiliary))
                auxiliary = self.sort_auxiliary(text_, auxiliary)   #按原始文本中的词序排序辅助信息
                
                #处理情感标签（转为数字ID）
                label = current_aspect['polarity']
                #标签映射：正向→0，中性→1，负向→2，冲突→3，无→4
                label = {a: _ for _, a in enumerate(['positive', 'neutral', 'negative', 'conflict', 'none'])}.get(label)

                auxiliary= self.convert_to_unicode(auxiliary)
                auxiliary_tokens = tokenizer.tokenize(auxiliary.strip())   #辅助句子分词

                
                # ======================================
                # 构建输入：[CLS] aux [SEP] text [SEP]
                # ======================================
                tokens = []
                segment_ids = []    #句子分段ID（0表示辅助句，1表示原始文本）

                # [CLS]
                tokens.append(self.CLS)
                if self.use_token_type:
                    segment_ids.append(0)
                else:
                    segment_ids.append(0)   # BART/RoBERTa 固定为 0

                # 辅助部分
                for t in auxiliary_tokens:
                    tokens.append(t)
                    segment_ids.append(0)

                # [SEP]
                tokens.append(self.SEP)
                segment_ids.append(0)

                # 正文
                for t in text_tokens:
                    tokens.append(t)
                    if self.use_token_type:
                        segment_ids.append(1)
                    else:
                        segment_ids.append(0)   # BART/RoBERTa 忽略 token_type_ids

                # [SEP]
                tokens.append(self.SEP)   
                if self.use_token_type:
                    segment_ids.append(1)
                else:
                    segment_ids.append(0)
                
                #将token转为索引ID
                input_ids = tokenizer.convert_tokens_to_ids(tokens)    
                input_mask = [1] * len(input_ids)    #构建注意力掩码（1表示有效token，0表示填充token）
                
                #截断过长序列（不超过最大序列长度）
                input_ids = input_ids[:self.opt.max_seq_len]
                input_mask = input_mask[:self.opt.max_seq_len]
                segment_ids = segment_ids[:self.opt.max_seq_len]
                #填充短序列（不足最大长度时补0）
                while len(input_ids) < self.opt.max_seq_len:
                    input_ids.append(self.pad_id)
                    input_mask.append(0)
                    segment_ids.append(0)

                #转为numpy数组
                input_ids = np.asarray(input_ids, dtype='int64')
                input_mask = np.asarray(input_mask, dtype='int64')

                #构建单条数据字典
                data = {
                    'text': text_,   #原始文本
                    'text_bert_indices': input_ids,  #BERT输入token索引
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
            if w in text_a_words:
                positions.append(text_a_words.index(w))     #在原文中的位置
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

        

# ==========================================
# SENTIHOOD
# ==========================================
class ABSADataset_absa_bert_sentihood_json(Dataset):
    """Sentihood数据集专用的ABSA数据集类"""
    def __init__(self, fname, tokenizer, opt):
        self.opt = opt
        self.tokenizer = tokenizer

        self.use_token_type = "token_type_ids" in tokenizer.model_input_names
        self.CLS = tokenizer.cls_token
        self.SEP = tokenizer.sep_token
        self.pad_id = tokenizer.pad_token_id

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
            text_tokens = tokenizer.tokenize(text.strip())   #文本分词
            #遍历每个方面
            for current_aspect in aspects:
                category = current_aspect['category'] 
                target = current_aspect['target']    #目标实体
                auxiliary = current_aspect['auxiliary']
                opinions = current_aspect['opinions']
                
                #合并观点词到辅助信息，并按原始文本词序排序
                auxiliary.extend(opinions)
                auxiliary= self.sort_auxiliary(text_,auxiliary)
                
                #辅助句子格式：target + category + sorted (auxiliary + opinions)
                auxiliary = target + ' ' + category + ' ' + auxiliary
                
                #处理情感标签（转为数字ID）
                label = current_aspect['polarity']   
                #标签映射：无→0，正向→1，负向→2（Sentihood数据集标签格式）
                label = {a: _ for _, a in enumerate(['None', 'Positive', 'Negative'])}.get(label)
                assert  label != None   #确保标签有效
                
                auxiliary= self.convert_to_unicode(auxiliary)
                auxiliary_tokens = tokenizer.tokenize(auxiliary.strip())  #辅助句子分词

                
                # ================================================
                # 构建输入：[CLS] 辅助句 [SEP] 原始文本 [SEP]
                # ================================================
                tokens = []
                segment_ids = []   #句子分段ID（0表示辅助句，1表示原始文本）

                tokens.append(self.CLS)   #起始标记
                segment_ids.append(0)

                for t in auxiliary_tokens:
                    tokens.append(t)
                    segment_ids.append(0)

                tokens.append(self.SEP)   #辅助句结束标记
                segment_ids.append(0)

                for t in text_tokens:
                    tokens.append(t)
                    if self.use_token_type:
                        segment_ids.append(1)
                    else:
                        segment_ids.append(0)

                tokens.append(self.SEP)      #原始文本结束标记
                if self.use_token_type:
                    segment_ids.append(1)
                else:
                    segment_ids.append(0)
                    
                #将token转为索引ID
                input_ids = tokenizer.convert_tokens_to_ids(tokens)  
                input_mask = [1] * len(input_ids)   #构建注意力掩码（1表示有效token，0表示填充token）

                #截断过长序列（不超过最大序列长度）
                input_ids = input_ids[:self.opt.max_seq_len]
                input_mask = input_mask[:self.opt.max_seq_len]
                segment_ids = segment_ids[:self.opt.max_seq_len]
                #填充短序列（不足最大长度时补0）
                while len(input_ids) < self.opt.max_seq_len:
                    input_ids.append(self.pad_id)
                    input_mask.append(0)
                    segment_ids.append(0)

                #转为numpy数组
                input_ids = np.asarray(input_ids, dtype='int64')
                input_mask = np.asarray(input_mask, dtype='int64')
                segment_ids = np.asarray(segment_ids, dtype='int64')

                #构建单条数据字典
                data = {
                    'text_bert_indices': input_ids,
                    'input_mask': input_mask,
                    'label': label,
                }
                all_data.append(data)
        self.data = all_data

    

    def sort_auxiliary(self, text_a, text_b):
        """
        按原始文本（text_a）中的词序对构建的辅助信息（text_b）排序，未在原始文本中出现的词保留其在text_b中的原始顺序，放在最后
        :param text_a: 原始文本
        :param text_b: 已经构建好的辅助信息列表 (target + category + auxiliary + opinions)
        :return: 排序后的辅助信息（字符串）
        """
        # ---------------------------------------------
        # >>>: 1.统一 word-level排序
        # ---------------------------------------------
        #统一处理text_b：若为字符串则按空格分词，若为列表则直接使用
        if isinstance(text_b, str):
            text_b = text_b.split()      #辅助信息转为词级列表

        text_a_words = text_a.strip().lower().split()    #原文分词（lower + split）用于排序对齐
        aux_words = [w.lower() for w in text_b]          #辅助词统一小写（保持 word-level ）

        # ---------------------------------------------
        # >>>: 2.计算每个辅助词在原文中的位置
        # ---------------------------------------------
        positions = []
        for w in aux_words:
            if w in text_a_words:
                pos = text_a_words.index(w)        #在原文中的首次出现位置
            else:
                pos = 10**9                        #不在原文 → 排在后面
            positions.append(pos)
 
        # ---------------------------------------------
        # >>>:3. 稳定排序（保证未出现辅助词保持原顺序）
        # ---------------------------------------------
        indexed = list(zip(aux_words, positions))
        indexed_sorted = sorted(indexed, key=lambda x: x[1])    # Python sorted 保证稳定性
        sorted_aux_words = [w for w, _ in indexed_sorted]
    
        # ---------------------------------------------
        # >>>: 4.返回字符串（适配 tokenizer.tokenize）
        # ---------------------------------------------
        return ' '.join(sorted_aux_words)
       
    
  
    def convert_to_unicode(self,text):
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

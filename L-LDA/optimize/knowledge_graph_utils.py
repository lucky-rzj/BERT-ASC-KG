# =================================================================================
# 知识图谱处理工具
# =================================================================================
# 核心功能：从 ConceptNet 全量数据中筛选领域相关的知识图谱子图，筛选规则包括：
# 1. 置信度过滤（仅保留置信度≥min_confidence的关系）
# 2. 关系类型限制（仅保留 RelatedTo/IsA/HasA 三种核心关系）
# 3. 语言过滤（仅保留英文概念，格式为 /c/en/...）
# 4. 领域约束（仅保留与领域核心词相关的关系）
# 5. 自关联去除（排除词与自身的关系）
# 6. 数量限制（每个词最多保留topk个关联词）
# =================================================================================

import csv    #用于解析 CSV 格式的 ConceptNet 数据
import gzip   #用于读取 gzip 压缩文件（ConceptNet 数据为 .csv.gz 格式）
import pickle as pk
import json  
import argparse 
import os
from collections import defaultdict
from collections import Counter
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer   #用于词形还原
import nltk
nltk.data.path.append('/hy-tmp/BERT-ASC-main/nltk_data') 
import re

 
# ---------------------------- 核心工具类：ConceptNet 处理器 -------------------------------
class ConceptNetProcessor:
    """
    解析本地 ConceptNet 全量数据，筛选领域相关的高置信度关系，构建领域知识图谱子图
    输出格式：{核心词: [(关联词1, 置信度1), (关联词2, 置信度2), ...]}
    """
    def __init__(self, conceptnet_path, domain_terms, min_confidence=0.7,relations=None, topk=30 ):  
        """
        类初始化：配置参数并自动构建领域关联词表
        :param conceptnet_path: ConceptNet 全量数据文件路径（.csv.gz 格式）
        :param domain_terms: 领域核心词列表（如餐饮领域：['food', 'price', 'service']）
        :param min_confidence: 最小置信度阈值（过滤低可靠性关系，默认0.7）
        :param relations: 要保留的关系类型（默认 ['/r/RelatedTo', '/r/IsA', '/r/HasA']）
        :param topk: 每个词最多保留的关联词数量（避免子图过大，默认30）
        """
        self.conceptnet_path = conceptnet_path    # ConceptNet 数据文件路径
        self.min_confidence = min_confidence      #置信度阈值
        self.relations = ['/r/RelatedTo', '/r/IsA', '/r/HasA']
        self.topk = topk

        #初始化词形还原器
        lemm = WordNetLemmatizer()
        self.lemmatize = lemm.lemmatize
        
        #处理领域核心词：转为小写 + 词形还原，并用集合存储（去重+快速查询）
        self.domain_terms = {lemm.lemmatize(w.lower()) for w in domain_terms}  
        self.related_words = defaultdict(list)   #初始化关联词存储结构：存储格式：{核心词: [(关联词1, 置信度1), (关联词2, 置信度2), ...]}
        self._build_related_words()             #初始化时，自动构建关联词表


    def _build_related_words(self):
        """
        私有方法：解析 ConceptNet 全量数据，按筛选规则构建领域关联词表
        核心流程：读取数据 → 多维度过滤 → 关系去重与排序 → 结果存储
        """
        print(f"[INFO] Loading ConceptNet from {self.conceptnet_path} ...")
        row_count = 0    #行计数器
        skipped_reasons = Counter()   #记录跳过原因计数
        
        #打开 gzip 压缩的 CSV 文件（rt 模式：文本模式读取，支持中文编码）
        with gzip.open(self.conceptnet_path, 'rt', encoding='utf-8', newline='') as f:   
            reader = csv.reader(f, delimiter='\t')   #创建 CSV 读取器，指定分隔符为制表符
            #遍历每一行
            for row in tqdm(reader, desc="Parsing ConceptNet", unit="rows"):   
                row_count += 1   #每处理一行+1

                # 过滤规则1：跳过不完整的行
                if len(row) < 5:
                    skipped_reasons['incomplete_row'] += 1
                    continue
                rel, start, end = row[1], row[2], row[3]  #解析行数据,ConceptNet CSV格式：[id, 关系, 起点, 终点, 置信度, ...]
                
                # 过滤规则2：解析并过滤低置信度关系
                try:
                    if row[4].startswith('{'):     #JSON格式
                        meta = json.loads(row[4])
                        confidence = float(meta.get('weight', 0.0))    #提取 weight 字段作为置信度
                    else:     #纯数字格式
                        confidence = float(row[4])   #取第4列
                except Exception:
                    skipped_reasons['invalid_confidence'] += 1
                    continue
                    
                # 过滤规则3：仅保留英文概念
                start_word = self._extract_word(start)
                end_word = self._extract_word(end)
                if not (start_word and end_word):
                    skipped_reasons['non_english'] += 1  
                    continue   #跳过非英文实体

                # 过滤规则4：仅保留置信度≥最小阈值的关系
                if confidence < self.min_confidence:
                    skipped_reasons['low_confidence'] += 1
                    continue

                # 过滤规则5：仅保留指定类型的关系
                if rel not in self.relations:
                    skipped_reasons['other_relation'] += 1
                    continue

                # 过滤规则6：仅保留与领域核心词相关的关系
                if start_word not in self.domain_terms:
                    skipped_reasons['not_domain_related'] += 1
                    continue

                # 过滤规则7：去除自关联关系
                if start_word == end_word:
                    skipped_reasons['self_relation'] += 1
                    continue

                # 过滤规则8：跳过数字、特殊字符
                if not re.match(r'^[a-z]+$', end_word):
                    continue

                # 9.所有过滤规则通过：双向添加关系（ConceptNet 关系是双向的，如 A→B 等价于 B→A）
                self.related_words[start_word].append((end_word, confidence))
                self.related_words[end_word].append((start_word, confidence))

                
        #置信度去重 + 排序 （去重逻辑：同一关联词保留最高置信度；排序逻辑：按置信度降序排列，取前topk个）
        for word, pairs in self.related_words.items():
            merged = defaultdict(float)
            for w, c in pairs:
                merged[w] = max(merged[w], c)
            self.related_words[word] = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:self.topk]

   
        #输出统计，打印总处理行数和结果统计
        print(f"\n[INFO] Total rows processed: {row_count:,}")
        total_skipped = sum(skipped_reasons.values())   #计算跳过的总行数
        print(f"[INFO] Total skipped: {total_skipped:,}")
        #按跳过次数降序输出每个原因的统计
        for reason, count in skipped_reasons.most_common():
            print(f"   - {reason:20s}: {count:,}")
        print(f"[INFO] Domain-related words extracted: {len(self.related_words):,}")    #输出最终筛选出的领域相关词数量

            
    def _extract_word(self, node):
        """
        私有辅助方法：从 ConceptNet 概念节点格式中提取英文词
        ConceptNet 节点格式：/c/语言代码/词（如 /c/en/food → 英文词 food）
        :param node: ConceptNet 节点字符串（如 /c/en/food）
        :return: 提取的英文词（小写），非英文节点返回 None
        """
        if node.startswith('/c/en/'):   #仅处理英文节点（/c/en/表示英文概念）
            word = node.split('/')[3].lower()     #分割路径并取第4个元素（如'/c/en/food' -> ['', 'c', 'en', 'food'] -> 'food'）
            return self.lemmatize(word) 
        return None   #非英文节点返回None

    
    def get_related(self, word, top_n=10):
        """
        公共方法：获取指定词的 Top N 关联词及置信度（供外部调用）
        :param word: 目标词（如 food）
        :param top_n: 要返回的关联词数量（默认10）
        :return: 关联词列表，格式 [(关联词1, 置信度1), ...]
        """
        return self.related_words.get(word, [])[:top_n]    #从相关词表中获取指定词的关联词，默认返回前10个



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='构建领域知识图谱子图')  
    parser.add_argument('--dataset', default='semeval', choices=['semeval', 'sentihood'],  help='指定数据集名称 (semeval 或 sentihood)')  
    parser.add_argument(
        '--conceptnet-path',
        default='/hy-tmp/BERT-ASC-main/conceptnet-assertions-5.7.0.csv.gz',
        help='ConceptNet数据文件路径'
    )
    parser.add_argument('--min-confidence', type=float, default=0.7, help='最小置信度阈值 (默认: 0.7)') 
    parser.add_argument('--topk', type=int, default=30, help='每个词最多保留的关联词数')
    args = parser.parse_args()  


    #根据数据集选择领域核心词
    if args.dataset == 'semeval':
        domain_terms = ['food', 'price', 'service', 'ambience']       #餐饮领域
    else:  
        domain_terms = ['price', 'safety', 'general','transit-location']     #交通领域

        
    #初始化 ConceptNet 处理器
    processor = ConceptNetProcessor(
        conceptnet_path=args.conceptnet_path,   # ConceptNet 数据路径
        domain_terms=domain_terms,     #领域核心词
        min_confidence=args.min_confidence,    #置信度阈值
        topk=args.topk     #每个词的最大关联词数
    )  

    
    #保存构建好的领域知识图谱子图
    save_path = f'../../datasets/{args.dataset}/conceptnet_domain_subgraph.pk'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)    #自动创建不存在的目录
    pk.dump(processor.related_words, open(save_path, 'wb'))
    print(f"Domain subgraph saved to: {save_path}")

   
# ========================================================
# 生成句法候选词（增强鲁棒性版本）
# ========================================================

import os
os.environ["CORENLP_HOME"] = "/hy-tmp/BERT-ASC-main/stanford-corenlp-4.5.10"   #指定 Stanford CoreNLP 的路径
import nltk
nltk.data.path.append("/hy-tmp/BERT-ASC-main/nltk_data")
import json
import argparse
from nltk.corpus import stopwords
from nltk.parse.corenlp import CoreNLPDependencyParser
from tqdm import tqdm
import numpy as np
import logging


# ------------------- 停用词集合化 -------------------
stops = set(stopwords.words('english'))
stops.update([
    'us', "didnt",'im','couldnt','even','shouldnt','ive','make','today','feel','sometimes','ive',
    'whatever','never','although','anyway','get','always','usually','want','go','would','one',
    'location - 1 -', 'location - 2 -'
])


# ------------------------ logger ------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# =================================================================
# 单句 enhancedDependencies 解析（SemEval）
# =================================================================
def get_parse(ann):  
    """ 从CoreNLPClient的注解结果中提取依存关系 """
    sentence = ann.sentence[0]   #取第一句（默认处理单句）
    dependency_parse = sentence.enhancedDependencies    #获取句子的增强依存关系
    #构建token索引到词的映射
    token_dict = {}   
    for i in range(0, len(sentence.token)):
        token_dict[sentence.token[i].tokenEndIndex] = sentence.token[i].word   #键：token结束索引，值：对应的词      

    #提取依存关系三元组（源词、关系类型、目标词）
    list_dep = []
    #遍历每个依存关系边
    for i in range(0, len(dependency_parse.edge)):
        source_node = dependency_parse.edge[i].source   #源节点（依存关系的起点）
        source_name = token_dict[source_node]    #源节点对应的词
        target_node = dependency_parse.edge[i].target  #目标节点（依存关系的终点）
        target_name = token_dict[target_node]    #目标节点对应的词
        dep = dependency_parse.edge[i].dep   #依存关系类型
        list_dep.append((source_name, dep, target_name))  #存入三元组列表
    return list_dep  #返回依存关系列表



# ===============================================
# 句法候选词提取函数----semeval
# ===============================================
TARGET_REL_SEMEVAL = {'nsubj', 'amod', 'advmod', 'ccomp', 'compound'}   #依存关系
def extract_opinion_from_dp_semeval(dp_rel, auxiliary_raw, text, all_categories):
    opinions = set()    #存储当前的句法候选
    text_words = [tok.lower() for tok in text.split()]
    aux = [w.lower() for w in auxiliary_raw]  
    all_cats_lower = {c.lower() for c in all_categories}
    for w in aux:
        if w not in text_words:
            continue
        #遍历所有依存关系
        for (l, rel, r) in dp_rel:
            if rel in TARGET_REL_SEMEVAL:
                if w == l:
                    opinions.add(r.lower()) 
                elif w == r:
                    opinions.add(l.lower())
    opinions = {
        o for o in opinions
        if o in text_words          # 必须真的出现在文本里
        and o not in stops          # 去掉停用词
        and o not in aux            # 去掉语义候选词本身
        and o not in all_cats_lower # 去掉所有类别名称（service/price/...）
    }

    # 排序
    return sort_auxiliary(text, list(opinions))

    

# ===============================================
# 句法候选词提取函数----sentihood
# ===============================================
TARGET_REL_SENTIHOOD = {'nsubj', 'amod', 'advmod', 'ccomp', 'compound'}
def extract_opinion_from_dp_sentihood(dp_rel, auxiliary_raw, text, all_categories, target=None):
    opinions = set()
    text_words = [tok.lower() for tok in text.split()]
    aux = [w.lower() for w in auxiliary_raw]
    all_cats_lower = {c.lower() for c in all_categories}
    tgt = target.lower() if target else None

    for w in aux:
        if w not in text_words:
            continue
        for (l, rel, r) in dp_rel:
            if rel in TARGET_REL_SENTIHOOD:
                if w == l:
                    opinions.add(r.lower())
                elif w == r:
                    opinions.add(l.lower())
    # 过滤：停用词、aux、所有类别名、target、自身不存在的词           
    opinions = {
        o for o in opinions
        if o in text_words
        and o not in stops
        and o not in aux
        and o not in all_cats_lower
        and (tgt is None or o != tgt)
    }
    
    return sort_auxiliary(text, list(opinions))
    

    
# ====================================================
# 排序函数---按 text_a 的单词顺序对 text_b 排序
# ====================================================
def sort_auxiliary(text_a, text_b):
    """
    按 text_a 的单词顺序对 text_b 排序：
    1. text_b 中存在于 text_a 的词 → 按 text_a 中的出现顺序排列
    2. text_b 中不存在于 text_a 的词 → 保留原始顺序，放结果末尾
    3. 保留 text_b 中的重复词
    """
    text_a_tokens = [w.lower() for w in text_a.split()]    #原始文本分词（基准顺序）
    #构建词→索引映射（O(n) 复杂度，比多次 text_a.index(w) 高效）
    word_to_index = {word: idx for idx, word in enumerate(text_a_tokens)}
    
    # 排序规则：
    # - 存在于 text_a 的词 → 用其在 text_a 中的索引排序
    # - 新增词 → 用“无穷大”作为索引（确保放末尾），同时保留原始顺序（sorted 是稳定排序）
    sorted_text_b = sorted(
        text_b,
        key=lambda w: word_to_index.get(w.lower(), float('inf'))
    )

    return sorted_text_b


    
# =======================================================
# semeval 主流程
# =======================================================
def extract_opinion_words(dataset='semeval'):
    """ 为semeval数据集提取句法候选词（使用 enhancedDependencies--增强依存关系）"""
    from stanfordnlp.server import CoreNLPClient
    #启动CoreNLP服务（指定需要的注解器：分词、分句、词性标注、依存句法分析）
    with CoreNLPClient(
        annotators=['tokenize','ssplit','pos','depparse'],
        timeout=60000,
        memory='16G',
        start_server=True    #自动启动服务器
    ) as client:
        
        #定义数据集对应的子文件（训练集/测试集/验证集）和方面类别
        files = {'semeval':['train','test']}
        for subfile in files[dataset]:
            #上阶段的 JSON 文件
            input_path = f'../../datasets/{dataset}/bert_{subfile}_with_aux.json'
            if not os.path.exists(input_path):
                logger.error(f"输入文件不存在：{input_path}")
                continue
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            new_data = []   #存储处理后的结果
            #遍历每个样本
            for d in tqdm(data):
                text = d['text']     #提取文本
                aspects = d['aspect']  #提取方面信息
                #----------------- 对文本进行句法依存解析 ---------------
                try:
                    ann = client.annotate(text)    #对文本进行句法分析：获取分词、词性、依存关系等注解结果
                    dp_rel = get_parse(ann)    #从注解结果中提取依存关系三元组（源词、关系类型、目标词）
                except Exception as e:
                    logger.error(f"[ERROR] 解析失败: {e}")
                    dp_rel = []

                    
                text_lower = text.lower()
                all_categories = {asp.get("category", "").lower() for asp in aspects}
                #遍历方面信息
                for asp in aspects:
                    category = asp.get("category", "")
                    polarity = asp.get("polarity", "")
                    auxiliary = asp.get("auxiliary", [])

                    if not dp_rel:
                        asp['opinions'] = []
                    else:
                        aux = auxiliary.copy()
                        if category.lower() in text_lower:
                            aux.append(category.lower())
                        asp["opinions"] = extract_opinion_from_dp_semeval(dp_rel, aux, text, all_categories)   #提取观点词  
                        
                    asp["opinions_for_kg"] = asp["opinions"].copy()
                    asp["validated_auxiliary_placeholder"] = ""

                new_data.append({'text': text, 'aspect': aspects})

                
            # ------------------ 保存结果 --------------------------
            output_path = os.path.join(os.path.dirname(input_path),f"bert_{subfile}_with_opinions.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=3, ensure_ascii=False)
            
            # -------------------------- 输出统计信息 -------------------------------
            #构建统计字典：键为类别名，值为该类别的观点词总数
            opinion_stats = {
                asp["category"]: len(asp.get("opinions", []))
                for d in new_data for asp in d["aspect"] if asp.get("opinions")
            }
            if opinion_stats:
                #计算每个类别的平均观点词数
                avg_per_cat = {k: np.mean([v for kk, v in opinion_stats.items() if kk == k]) for k in set(opinion_stats.keys())}
                logger.info("[类别平均观点词数] " + ", ".join([f"{k}:{v:.2f}" for k, v in avg_per_cat.items()]))

            logger.info(f"[DONE] {dataset} {subfile} 处理完成，结果保存至 {output_path}")
           
       

# ======================================================
# sentihood 主流程
# ======================================================
def extract_opinion_words_sentihood(dataset='sentihood'):
    """ 为sentihood数据集提取句法候选词（使用 basic dependency triples--基本依存） """
    parser = CoreNLPDependencyParser(url='http://localhost:9000')    #连接本地启动的CoreNLP服务
    
    files = {'sentihood':['train','test','dev']}
    for subfile in files[dataset]:
        input_path = f'../../datasets/{dataset}/bert_{subfile}_with_aux.json'
        if not os.path.exists(input_path):
            logger.error(f"输入文件不存在：{input_path}")
            continue
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        new_data = []   #存储处理后的新数据
        for d in tqdm(data):
            text = d['text']
            aspects = d['aspect']
            text_lower = text.lower()
            
            #------------- basic dependency parse 对文本进行句法解析 --------   
            try:
                dp_tree, = parser.raw_parse(text)   #对文本进行句法分析，生成依存树
                triples = dp_tree.triples()    #提取句法依存三元组
                #将依存树转成 (词, 关系, 词) 的三元组列表 dp_rel
                dp_rel = [(l[0].lower(), rel, r[0].lower()) for (l, rel, r) in triples]   
            except Exception as e:
                logger.warning(f"[WARN] 解析失败: {e}")
                dp_rel = []
                
            all_categories = {asp.get("category", "").lower() for asp in aspects}
            
            for asp in aspects:
                category = asp.get("category","")
                polarity = asp.get("polarity","")
                auxiliary = asp.get("auxiliary",[])
                target = asp.get("target", "").lower()  # sentihood 专属字段

                if not dp_rel:
                    asp['opinions'] = []
                else:
                    aux = auxiliary.copy()
                    if target and target in text_lower:
                        aux.append(category.lower())
                    asp["opinions"] = extract_opinion_from_dp_sentihood(dp_rel,aux,text,all_categories,target=target)
                    
                asp["opinions_for_kg"] = asp["opinions"].copy()
                asp["validated_auxiliary_placeholder"] = ""
                
            new_data.append({'text': text, 'aspect': aspects})


        # ------------------ 保存结果 --------------------------
        output_path = os.path.join(os.path.dirname(input_path),f"bert_{subfile}_with_opinions.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=3, ensure_ascii=False)
            
        # -------------------------- 输出统计信息 -------------------------------
        #构建统计字典：键为类别名，值为该类别的观点词总数
        opinion_stats = {
            asp["category"]: len(asp.get("opinions", []))
            for d in new_data for asp in d["aspect"] if asp.get("opinions")
        }
        if opinion_stats:
            #计算每个类别的平均观点词数
            avg_per_cat = {k: np.mean([v for kk, v in opinion_stats.items() if kk == k]) for k in set(opinion_stats.keys())}
            logger.info("[类别平均观点词数] " + ", ".join([f"{k}:{v:.2f}" for k, v in avg_per_cat.items()]))

        logger.info(f"[DONE] {dataset} {subfile} 处理完成，结果保存至 {output_path}")
           

        
def main():
    parser = argparse.ArgumentParser(description='为SemEval/Sentihood数据集生成句法候选词') 
    parser.add_argument('--dataset', default='semeval',type=str, choices=['semeval','sentihood'],required=True)
    opt = parser.parse_args()

    
    if opt.dataset == 'semeval':
        extract_opinion_words(opt.dataset)
    else:
        extract_opinion_words_sentihood(opt.dataset)


if __name__ == '__main__':
    main()

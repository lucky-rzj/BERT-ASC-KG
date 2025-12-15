# ======================================================================================================
# 生成-过滤-验证闭环
# ======================================================================================================
# 功能:生成候选辅助句（深度模型 GPT-2） → 过滤（BLEU） → 验证（知识图谱） → 修正（KG-based） → 输出高质量辅助句
# 模块说明:
# generate_candidates():使用 GPT-2生成候选句（生成式模型动态生成）
# filter_and_validate():使用 BLEU 分数衡量句子与原句的相关度。调用知识图谱函数进行语义验证与修正。
# 数据存储模块：bert_test/train/dev_validated.json
# =======================================================================================================

import os
import sys
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
import nltk
nltk.data.path.append('/hy-tmp/BERT-ASC-main/nltk_data')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction   #从nltk导入BLEU评分函数和平滑方法
from transformers import GPT2LMHeadModel, GPT2Tokenizer    #从transformers库导入GPT-2模型和GPT-2分词器
import logging  
import time
from collections import defaultdict  #用于统计类别/极性数据
#from aux_scorer import AuxScorer
#scorer = AuxScorer("/hy-tmp/BERT-ASC-main/ASC_generating/aux_scorer_model")


# ------------------------日志配置 -----------------------
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)   


# ---------------------- 模块导入 --------------------------
sys.path.append(os.path.dirname(__file__))   #添加当前脚本所在目录到搜索路径
#导入知识图谱验相关的校验函数
try:  
    from knowledge_graph_utils_offline import (
        map_to_aspect_category,   #语义映射
        correct_semantic_bias,    #偏差修正
        disambiguate_ambiguous_sentence   #歧义消解
    )
    USE_KG = True   #如果导入成功，设置使用知识图谱的标志为True
    logger.info("[INFO] 已启用离线知识图谱验证模式（ConceptNet 哈希缓存）")
except Exception as e:
    USE_KG = False
    logger.warning(f"[WARN] 离线知识图谱模块加载失败: {e}，使用简化验证逻辑。")

    
# -------------------------- 模型加载 -------------------------------
MODEL_PATH = "/hy-tmp/BERT-ASC-main/ASC_generating/models/gpt2"   #GPT-2模型文件路径
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)   #从预训练模型路径加载GPT-2的分词器
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   #使用eos_token作为pad_token
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)     #从预训练模型路径加载GPT-2的语言模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
smoother = SmoothingFunction().method4  #初始化BLEU评分的平滑函数（使用方法4）



# --------------------------------------------------------------------
# (1) 生成候选辅助句
# ---------------------------------------------------------------------
def generate_candidates(original_sentence, target_aspect, num_candidates=3):
    """使用 GPT-2 生成多个辅助句候选"""
    #构建输入给GPT-2的文本提示，包含原句、目标方面
    # ---- Few-shot + 强约束 + 固定 examples（不使用 target_aspect） ----
    input_text = (
        "Sentence: The food was cold but the service was excellent.\n"
        "Aspect: food\n"
        "Auxiliary sentence: What is the sentiment about the food?\n\n"

        "Sentence: The room was clean but a bit noisy.\n"
        "Aspect: room\n"
        "Auxiliary sentence: How is the service described?\n\n"

        f"Sentence: {original_sentence}\n"
        f"Aspect: {target_aspect}\n"
        "Generate ONE short auxiliary question (less than 12 words) that helps analyze sentiment.\n"
        "ONLY output the question.\n"
        "Auxiliary sentence:"
    )

    #使用分词器对输入文本进行编码
    input_ids = tokenizer.encode(
        input_text,
        return_tensors="pt",   #返回PyTorch张量
        padding=True,      #启用填充
        truncation=True,   #启用截断
        max_length=512     #输入文本的最大长度限制为512个token
    ).to(device)   #将编码后的张量移动到模型所在的设备
    
    # -------------------- 使用GPT-2生成候选句 -------------------------
    with torch.no_grad():
        outputs = model.generate(
            input_ids,                 #输入的token ID序列
            max_new_tokens=50,         #生成的新token的最大数量
            num_return_sequences=3,    #返回3个不同的生成序列（候选句）
            temperature=0.6,           #控制生成的随机性
            top_k=50,                  #采样时，只从概率最高的50个token中选择
            top_p=0.8,                 #采样时，只从累积概率达到0.8的token集合中选择
            do_sample=True,            #启用随机采样
            pad_token_id=tokenizer.pad_token_id,    # pad_token的ID
            eos_token_id=tokenizer.eos_token_id,    # eos_token的ID
            repetition_penalty=1.2      #重复惩罚因子，减少生成文本中的重复
        )

    candidates = []   #存储生成的候选句
    
    #遍历每个生成的结果
    for output in outputs:   
        decoded = tokenizer.decode(output, skip_special_tokens=True)   #将生成的token ID序列解码为文本
        #从解码文本中提取辅助句部分
        if "Auxiliary sentence:" in decoded:
            candidate = decoded.split("Auxiliary sentence:")[-1].strip()   #如果包含提示词，则截取其后的内容作为候选句
        else:
            candidate = decoded.replace(input_text, "").strip()   #否则，移除输入的提示文本，剩余部分作为候选句
        if candidate:   
            candidates.append(candidate)   #如果候选句不为空，则添加到列表中   
            
    #确保返回指定数量的候选句
    while len(candidates) < num_candidates:    
        #如果生成的候选句不足，则用一个默认的辅助句填充
        candidates.append(f"What is the sentiment about {target_aspect}?")

    return candidates


    
# --------------------------------------------------
# (2) 过滤 + 语义验证
# --------------------------------------------------
def filter_and_validate(original_sentence, target_aspect, candidates,bleu_threshold=0.2, semantic_threshold=0.5):

    if not candidates:
        return None

    valid_candidates = []
    original_tokens = original_sentence.lower().split()

    for candidate in candidates:

        if len(candidate.split()) < 3:
            continue

        # ------------------ 初次 BLEU ------------------
        bleu = sentence_bleu(
            [original_tokens],
            candidate.lower().split(),
            smoothing_function=smoother
        )

        # 若 BLEU 太低 → 直接过滤
        if bleu < bleu_threshold:
            continue

        # KG --------------------------
        if USE_KG:
            try:
                # ========== 提取观点词 ==========
                tokens = candidate.split()
                opinion_word = None
                for tok in reversed(tokens):
                    t = tok.lower().strip(".,!?")
                    if t.isalpha() and len(t) > 2:
                        opinion_word = t
                        break
                if not opinion_word:
                    opinion_word = target_aspect

                entity = target_aspect

                semantic_score = map_to_aspect_category(entity, opinion_word, target_aspect)

                # KG 不通过 → 修正
                if semantic_score < semantic_threshold:
                    candidate = correct_semantic_bias(candidate, target_aspect)
                    candidate = disambiguate_ambiguous_sentence(candidate, target_aspect)

                    # ========== 修正后重新提取观点词 ==========
                    new_tokens = candidate.split()
                    new_opinion = None
                    for tok in reversed(new_tokens):
                        t = tok.lower().strip(".,!?")
                        if t.isalpha() and len(t) > 2:
                            new_opinion = t
                            break
                    if not new_opinion:
                        new_opinion = target_aspect

                    semantic_score = map_to_aspect_category(entity, new_opinion, target_aspect)

                # ========== 修正后重新计算 BLEU ==========
                bleu = sentence_bleu(
                    [original_tokens],
                    candidate.lower().split(),
                    smoothing_function=smoother
                )

            except Exception as e:
                print(f"[KG ERROR] {e}")
                semantic_score = 0.5

        else:
            semantic_score = 0.5

        # ------------ 最终得分（弱化KG影响） --------------
        #scorer_score = scorer.score(original_sentence, target_aspect, candidate)
        #combined_score = (0.4 * bleu+ 0.3 * semantic_score+ 0.3 * scorer_score)

        combined_score = 0.6 * bleu + 0.4 * semantic_score

        valid_candidates.append((candidate, combined_score))

    if valid_candidates:
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        return valid_candidates[0][0]

    return None


                            
# --------------------------------------------------------------------
# (3) 数据集批量处理
# --------------------------------------------------------------------
def process_dataset(dataset):
    """处理数据集，将最优辅助句融入数据集，生成并替换固定格式的辅助句"""
    files = {'semeval': ['train', 'test'], 'sentihood': ['train', 'test', 'dev']}
    base_dir = "/hy-tmp/BERT-ASC-main/datasets"
    #遍历每个文件拆分
    for split in files[dataset]:   
        input_path = f"{base_dir}/{dataset}/bert_{split}.json"   #输入文件路径
        output_path = f"{base_dir}/{dataset}/bert_{split}_validated.json"      #输出文件路径
        if not os.path.exists(input_path):
            logger.error(f"[ERROR] 输入文件不存在: {input_path}")
            continue
        with open(input_path, 'r', encoding='utf-8') as f:   #加载输入的JSON数据集
            data = json.load(f)
        logger.warning(f"[LOAD] {dataset}-{split}: {len(data)} samples loaded from {input_path}")   

        new_data = []   #存储处理后的新数据
        
        total_aux_len, valid_count = 0, 0   #统计变量（辅助句总长度、有效辅助句数量）
        #用于按方面类别和情感极性进行统计的字典
        category_stats, polarity_stats = defaultdict(list), defaultdict(lambda: {"count": 0, "valid": 0})   
        start_time = time.time()   #记录当前时间
        #遍历每个数据项
        for idx, item in enumerate(tqdm(data, desc=f"[{dataset}] {split} 验证辅助句", dynamic_ncols=True)):  
            text = item['text']   #获取原文本
            new_aspects = []      #存储方面信息
            #遍历每个方面信息
            for aspect in item['aspect']:    
                category = aspect['category']    #获取方面类别
                polarity = aspect.get('polarity', 'none')  #获取情感极性
                polarity_stats[polarity]["count"] += 1    #更新该极性的总计数
               #处理辅助词列表，确保其为列表类型
                auxiliary = aspect.get('auxiliary', [])
                if isinstance(auxiliary, str):
                    auxiliary = [auxiliary] if auxiliary else []
                #处理观点词列表，确保其为列表类型
                opinions = aspect.get('opinions', [])
                if isinstance(opinions, str):
                    opinions = [opinions] if opinions else []

                candidates = generate_candidates(text, category)   #生成候选辅助句
                validated_aux = filter_and_validate(text, category, candidates)    #最优辅助句   
                #如果未生成有效的最优辅助句，则使用备用逻辑生成一个
                if not validated_aux:  
                    joined = ' '.join(aspect.get('auxiliary', []) + aspect.get('opinions', []))  #拼接辅助词和观点词
                    validated_aux = f"What is the sentiment of {category} {joined}?"   #生成备用辅助句
                else:
                    valid_count += 1
                    polarity_stats[polarity]["valid"] += 1

                total_aux_len += len(validated_aux.split())    #累计辅助句长度
                category_stats[category].append(len(validated_aux.split()))  #记录各类别辅助句长度

                
                # ============================================================
                # ✅ ✅ 根据数据集结构分别构造 JSON aspect 字段
                # ============================================================
                if dataset == "semeval":
                    formatted_aspect = {
                        "category": category,
                        "polarity": polarity,
                        "auxiliary": auxiliary,
                        "opinions": opinions,
                        "validated_auxiliary": validated_aux
                    }
                else:  
                    formatted_aspect = {
                        "category": category,
                        "polarity": polarity,
                        "auxiliary": auxiliary,
                        "target": aspect.get("target"),                
                        "opinions": opinions,
                        "validated_auxiliary": validated_aux
                    }
                new_aspects.append(formatted_aspect)   #将处理后的方面添加到列表
            new_data.append({'text': text, 'aspect': new_aspects})  #将处理后的文本和方面信息添加到新数据列表

            #每100条输出进度时间
            if (idx + 1) % 100 == 0:
                avg_time = (time.time() - start_time) / (idx + 1)
                logger.warning(f"已验证 {idx+1} 条，平均耗时 {avg_time:.2f}s/样本")
                
        #将处理后的新数据保存到输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=3, ensure_ascii=False)

            
        # ------------------------ 统计信息 ------------------------- 
        avg_len = total_aux_len / max(valid_count, 1)    #平均辅助句长度
        logger.info(f"[STATS] 数据集: {dataset} | 划分: {split}")
        logger.info(f" - 成功生成辅助句数量: {valid_count}/{len(data)}")
        logger.info(f" - 平均辅助句长度: {avg_len:.2f} 个词")
        #按极性统计数据
        for pol, stat in polarity_stats.items():   
            rate = stat["valid"] / max(stat["count"], 1)
            logger.warning(f"  - {pol}: {rate*100:.1f}% ({stat['valid']}/{stat['count']})")

        print(f"[DONE] {dataset} {split} validated auxiliaries saved to {output_path}")   



# --------------------------------------------------------------------
# (4) 主函数入口
# --------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="生成 + 验证辅助句")
    parser.add_argument('--dataset', required=True, choices=['semeval', 'sentihood'])
    args = parser.parse_args()
    
    print(f"[INFO] 使用数据集: {args.dataset}")
    #调用数据集处理函数，开始处理指定的数据集
    process_dataset(args.dataset)

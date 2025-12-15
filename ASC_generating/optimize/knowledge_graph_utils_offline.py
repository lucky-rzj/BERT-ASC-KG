# ======================================================================================================
# 知识图谱语义校验-----本地 ConceptNet 缓存 + 语义匹配 + 修正 + 消歧
# ======================================================================================================
# 功能：语义映射、偏差修正、歧义消解
# 模块说明：
# 本模块通过 ConceptNet + spaCy + nltk 验证机制，对生成辅助句进行语义一致性检查，确保辅助句与目标方面一致。
# 1. map_to_aspect_category():语义映射模块,判断实体与观点词是否与目标Aspect语义一致
# 2. correct_semantic_bias():语义偏差修正模块,检测句子中观点词是否属于目标Aspect语义范畴，若不匹配则替换
# 3. disambiguate_ambiguous_sentence():歧义消解模块,针对多义词（如 “heavy”、“good”、“high”）进行语义消歧
# 依赖项: spacy, en_core_web_sm, ConceptNet API
# ======================================================================================================

import os
import re
import pickle
import logging
import nltk
from nltk.tokenize import word_tokenize 



# ----------------------------基础配置-----------------------------
nltk.data.path.append('/hy-tmp/BERT-ASC-main/nltk_data')     #指定NLTK数据路径 
CACHE_INDEX_PATH = "/hy-tmp/BERT-ASC-main/ASC_generating/data/concept_index_cache.pkl"  #本地知识图谱索引文件路径
local_runtime_cache = {}    #内存级缓存


# ----------------------------日志设置----------------------------- 
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)



# ==========================================================
# 缓存加载----加载 ConceptNet 本地缓存
# ==========================================================
concept_index = {}    #存放本地知识图谱索引
if os.path.exists(CACHE_INDEX_PATH):
    try:
        with open(CACHE_INDEX_PATH, "rb") as f:
            concept_index = pickle.load(f)
        logger.info(f"[LOCAL INDEX] 已加载 ConceptNet 离线缓存，共 {len(concept_index):,} 个实体。")
    except Exception as e:
        logger.error(f"[LOCAL INDEX] 加载 concept_index_cache.pkl 失败: {e}")
else:
    logger.error(f"[LOCAL INDEX] 未找到 concept_index_cache.pkl，请先执行 build_conceptnet_cache_auto_hash.py 构建。")


    
# =================================================================
# 小领域安全词典（白名单 + 黑名单）
# =================================================================
DOMAIN_OPINION = {
    "food": ["fresh", "tasty", "delicious", "spicy", "good"],
    "service": ["friendly", "slow", "helpful", "rude", "nice"],
    "price": ["expensive", "cheap", "affordable", "reasonable"],
    "ambience": ["cozy", "clean", "noisy", "quiet", "comfortable"]
}

BAD_KG_WORDS = {
    "organism","radiation","chemical","physics","biology","protein",
    "device","tool","machine","muscle","system","energy","cell"
}


# ===========================================================
# 基础 aspect semantics（会进行安全扩展）
# ===========================================================
aspect_semantics = {
    "food": ["food", "taste", "flavor", "cuisine", "fresh", "tasty", "dish", "ingredient"],
    "service": ["service", "staff", "waiter", "helpful", "friendly", "attentive", "efficient"],
    "price": ["price", "cost", "expensive", "cheap", "affordable", "value", "charge"],
    "ambience": ["ambience", "environment", "atmosphere", "clean", "cozy", "noisy", "decor"]
}


# ============================================================
# 自动扩展 aspect_semantics（基于 ConceptNet）
# ============================================================
def expand_aspect_semantics():
    logger.info("[ASPECT EXPAND SAFE] 启动小领域安全扩展（限制噪声、数量）")

    MAX_EXPAND = 10

    for aspect, seeds in aspect_semantics.items():
        extended = set(seeds)

        for seed in seeds:
            edges = concept_index.get(seed, [])
            count = 0

            for e in edges:
                if e[0] != "/r/RelatedTo":
                    continue

                label = e[1].split("/")[-1].lower()

                if not label.isalpha():
                    continue
                if label in BAD_KG_WORDS:
                    continue
                if len(label) > 15:
                    continue
                if float(e[2]) < 1.0:
                    continue

                extended.add(label)
                count += 1
                if count >= MAX_EXPAND:
                    break

        aspect_semantics[aspect] = list(extended)
        logger.info(f"[ASPECT EXPAND SAFE] {aspect}: 扩展后 {len(extended)} 词")

expand_aspect_semantics()



# ====================================================================================
# 从 ConceptNet 查询语义关系 ———— 使用 local_runtime_cache + concept_index
# ====================================================================================
def get_semantic_relations(entity, relation_type="RelatedTo", limit=3, timeout=10):
    """
    快速查询 ConceptNet 实体的语义关系，内存缓存  → 本地索引 
    查询顺序：
        ① 优先，查内存缓存---local_runtime_cache,（存储本次程序执行期间已经查询过的实体（entity）的 ConceptNet 结果）
        ② 其次，查本地索引---concept_index_cache.pkl,  预加载的本地知识图谱索引
    """
    #输入合法性检查：如果实体为空或非字符串类型，直接返回空列表
    if not entity or not isinstance(entity, str):
        return []
    #实体标准化：转为小写，替换空格为下划线，移除非字母数字和下划线的字符
    entity = re.sub(r'[^a-z_]', '', entity.strip().lower().replace(" ", "_"))
    #构建缓存键：由实体和关系类型组成
    cache_key = f"{entity}::{relation_type}"

    # ------------------- 内存缓存 -------------------
    #如果缓存键已存在于内存缓存中，直接返回对应的结果
    if cache_key in local_runtime_cache:
        logger.debug(f"[KG-CACHE] 命中内存缓存: {cache_key}")
        return local_runtime_cache[cache_key]

    # ----------------- 本地 ConceptNet 实体哈希索引 -----------------
    #如果实体存在于本地知识图谱索引中
    if entity in concept_index:
        edges = concept_index[entity]   #从索引中获取该实体的所有边（edges）
        #筛选出符合关系类型的边，并按照权重排序后取前limit条
        results = [
            {"end": {"label": e[1].split("/")[-1]}, "weight": float(e[2])}
            for e in edges if e[0] == f"/r/{relation_type}"
        ][:limit]
        logger.debug(f"[KG-INDEX] 实体={entity}, 关系={relation_type}, 返回={len(results)}")
        #将结果存入内存缓存和磁盘缓存
        local_runtime_cache[cache_key] = results 
        return results
    return []


    
# ======================================================
# 语义映射（安全版）
# ======================================================
def map_to_aspect_category(entity, opinion_word, target_aspect):
    """映射实体和观点词到目标方面，计算匹配度"""
    logger.info(f"[KG-MAP] aspect={target_aspect} | entity={entity} | opinion={opinion_word}")
    
    target_aspect = target_aspect.lower()
    target_semantics = aspect_semantics.get(target_aspect, [])   


    # 白名单：GPT 正常生成的观点词 → 直接匹配
    if opinion_word in DOMAIN_OPINION.get(target_aspect, []):
        return 1.0

    # 黑名单：百科噪声 → 避免修正
    if opinion_word in BAD_KG_WORDS:
        return 0.4

    # KG 辅助判断
    edges = get_semantic_relations(opinion_word, "RelatedTo")
    cats = [e["end"]["label"].lower() for e in edges]
    
    if any(c in target_semantics for c in cats):
        return 0.8

    return 0.4   #默认中等分，避免过修正


    
# ========================
# 偏差修正
# ========================
def correct_semantic_bias(candidate_sentence, target_aspect):
    """修正辅助句的语义偏差，替换不匹配的观点词"""
    tokens = word_tokenize(candidate_sentence.lower())
    if len(tokens) < 3:
        return candidate_sentence

    # 寻找简单结构 X is Y
    entity = opinion = None
    for i, tok in enumerate(tokens):
        if tok in ["is","are","feels","seems"] and 0 < i < len(tokens)-1:
            entity = tokens[i-1]
            opinion = tokens[i+1]
            break

    if not opinion:
        return candidate_sentence

    # 白名单保护
    if opinion in DOMAIN_OPINION.get(target_aspect, []):
        return candidate_sentence

    # KG 判断
    score = map_to_aspect_category(entity, opinion, target_aspect)

    if score >= 0.5:
        return candidate_sentence

    # 修正：使用小领域词典
    replacement = DOMAIN_OPINION[target_aspect][0]
    return candidate_sentence.replace(opinion, replacement, 1)




# =====================
# 歧义消解
# =====================
def disambiguate_ambiguous_sentence(candidate_sentence, target_aspect):
    """对含歧义词的句子进行消歧，明确指向目标方面"""
    ambiguous = {"good","bad","great","high","low","strong","weak"}
    tokens = word_tokenize(candidate_sentence.lower())
    opinion_word = next((t for t in tokens if t in ambiguous), None)

    if not opinion_word:
        return candidate_sentence
    logger.info(f"检测到歧义观点词: {opinion_word}，开始消歧")
    
    relations = ["RelatedTo", "Synonym", "IsA"]
    confidence = {}   #存储每个语义范畴的置信度
    #遍历每种关系类型
    for rel in relations:
        edges = get_semantic_relations(opinion_word, rel)
        for edge in edges:
            category = edge["end"]["label"].lower()
            if category in BAD_KG_WORDS:
                continue
            confidence[category] = confidence.get(category, 0) + edge["weight"]

    
    target_semantics = aspect_semantics.get(target_aspect, [])
    best_category = max(target_semantics, key=lambda s: confidence.get(s, 0)) 
    score = confidence.get(best_category, 0)

    logger.info(
        f"[KG-DISAMBIG-RESULT] aspect={target_aspect} | ambiguous={opinion_word} "
        f"| best_category={best_category} | confidence={confidence:.3f}"
    )

    if score < 0.5:   # KG 强 signal 才修正
        return candidate_sentence

    if best_category in candidate_sentence.lower():
        logger.info("[KG-DISAMBIG] 句子中已包含最佳类别，不执行拼接。")
        return candidate_sentence

    
    words = candidate_sentence.split()   #将句子按空格分割成单词列表，以便进行插入操作
    new_words = []
    for w in words:
        #遍历单词列表，找到歧义观点词的位置
        if w.lower() == opinion_word:
            # 构造自然语序：best_category + opinion_word，示例：opinion "high" + category "price" → "high price"
            fixed_phrase = f"{opinion_word} {best_category}"
            new_words.append(fixed_phrase)
        else:
            new_words.append(w)
    new_sentence = " ".join(new_words)   #将修改后的单词列表重新组合成句子
    logger.info(f"[KG-DISAMBIG-FINAL] {candidate_sentence} → {new_sentence}")
    return new_sentence
                




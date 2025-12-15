# =================================================================================
# 种子词提取-----L-LDA 训练 → 初始种子词 → 知识图谱子图 → 融合增强 → 保存增强种子词
# =================================================================================
# 功能：结合 L-LDA 主题词 与 ConceptNet 知识图谱进行种子词“受控增强”
# 1. 先用 L-LDA 训练并提取初始种子词
# 2. 利用 ConceptNet 领域子图补充候选种子词
# 3. 对 KG 补充的词进行严格过滤：
#（1）限制 domain: 领域过滤 
#（2）限制句子 co-occurrence: 语料共同出现
#（3）限制词性 pos:
#（4）限制 embedding similarity: 与原始种子词的语义一致性
# 4. 排序:L-LDA 种子词固定排在前面，KG 补充词排在后面
# 5. 每个类别最终保留 top_n=30 个增强种子词
# =================================================================================

import random
import sys
import pickle as pk
import argparse
import os
import json
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.data.path.append('/hy-tmp/BERT-ASC-main/nltk_data')
sys.path.append('../')
import gensim
import model.labeled_lda as llda 


# ------------------------- 停用词 -------------------------
stops = stopwords.words('english')
stops.extend([
    'us', "didnt", 'im', 'make', 'today', 'feel', 'sometimes', 'ive','whatever', 'never', 'although', 
    'anyway', 'get', 'always','usually', 'want', 'go', 'also', 'would', 'one', 'theres'])


# ------------------------- 加载知识图谱子图 ----------------------------
def load_kg_subgraph(dataset):
    """
    加载预构建的领域 ConceptNet 子图：
    格式：{词: [(关联词1, 置信度1), ...]}
    """
    kg_path = f'../../datasets/{dataset}/conceptnet_domain_subgraph.pk'
    if not os.path.exists(kg_path):
        raise FileNotFoundError(f"知识图谱文件不存在: {kg_path}")
    print(f"[INFO] Loading knowledge graph from {kg_path}")
    return pk.load(open(kg_path, 'rb'))


# ------------------------- 简单预处理 + 分词 ----------------------------
def _simple_tokenize(text):
    import string
    text = text.strip().lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return [w for w in word_tokenize(text) if w]

    
# ================================================================
# ① 语料统计：全局词频 + 按类别的词频（用于 domain / co-occurrence）
# ================================================================
def build_corpus_stats(dataset, categories):
    """
    :return:
        cat_token_freq: {category: Counter(词→频次)}  —— co-occurrence 约束
        global_freq   : Counter(词→频次)             —— domain 约束
    """
    train_path = f'../../datasets/{dataset}/train.pk'
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.pk 不存在: {train_path}")
    train_data = pk.load(open(train_path, 'rb'))

    cat_token_freq = {c: Counter() for c in categories if c != 'anecdotes'}
    global_freq = Counter()
    
    for label_str, sent in train_data.items():
        tokens = _simple_tokenize(sent)
        if not tokens:
            continue
        global_freq.update(tokens)
        labels = label_str.split()
        for c in labels:
            if c in cat_token_freq:
                cat_token_freq[c].update(tokens)

    return cat_token_freq, global_freq


    
# ============================================================
# POS 限制 —— 按数据集区分
# ============================================================

pos_allow_semeval = {
    'food':     {'NN', 'NNS','JJ'},                     # 食物主要是名词
    'price':    {'NN', 'NNS','JJ'},                      # 价格为名词/形容词
    'service':  {'NN', 'VB', 'VBD', 'VBG', 'VBN', 'JJ'},   # 服务涉及名词 + 动词 + 形容词
    'ambience': {'NN', 'NNS','JJ'}                       # 氛围主要是形容词/名词
}

pos_allow_sentihood = {
    'price':             {'NN','NNS', 'JJ'},
    'safety':            {'NN', 'NNS', 'JJ', 'JJR', 'JJS'},
    'general':           {'NN', 'NNS', 'JJ', 'JJR', 'JJS'},
    'transit-location':  {'NN', 'NNS', 'JJ'}
}

def get_pos_allow(dataset):
    if dataset == 'semeval':
        return pos_allow_semeval
    elif dataset == 'sentihood':
        return pos_allow_sentihood
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


    
# ==============================================================================
# ② 受控增强：在这里实现 4 重限制
#    - 限制domain：只保留在语料中出现频次 ≥ min_domain_freq 的词
#    - 限制句子co-occurrence：只保留在“该类别句子”中出现频次 ≥ min_cooccur 的词
#    - 限制POS：限定词性（按 category 分）
#    - 限制embedding similarity：与 L-LDA 种子平均向量的余弦相似度 ≥ sim_threshold
# ===============================================================================
def enhance_seeds(
    lda_seeds,
    kg_subgraph,
    dataset,
    w2v_model,
    top_n=30,
    decay=0.8,
    min_domain_freq=1,
    min_cooccur=1,
    sim_threshold=0.25
):
    """
    :param lda_seeds: {类别: [L-LDA 初始种子词（含类别名）]}
    :param kg_subgraph: ConceptNet 领域子图 {词: [(关联词, 置信度), ...]}
    :param dataset: 数据集名称（用于加载 train.pk 做 co-occurrence）
    :param w2v_model: 词向量模型，用于 embedding similarity
    """
    # ---------- 1) 从语料构建统计 ----------
    cat_token_freq, global_freq = build_corpus_stats(dataset, lda_seeds.keys())

    # ---------- 2) 每个方面允许的 POS ----------
    pos_allow = get_pos_allow(dataset)

    
    enhanced = {}
    for category, seeds in lda_seeds.items():
        # 先过滤掉停用词
        seeds = [s for s in seeds if s not in stops]

        # ---------- 3) L-LDA 种子平均向量（用于 similarity） ----------
        valid_seed_vecs = [w2v_model.wv[s] for s in seeds if s in w2v_model.wv]
        avg_seed_vec = np.mean(valid_seed_vecs, axis=0) if valid_seed_vecs else None

        # ---------- 4) domain 词表（语料中高频的词才算“本领域”） ----------
        domain_vocab = {w for w, c in global_freq.items() if c >= min_domain_freq}

        kg_candidates = {}   # w → 加权得分（decay * confidence）

        # ---------- 5) 从 ConceptNet 子图补充 & 4 重约束过滤 ----------
        for seed in seeds:
            if seed not in kg_subgraph:
                continue
            for w, conf in kg_subgraph[seed]:
                w = w.lower()
                if w in stops or w in seeds:
                    continue

                # (1) domain 过滤：只保留在本语料中出现过一定频次的词
                if w not in domain_vocab:
                    continue

                # (2) co-occurrence 过滤：该词在“标为此类别的句子”里至少出现 min_cooccur 次
                if category in cat_token_freq and cat_token_freq[category][w] < min_cooccur:
                    continue

                # (3) POS 过滤：按类别限制词性
                pos = nltk.pos_tag([w])[0][1]
                if pos not in pos_allow.get(category, {'NN', 'JJ'}):
                    continue

                # (4) embedding similarity 过滤
                if avg_seed_vec is not None and w in w2v_model.wv:
                    sim = cosine_similarity(
                        avg_seed_vec.reshape(1, -1),
                        w2v_model.wv[w].reshape(1, -1)
                    )[0][0]
                    if sim < sim_threshold:
                        continue

                # 通过 4 重过滤：保留（取最大得分）
                score = decay * conf
                if w in kg_candidates:
                    kg_candidates[w] = max(kg_candidates[w], score)
                else:
                    kg_candidates[w] = score

        # ---------- 6) 按得分排序 KG 候选词 ----------
        sorted_kg = sorted(kg_candidates.items(), key=lambda x: x[1], reverse=True)

        # ---------- 7) 合并：L-LDA 前 20 个 + KG 动态补充，总数不超过 top_n ----------
        num_ll = min(20, len(seeds))                # L-LDA 固定保留前 20
        remaining = top_n - num_ll                  # KG 可补充的最大数量
        num_kg = min(len(sorted_kg), remaining)     # KG 动态数量

        final = seeds[:num_ll] + [w for w, _ in sorted_kg[:num_kg]]

        enhanced[category] = final

    return enhanced



# ------------------ L-LDA 训练 -----------------------------
def train(dataset, n_iter=400, save_every=100):
    print(f"[INFO] Training L-LDA on {dataset} (iterations={n_iter})")

    train_path = f'../../datasets/{dataset}/train.pk'
    test_path = f'../../datasets/{dataset}/test.pk'
    labeled_documents_train = pk.load(open(train_path, 'rb'))
    labeled_documents_train = [(v, k.split()) for k, v in labeled_documents_train.items()]
    labeled_documents_test = pk.load(open(test_path, 'rb'))
    labeled_documents_test = [(v, k.split()) for k, v in labeled_documents_test.items()]

    llda_model = llda.LldaModel(labeled_documents=labeled_documents_train, alpha_vector=0.01)
    save_model_dir = f"../../datasets/{dataset}"
    os.makedirs(save_model_dir, exist_ok=True)

    print("[INFO] 开始第一阶段训练（训练集）...")
    while True:
        llda_model.training(1)
        print(
            f"[TRAIN] Iter {llda_model.iteration:04d} | "
            f"Perplexity={llda_model.perplexity():.4f} | "
            f"Δβ={llda_model.delta_beta:.6f}"
        )
        if llda_model.iteration % save_every == 0:
            llda_model.save_model_to_dir(save_model_dir)
            print(f"[AUTO-SAVE] Model checkpoint saved at iteration {llda_model.iteration}")
        if llda_model.iteration >= n_iter:
            break

    update_labeled_documents = random.sample(labeled_documents_test, k=min(2, len(labeled_documents_test)))
    print("[INFO] Updating model with test samples...")
    llda_model.update(labeled_documents=update_labeled_documents)

    print("[INFO] 开始第二阶段训练（迭代优化）...")
    while True:
        llda_model.training(1)
        print(
            f"[TRAIN] Iter {llda_model.iteration:04d} | "
            f"Perplexity={llda_model.perplexity():.4f} | "
            f"Δβ={llda_model.delta_beta:.6f}"
        )
        if llda_model.iteration % save_every == 0:
            llda_model.save_model_to_dir(save_model_dir)
            print(f"[AUTO-SAVE] Model checkpoint saved at iteration {llda_model.iteration}")
        if llda_model.iteration >= n_iter * 2:
            break

    llda_model.save_model_to_dir(save_model_dir)
    print(f"[DONE] Training complete. Final model saved to {save_model_dir}")


    
# ------------------- 种子词提取 + KG 受控增强 ---------------------------
def inference(dataset, n=20, top_n=30, decay=0.8,
              embedding_path='/hy-tmp/BERT-ASC-main/ASC_generating/embeddings/restaurant.bin'):
    save_model_dir = f"../../datasets/{dataset}"

    # 1) 加载训练好的 L-LDA 模型
    llda_model = llda.LldaModel()
    llda_model.load_model_from_dir(save_model_dir, load_derivative_properties=False)

    # 2) 加载类别列表
    categories_path = f'../../datasets/{dataset}/categories.pk'
    categories = pk.load(open(categories_path, 'rb'))

    # 3) 加载 Word2Vec 模型（用于 embedding similarity）
    if embedding_path.endswith('.vec'):
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, binary=False)
    elif embedding_path.endswith('.bin'):
        try:
            w2v_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, binary=True)
        except Exception:
            w2v_model = gensim.models.Word2Vec.load(embedding_path)
    else:
        w2v_model = gensim.models.Word2Vec.load(embedding_path)

    # 4) 提取 baseline 风格的 L-LDA 初始 seeds
    categories_seed = {}
    for c in categories:
        if c == 'anecdotes':
            continue
        seeds = llda_model.top_terms_of_topic(c, n, False)
        # 过滤：去掉包含其他类别名的词 + 停用词
        seeds = [s for s in seeds if all(c2 not in s for c2 in categories) and s not in stops]
        seeds.append(c)    #加入类别名本身
        seeds = list(sorted(set(seeds).difference(set(stops))))
        categories_seed[c] = seeds

    # 5) 加载 ConceptNet 领域子图
    kg_subgraph = load_kg_subgraph(dataset)

    # 6) 做受控增强（4 重限制）
    enhanced_seeds = enhance_seeds(
        lda_seeds=categories_seed,
        kg_subgraph=kg_subgraph,
        dataset=dataset,
        w2v_model=w2v_model,
        top_n=top_n,
        decay=decay
    )

    # 7) 保存增强种子
    pk.dump(enhanced_seeds, open(f'../../datasets/{dataset}/categories_seeds_lc.pk', 'wb'))
    json.dump(
        enhanced_seeds,
        open(f'../../datasets/{dataset}/categories_seeds_lc.json', 'w', encoding='utf-8'),
        indent=2,
        ensure_ascii=False
    )
    print(f"[DONE] Enhanced seeds saved to ../../datasets/{dataset}/categories_seeds_lc.pk & .json")


def main():
    parser = argparse.ArgumentParser(description="基于 L-LDA + ConceptNet（带 4 重约束）生成增强种子词")
    parser.add_argument('--dataset', type=str, default='semeval', help='选择数据集: semeval / sentihood')
    parser.add_argument('--n_iter', type=int, default=400, help='训练迭代次数 (默认: 400)')
    parser.add_argument('--save-every', type=int, default=100, help='每多少次迭代保存一次断点')
    parser.add_argument('--n', type=int, default=20, help='每个 topic 取多少 top 词作为初始 seeds')
    parser.add_argument('--topn', type=int, default=30, help='每个类别保留的增强种子数量')
    parser.add_argument('--decay', type=float, default=0.8, help='知识图谱置信度衰减系数 (默认: 0.8)')
    parser.add_argument('--embedding_path',
                        default='/hy-tmp/BERT-ASC-main/ASC_generating/embeddings/restaurant.bin',
                        type=str,
                        help='Word2Vec 词向量模型路径，用于限制 KG 候选词的 embedding 相似度')
    parser.add_argument('--no-train', action='store_true',
                        help='若指定，仅基于已有 L-LDA 模型做推理与种子增强，不重新训练')
    opt = parser.parse_args()

    # 可选：是否重新训练 L-LDA
    if not opt.no_train:
        train(opt.dataset, n_iter=opt.n_iter, save_every=opt.save_every)

    # 推理 + KG 受控增强
    inference(
        dataset=opt.dataset,
        n=opt.n,
        top_n=opt.topn,
        decay=opt.decay,
        embedding_path=opt.embedding_path
    )


if __name__ == '__main__':
    main()

#=============================================================
# 加载数据 → 训练模型 → 更新 → 推理 → 生成种子词
#=============================================================

import random   #导入随机数生成模块，用于随机采样等操作
import sys      #导入系统模块
import pickle as pk    #导入pickle模块，用于数据的序列化和反序列化，保存和加载数据
sys.path.append('../')    #将上级目录添加到系统路径中，以便导入其他模块
import model.labeled_lda as llda   #导入自定义的labeled_lda模块，用于L-LDA模型相关操作
import argparse   #导入参数解析模块
import nltk
import json  # 修正1：新增依赖（情感词典加载需要）
from nltk.corpus import stopwords, wordnet  # 修正2：导入wordnet（情感词匹配需要）
nltk.data.path.append('/hy-tmp/BERT-ASC-main/nltk_data')   #设置nltk数据的路径

# 修正3：补充停用词列表缺失的逗号
stops = stopwords.words('english')  #加载英文停用词
stops.extend(['us', "didnt",'im', 'make','today', 'feel', 'sometimes', 'ive', 'whatever','never',
              'although','anyway','get','always', 'usually', 'want', 'go','also','would', 'one', 'theres'
])   #扩展停用词列表

# -----------------------
# 关键：导入/复用情感词匹配函数（从preprocessing.py复用核心逻辑）
# -----------------------
def load_senticnet(path="../datasets/senticnet/senticnet.json"):
    """加载SenticNet情感词典（复用preprocessing.py逻辑）"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            senticnet = json.load(f)
    except FileNotFoundError:
        print(f"错误：未找到SenticNet词典文件，请检查路径：{path}")
        return {}
    sentiment_words = set()
    for word in senticnet:
        try:
            polarity = float(senticnet[word]['polarity'])
            if abs(polarity) > 0.1:
                sentiment_words.add(word.lower())
        except (KeyError, ValueError):
            continue
    return sentiment_words

# 加载情感词典（全局变量）
sentiment_words = load_senticnet()

def is_sentiment_word(word):
    """判断词是否为情感相关词（复用preprocessing.py逻辑）"""
    if word is None or word.strip() == "":
        return False
    word_lower = word.lower()
    if word_lower in sentiment_words:
        return True
    try:
        for syn in wordnet.synsets(word_lower):
            for lemma in syn.lemmas():
                if lemma.name().lower() in sentiment_words:
                    return True
    except:
        pass
    return False

# -----------------------
# 训练函数
# -----------------------
def train(dataset):  
    n_iter= 400   #设置训练迭代次数

    labeled_documents_train  = pk.load(open('../datasets/{}/train.pk'.format(dataset), 'rb'))  #加载训练集带标签的文档数据
    labeled_documents_train = [(v, k.split()) for k, v in labeled_documents_train.items()]    #将文档数据转化为（文档内容，标签列表）的形式
    labeled_documents_test = pk.load(open('../datasets/{}/test.pk'.format(dataset), 'rb'))    #加载测试集带标签的文档数据
    labeled_documents_test = [(v, k.split()) for k, v in labeled_documents_test.items()]      #将文档数据转化为（文档内容，标签列表）的形式
    llda_model = llda.LldaModel(labeled_documents=labeled_documents_train, alpha_vector=0.01)    #初始化L-LDA模型
    print(llda_model)   #打印模型初始信息

    
    #--------------------- 第一次训练 ------------------------
    while True:
        print("iteration %s sampling..." % (llda_model.iteration + 1))   #打印当前迭代次数的采样信息
        llda_model.training(1)    #进行一次训练迭代
        #打印当前迭代后的困惑度和beta的变化量
        print("after iteration: %s, perplexity: %s" % (llda_model.iteration, llda_model.perplexity()))   
        print("delta beta: %s" % llda_model.delta_beta)
        if llda_model.iteration > n_iter:    #当迭代次数超过设定的n_iter时，跳出循环
            break
            
    #更新模型
    print("before updating: ", llda_model)    #打印更新前的模型信息
    update_labeled_documents = random.sample(labeled_documents_test, k=2)   #从测试集中随机采样2个文档作为更新用的带标签文档
    llda_model.update(labeled_documents=update_labeled_documents)    #使用采样的文档更新模型
    print("after updating: ", llda_model)    #打印更新后的模型信息

    #---------------------------- 第二次 训练模型 -----------------------
    while True:
        print("iteration %s sampling..." % (llda_model.iteration + 1))   #打印当前迭代次数
        llda_model.training(1)   #进行一次训练迭代
        #打印当前迭代后的困惑度和beta的变化量
        print("after 1 iteration: %s, perplexity: %s" % (llda_model.iteration, llda_model.perplexity()))
        print("delta beta: %s" % llda_model.delta_beta)
        if llda_model.iteration > n_iter*2:    #当迭代次数超过2倍的n_iter时，跳出循环
            break
            
    save_model_dir = "../datasets/{}".format(dataset)   #设置模型保存目录
    llda_model.save_model_to_dir(save_model_dir)        #将模型保存到指定目录


    
# -----------------------
# 推理函数（核心修正）
# -----------------------
def inference(dataset, n = 15):  # 修正4：统一参数为Top-15（匹配注释）
    save_model_dir  = "../datasets/{}".format(dataset)    #模型保存目录
    llda_model = llda.LldaModel()    #初始化一个新的L-LDA模型
    llda_model.load_model_from_dir(save_model_dir, load_derivative_properties=False)   #从指定目录加载模型

    labeled_documents_test = pk.load(open('../datasets/{}/test.pk'.format(dataset), 'rb'))   #加载测试集带标签的文档数据
    labeled_documents_test = [(v, k.split()) for k, v in labeled_documents_test.items()]     #将文档数据转化为（文本，标签列表）形式
    document = random.sample(labeled_documents_test, k=1)[0][0]    #从测试集中随机采样一个文档作为推理的文档
    topics = llda_model.inference(document=document, iteration=100, times=10)  #对文档进行推理，得到主题分布

    categories = pk.load(open('../datasets/{}/categories.pk'.format(dataset), 'rb'))   #加载数据集的类别信息
    #打印每个类别的前15个顶级词（跳过'anecdotes'类别）
    print("\n=== 原始Top-15主题词 ===")
    for c in categories:
        if c == 'anecdotes':
            continue
        print(f"Top-15 terms of topic {c}: ", llda_model.top_terms_of_topic(c, n, False))
    
    categories_seed = {}   #用于存储每个类别的种子词
    # 修正5：适配不同数据集的领域补充词
    domain_supplements = {
        # SemEval（餐厅）补充词
        'semeval': {
            'food': ['tasty', 'delicious', 'fresh', 'awful', 'yummy', 'bland', 'juicy', 'spicy', 'soggy', 'crispy'],
            'service': ['friendly', 'helpful', 'rude', 'slow', 'prompt', 'unhelpful', 'polite', 'attentive', 'ignorant', 'efficient'],
            'price': ['cheap', 'expensive', 'reasonable', 'overpriced', 'affordable', 'pricy', 'inexpensive', 'worthwhile', 'costly', 'budget'],
            'ambience': ['cozy', 'noisy', 'pleasant', 'dirty', 'charming', 'dreary', 'inviting', 'cramped', 'elegant', 'loud']
        },
        # Sentihood（地点评价）补充词
        'sentihood': {
            'general': ['great', 'terrible', 'wonderful', 'awful', 'amazing', 'horrible', 'nice', 'poor', 'excellent', 'bad'],
            'price': ['cheap', 'expensive', 'reasonable', 'overpriced', 'affordable', 'pricy', 'inexpensive', 'costly', 'budget', 'worth'],
            'transit-location': ['convenient', 'accessible', 'remote', 'isolated', 'central', 'far', 'close', 'inconvenient', 'near', 'distant'],
            'safety': ['safe', 'secure', 'dangerous', 'risky', 'unsafe', 'protected', 'threatening', 'harmless', 'vulnerable', 'guarded']
        }
    }
    # 根据数据集选择补充词
    supplements_map = domain_supplements.get(dataset, domain_supplements['semeval'])

    for c in categories:
        if c == 'anecdotes':
            continue
        # 1. 获取原始Top-N主题词
        raw_seeds = llda_model.top_terms_of_topic(c, n*2, False)  # 多取一倍，避免过滤后不足
        # 2. 过滤：保留情感词 + 去停用词
        sentiment_seeds = [s for s in raw_seeds if is_sentiment_word(s) and s not in stops]
        
        # 3. 补充领域情感词（若不足n个）
        if len(sentiment_seeds) < n:
            supplements = supplements_map.get(c, [])
            for supplement in supplements:
                if supplement not in sentiment_seeds and supplement not in stops:
                    sentiment_seeds.append(supplement)
                if len(sentiment_seeds) >= n:
                    break
        
        # 4. 最终保留Top-n，去重（避免补充词重复）
        final_seeds = list(dict.fromkeys(sentiment_seeds))[:n]  # 去重且保留顺序
        categories_seed[c] = final_seeds

    # 打印最终种子词
    print("\n=== 过滤后情感种子词 ===")
    for c in categories_seed:
        print(f"Top-{n} sentiment seeds of {c}: ", categories_seed[c])
    
    # 保存种子词
    pk.dump(categories_seed, open('../datasets/{}/categories_seeds.pk'.format(dataset), 'wb'))   


# -----------------------
# 主函数
# -----------------------
def main ():   
    parser = argparse.ArgumentParser()   #创建参数解析器
    parser.add_argument('--dataset', default='semeval', type=str, help='semeval, sentihood')
    opt = parser.parse_args()    #解析命令行的参数
    train(opt.dataset)      #调用训练函数，传入数据集名称
    inference(opt.dataset)  #调用推理函数，传入数据集名称


if __name__ == '__main__':   
    main()    
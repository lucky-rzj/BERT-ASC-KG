#对原始的 SemEval-2014（餐厅领域）和 Sentihood（地点评价领域）数据集进行标准化预处理，最终生成 L-LDA 模型可直接读取的输入数据（.pk 格式文件）

import xml.etree.ElementTree     #导入XML解析库，用于解析semeval数据集的XML文件
path_base = r"../datasets/raw/"  #原始数据集存储路径   
import pickle as pk    #导入pickle库，用于数据序列的初始化
import string       #导入字符串处理库
import csv, re    #导入CSV和正则表达式库，用于文本清洗
from tqdm import tqdm   #导入进度条库
import json   #导入JSON库，用于处理sentihood数据集的JSON文件
import nltk  
nltk.data.path.append('/hy-tmp/BERT-ASC-main/nltk_data')   #设置nltk数据的路径
from nltk.tokenize import word_tokenize    #导入分词
from nltk.corpus import stopwords, wordnet     #导入停用词表
from sklearn.feature_extraction.text import TfidfVectorizer   #导入TF-IDF向量器，用于提取关键词

stops = stopwords.words('english')    #加载英文停用词
stops.extend(['us', "didnt",'im', 'make','today', 'feel', 'sometimes', 'ive', 'whatever','never','although','anyway','get','always', 'usually', 'want', 'go','would', 'one'])  #扩展通用词表
strong_sentiment_words = []  


#------------------------- 文本清洗函数 ---------------------------
def cleaning(text): 
    if text is None:  # 处理空文本
        return ""  
    words_to_repere = ["it's", "don't", "'", "isn't", "you've", "'s"]   #需要特殊处理的缩写词
    for w in words_to_repere:  
        text = text.replace(w, ' ' + w + ' ')    #在缩写词前后加空格，避免分词错误
    text = re.sub(r'[^\w\s]', '', text)  #移除所有非单词和非空格字符（即标点符号）
    return text


#---------------------- 新增情感词典加载函数 -------------------------------
def load_senticnet(path="../datasets/senticnet/senticnet.json"):
    """加载SenticNet情感词典"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            senticnet = json.load(f)
    except FileNotFoundError:
        print(f"错误：未找到SenticNet词典文件，请检查路径：{path}")
        print("提示：官方下载文件名为senticnet5.json，可将其重命名为senticnet.json或修改上述path参数")
        return {}  # 返回空字典，避免程序崩溃

    # 构建情感词集合(包含积极和消极情感词)
    sentiment_words = set()
    for word in senticnet:
        # 提取具有情感极性的词(排除中性词)
        try:
            polarity = float(senticnet[word]['polarity'])
            if abs(polarity) > 0.1:  # 设置极性阈值过滤弱情感词
                sentiment_words.add(word.lower())  # 统一小写，避免匹配失败
        except (KeyError, ValueError):
            continue  # 跳过格式异常的条目
    return sentiment_words


# 加载情感词典(全局变量)
sentiment_words = load_senticnet()


#---------------------------- 新增情感词匹配函数 ----------------------------
def is_sentiment_word(word):
    """判断词是否为情感相关词"""
    if word is None or word.strip() == "":
        return False
    word_lower = word.lower()
    if word_lower in sentiment_words:
        return True

    # 补充WordNet同义词检查
    try:  # 处理无同义词的情况
        for syn in wordnet.synsets(word_lower):
            for lemma in syn.lemmas():
                if lemma.name().lower() in sentiment_words:
                    return True
    except:
        pass
    return False



#------------------------------- 处理semeval14数据集的函数 --------------------------------------
def generate_semeval14_training(train=True):  
    file_src= 'Restaurants_Test_Gold.xml'   #测试集文件名
    file_save= 'test'   #保存的测试集文件名前缀
    if train:
        file_src='Restaurants_Train_v2.xml'   #训练集文件名
        file_save = 'train'   #保存的训练集文件名前缀
    data = {}    #用于存储处理后的数据（键：方面类别，值：该类别对应的文本）
    e = xml.etree.ElementTree.parse(path_base+'semeval/' + file_src).getroot()    #解析XML文件
    reviews = e.findall('sentence')    #获取所有句子节点
    for review in (reviews):
        text = review.find('text').text   #获取句子文本
        aspect_term = []   #存储方面类别
        aspect_polarity = []    #存储情感极性（预留，未使用）
        options = review.findall('aspectCategories')   #获取方面类别节点
        for option in (options):   
            suboptions = option.findall('aspectCategory')   #获取具体方面类别
            for suboption in (suboptions):
                aspect = suboption.get("category")   #获取方面类别的名称
                polarity = suboption.get("polarity")   #获取情感极性
                if aspect=='anecdotes/miscellaneous':
                    aspect='anecdotes'   #简化类别名称
                aspect_term.append(aspect)
                aspect_polarity.append(polarity)

        text = cleaning(text)  # 统一调用cleaning函数，保持清洗逻辑一致
        text= text.strip().lower()     #去除首尾空格并转化为小写
        text = ' '.join([w for w in word_tokenize(text) if w not in stops])   

        if len(aspect_term)>1:
            continue   #跳过包含多个方面类别的句子（简化处理）
        if aspect == 'anecdotes': continue    #跳过'anecdotes'类别（可选）
        try:
            data[' '.join(aspect_term)]+=' '+text   #已存在的类别，追加文本
        except:
            data[' '.join(aspect_term)]= text   #新类别，初始化文本

    pk.dump(data, open('../datasets/semeval/{}.pk'.format(file_save), 'wb'))   #保存处理后的数据为pickle文件，供L-LDA模型使用

    

#-------------------------- 处理sentihood数据集的函数 ------------------------------------
def process_sentihood():   
    categories = {'general','price', 'transit-location', 'safety'}   #关注的方面类别
    for phase in ['train', 'test']:  #处理训练集和测试集
        data_to_save = dict()  #存储处理后的数据（键：方面类别，值：文本）
        with open('../datasets/raw/sentihood/sentihood-{}.json'.format(phase),'r') as f :
            data= json.load(f)  #加载JSON格式的原式数据
            for d in data:
                aspect_category = [ac['aspect'] for ac in d['opinions']]    #提取方面类别
                aspect_category= list(set(aspect_category))   #去重
                if not len(aspect_category) or len(aspect_category)>1 or not len(set(aspect_category).intersection(categories)):   #过滤无效样本
                    continue
                text = d['text']   #获取文本
                text = cleaning(text)    #统一调用cleaning函数
                text= text.replace('LOCATION1','').replace('LOCATION2','')    #移除地点标记
                text = text.strip().lower()   #去除空格并小写
                text = ' '.join([w for w in word_tokenize(text) if w not in stops])   #分词并过滤停用词
                try:    #按方面类别聚合文本
                    data_to_save[' '.join(aspect_category)] += ' ' + text
                except:
                    data_to_save[' '.join(aspect_category)] = text
        pk.dump(data_to_save, open('../datasets/sentihood/{}.pk'.format( phase), 'wb'))   #保存处理后数据
        print(f"Sentihood {phase} 处理完成，类别：{data_to_save.keys()}")   # 优化：统一日志格式
        f.close()


        
#----------------------- 用TF-IDF优化sentihood数据集处理 ---------------------------
def process_sentihood_tf_idf():    
    categories = {'general','price', 'transit-location', 'safety'}
    for phase in ['train', 'test']:
        data_to_save=dict()
        corpus = list()   #存储所有文本
        aspect = []  #存储每个文本对应的方面类别
        with open('../datasets/raw/sentihood/sentihood-{}.json'.format(phase),'r') as f :
            data= json.load(f)
            for d in data:
                aspect_category = [ac['aspect'] for ac in d['opinions']]
                aspect_category = list(set(aspect_category))
                aspect.append(aspect_category)  #记录类别
                text = d['text']
                text = cleaning(text) 
                text = text.replace('LOCATION1', '').replace('LOCATION2', '')   
                text = text.strip().lower()
                text = ' '.join([w for w in word_tokenize(text) if w not in stops+strong_sentiment_words])   #分词并过滤停用词和强情感词
                corpus.append(text)  #记录文本
        f.close()
        
        vectorizer = TfidfVectorizer()  #初始化TF-IDF向量器
        tf_idf = vectorizer.fit_transform(corpus)   #计算TF-IDF矩阵
        for i in range(len(corpus)):
            aspect_category = aspect[i]
            text = corpus[i]   
            if not len(aspect_category) or len(aspect_category) > 1 or not len(set(aspect_category).intersection(categories)):   #过滤无效样本
                continue
            vocab = dict(zip(vectorizer.get_feature_names_out(), tf_idf.toarray()[i]))     #构建词汇TF-IDF值映射
            print(text)
            text = ' '.join([w for w in text.split() if w in vocab and vocab.get(w)>.15])   #保留TF-IDF值>0.15的词汇（过滤低重要度词）
            print(text)
            print(aspect_category)
            print('------------')
            try:    #按类别聚合文本
                data_to_save[' '.join(aspect_category)] += ' ' + text
            except:
                data_to_save[' '.join(aspect_category)] = text

        pk.dump(data_to_save, open('../datasets/sentihood/{}.pk'.format(phase), 'wb'))   #保存处理后的数据
        print(f"Sentihood TF-IDF {phase} 处理完成，类别：{data_to_save.keys()}")
        f.close()


        
#---------------------------- 用TF-IDF优化semeval数据集处理 ----------------------------------
def process_semeval_tf_idf():      
    for phase, file_src in zip(['train', 'test'], ['Restaurants_Train_v2.xml','Restaurants_Test_Gold.xml']):   #遍历训练集和测试集对应的文件
        data_to_save=dict()
        corpus = list()
        aspect = []
        e = xml.etree.ElementTree.parse(path_base + 'semeval/' + file_src).getroot()
        reviews = e.findall('sentence')
        for review in (reviews):
            text = review.find('text').text
            aspect_term = []
            options = review.findall('aspectCategories')
            for option in (options):
                suboptions = option.findall('aspectCategory')
                for suboption in (suboptions):
                    aspect_term.append(suboption.get("category"))   #记录方面类别

            aspect.append(aspect_term)
            text = cleaning(text)
            text = text.strip().lower()
            text = ' '.join([w for w in word_tokenize(text) if w not in stops+strong_sentiment_words])   #分词并过滤停用词和强情感词
            corpus.append(text)  #记录文本

        vectorizer = TfidfVectorizer()
        tf_idf = vectorizer.fit_transform(corpus)   #遍历所有文本，按TF-IDF过滤并聚合
        for i in tqdm(range(len(corpus))):   #tqdm显示进度
            aspect_category = aspect[i]
            text = corpus[i]
            if not len(aspect_category) or len(aspect_category) > 1 or 'anecdotes/miscellaneous' in aspect_category :   #过滤无效样本
                continue

            vocab = dict(zip(vectorizer.get_feature_names_out(), tf_idf.toarray()[i]))
            text = ' '.join([w for w in text.split() if w in vocab and vocab.get(w)>.15])   #保留TF-IDF值>0.15的词汇（修正注释错误）
            try:   #按类别聚合
                data_to_save[' '.join(aspect_category)] += ' ' + text
            except:
                data_to_save[' '.join(aspect_category)] = text

        pk.dump(data_to_save, open('../datasets/semeval/{}.pk'.format(phase), 'wb'))   #保存数据
        print(f"SemEval TF-IDF {phase} 处理完成，类别：{data_to_save.keys()}")  # 新增：补充打印日志


if __name__ == '__main__':
    #主函数：执行sentihood和semeval的TF-IDF预处理
    process_sentihood_tf_idf()
    process_semeval_tf_idf()
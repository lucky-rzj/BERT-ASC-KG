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
nltk.data.path.append('/hy-tmp/BERT-ASC-main/nltk_data')   #设置nltk数据的路径
from nltk.corpus import stopwords   #从nltk语料库中导入停用词列表
stops = stopwords.words('english')  #加载英文停用词
stops.extend(['us', "didnt",'im', 'make','today', 'feel', 'sometimes', 'ive' 'whatever','never',
              'although','anyway','get','always', 'usually', 'want', 'go','also','would', 'one', 'theres'
])   #扩展停用词列表



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
    update_labeled_documents =random.sample(labeled_documents_test, k=2)   #从测试集中随机采样2个文档作为更新用的带标签文档
    llda_model.update(labeled_documents=update_labeled_documents)    #使用采样的文档更新模型
    print("after updating: ", llda_model)    #打印更新后的模型信息

    #---------------------------- 第二次 训练模型 -----------------------
    while True:
        print("iteration %s sampling..." % (llda_model.iteration + 1))   #打印当前迭代次数
        llda_model.training(1)   #进行一次训练迭代
        #打印当前迭代后的困惑度和beta的变化量
        print("after 1 iteration: %s, perplexity: %s" % (llda_model.iteration, llda_model.perplexity()))
        print("delta beta: %s" % llda_model.delta_beta)
        if llda_model.iteration >n_iter*2:    #当迭代次数超过2倍的n_iter时，跳出循环
            break
            
    save_model_dir = "../datasets/{}".format(dataset)   #设置模型保存目录
    llda_model.save_model_to_dir(save_model_dir)        #将模型保存到指定目录


    
# -----------------------
# 推理函数
# -----------------------
def inference(dataset, n = 20):  
    save_model_dir  = "../datasets/{}".format(dataset)    #模型保存目录
    llda_model = llda.LldaModel()    #初始化一个新的L-LDA模型
    llda_model.load_model_from_dir(save_model_dir, load_derivative_properties=False)   #从指定目录加载模型

    labeled_documents_test = pk.load(open('../datasets/{}/test.pk'.format(dataset), 'rb'))   #加载测试集带标签的文档数据
    labeled_documents_test = [(v, k.split()) for k, v in labeled_documents_test.items()]     #将文档数据转化为（文本，标签列表）形式
    document=random.sample(labeled_documents_test, k=1)[0][0]    #从测试集中随机采样一个文档作为推理的文档
    topics = llda_model.inference(document=document, iteration=100, times=10)  #对文档进行推理，得到主题分布，迭代100次，重复10次

    categories= pk.load(open('../datasets/{}/categories.pk'.format(dataset), 'rb'))   #加载数据集的类别信息
    #打印每个类别的前15个顶级词（跳过'anecdotes'类别）
    for c in categories:
        if c == 'anecdotes':
            continue
        print("Top-15 terms of topic : ",c,  llda_model.top_terms_of_topic(c, n, False))
    categories_seed={}   #用于存储每个类别的种子词
    for c in categories:
        if c=='anecdotes':
            continue
        seeds=llda_model.top_terms_of_topic(c, n, False)    #获取每个类别的前n个顶级词作为种子词
        seeds=[s for s in seeds if s not in categories ]    #过滤掉不属于类别的词
        seeds.append(c)   #将类别本身添加到种子词中
       
        seeds=list(set(seeds).difference(set(stops)))   #去除种子词中的停用词
        categories_seed[c]=seeds    #存储当前类别的种子词
    pk.dump(categories_seed, open('../datasets/{}/categories_seeds.pk'.format(dataset), 'wb'))    #将类别种子词保存到pickle文件中



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
  
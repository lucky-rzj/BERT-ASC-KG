# ====================================================================================================================
# 生成语义候选词
# 功能：基于 Word2Vec 相似度，从句子中挑选与 aspect（方面类别）相关的词
# ===================================================================================================================


import json    
import re     
import gensim   #自然语言处理库（加载预训练Word2Vec词向量模型）
import pickle as pk   #序列化库（加载预定义的方面类别和种子词数据）
import xml.etree.ElementTree  #解析XML文件
from nltk.tokenize import word_tokenize   #分词
from nltk.corpus import stopwords  #停用词
import nltk
nltk.data.path.append('/hy-tmp/BERT-ASC-main/nltk_data')   #设置nltk数据的路径
import string  
import numpy as np   
from sklearn.metrics.pairwise import cosine_similarity    #计算余弦相似度
from tqdm import tqdm 
import argparse    

stops = stopwords.words('english')    #加载英文停用词
stops.extend(['us', "didnt",'im','couldnt', 'make','today', 'feel', 'sometimes', 'ive' ,'whatever','never','although','anyway','get','always', 'usually', 'want', 'go','would', 'one'])    #扩展停用词表



def semeval(threshold=.3):
    """ 为semeval数据集生成语义候选词 """
    model = gensim.models.Word2Vec.load(r'embeddings/restaurant.bin')   #加载预训练的Word2Vec词向量模型（餐厅领域）
    path_base= '../datasets/raw/semeval/'   #原始semeval数据
    #加载方面类别和对应的种子词（由L-LDA生成）
    categories= pk.load(open('../datasets/semeval/categories.pk', 'rb'))  
    category_seed_words= pk.load(open('../datasets/semeval/categories_seeds.pk', 'rb'))  
    categories.append('anecdotes')    #补充“轶事”方面类别
  
    
    #遍历训练集和测试集
    for phase, file_src,  in zip(['train', 'test'], ['Restaurants_Train_v2.xml', 'Restaurants_Test_Gold.xml']):  
        data_to_save =[]   #存储最终结果
        corpus = list()    #存储原始文本
        aspect = []     #存储Aspect信息
        
        e = xml.etree.ElementTree.parse(path_base + file_src).getroot()   #解析XML
        reviews = e.findall('sentence')    #提取所有"sentence"标签
        #遍历每个句子标签
        for review in tqdm(reviews): 
            text = review.find('text').text    #获取句子原始文本
            aspect_category = []      #存储当前句子标注的方面类别
            aspect_polarity =dict()     #存储每个方面类别的情感极性
            #提取句子中的方面类别和对应极性（从XML的aspectCategories标签中解析）
            options = review.findall('aspectCategories')     #提取Aspect类别相关标签
            for option in (options):
                suboptions = option.findall('aspectCategory')   #提取每个具体的Aspect类别
                for suboption in (suboptions):
                    current_aspect =suboption.get("category").replace('anecdotes/miscellaneous', 'anecdotes')    #简化类别名称
                    aspect_category.append(current_aspect)    #记录真实Aspect类别
                    aspect_polarity[current_aspect]=suboption.get("polarity")   #记录对应情感极性

                    
            #文本预处理：小写转换 → 去除标点 → 分词 → 过滤停用词
            text = text.strip().lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            text_w_stops = [w for w in word_tokenize(text) if w not in stops]
            corpus.append(text)   #存储预处理后的文本
            

            #初始化字典：键：方面类别，值：语义候选词
            category_representatives={a:[] for a in aspect_category}
            #为每个方面类别提取语义候选词
            for a in aspect_category:
                if a =='anecdotes':
                    category_representatives[a].append('anecdotes')    #特殊处理"anecdotes"类别：直接添加类别名称作为候选词
                    continue
                    
                representatives=[]    #存储当前类别的语义候选词
                
                #遍历预处理后的每个词汇，判断是否为该类别的候选词
                for w in text_w_stops:
                    if w not in model.wv or  w in representatives :   #跳过：词汇不在词向量模型中（无语义向量）或已添加到候选词列表（去重）
                        continue
                    #获取当前类别的种子词向量
                    seed_vec = np.array([model.wv[w_] for w_ in category_seed_words.get(a) if w_ in model.wv])
                    #计算当前词与种子词的最大余弦相似度
                    score = np.max(cosine_similarity(seed_vec, np.expand_dims(model.wv[w], axis=0)), axis=0)[0]   
                    
                    #相似度超过阈值则视为候选词（阈值可通过函数参数调整）
                    if score > threshold:
                        representatives.append(w)
                        
                representatives = list(set(representatives).difference(set(categories)))   #候选词去重并排除类别本身名称
                category_representatives[a]= representatives   #保存当前类别的候选词


            # -------------- 构建输出数据结构 -------------------
            current_data= dict()   #存储当前句子的处理结果
            current_data['text']= text   #存储预处理后的文本
            asp_tem =[]    #存储每个方面类别的信息
            
            #遍历所有预定义的方面类别
            for aspect in categories:
                if aspect in category_representatives:
                    #有标注的类别：填入极性和候选词
                    temp={'category':aspect, 
                          'polarity':aspect_polarity.get(aspect), 
                          'auxiliary':list(set(category_representatives.get(aspect)))   #再次去重确保干净
                    }
                else:
                    #无标注的类别：极性设为"none"，候选词为空列表
                    temp = {'category': aspect, 
                            'polarity': 'none', 
                            'auxiliary': [] 
                    }  
                asp_tem.append(temp)
                
            current_data['aspect']=asp_tem    #关联所有类别信息
            data_to_save.append(current_data)   #将当前句子结果加入最终列表
            
        print(f"处理{phase}集样本数：{len(data_to_save)}")   #打印处理的样本数量   
        with open('../datasets/semeval/bert_{}.json'.format(phase), 'w') as f :   #保存结果为JSON文件
            json.dump(data_to_save, f, indent=3)
        f.close()



def sentihood(threshold=.4):   
    """ 为sentihood数据集生成语义候选词（逻辑类似semeval，但需处理地点标记）"""
    model = gensim.models.Word2Vec.load(r'embeddings/restaurant.bin')
    category_seed_words = pk.load(open('../datasets/sentihood/categories_seeds.pk', 'rb'))   #加载种子词
    categories = ['general', 'price', 'transit-location', 'safety']    #关注的方面类别

    
    #遍历训练集、测试集、验证集
    for phase in ['train', 'test', 'dev']:   
        data_to_save = []   #存储最终处理结果
        
        # 1. 路径后缀从 .json 改为 .jsonl（匹配你的数据集文件）
        file_path = '../datasets/raw/sentihood/sentihood-{}.jsonl'.format(phase)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = []
            # 2. 按行读取 .jsonl 文件（每行一个 JSON 对象）
            for line in f:
                line = line.strip() #去除换行符/空格
                if line:  #跳过空行
                    data.append(json.loads(line))   #解析每行的 JSON 数据

            #遍历每个样本
            for d in tqdm(data):    
                aspects = d['opinions']   #提取方面信息
                text = d['text']    #提取原始文本

                #文本预处理：移除地点标记（LOCATION1/LOCATION2）→ 小写 → 去标点 → 分词 → 过滤停用词
                text_w_stops = text.replace('LOCATION1', '').replace('LOCATION2', '')
                text_w_stops = text_w_stops.strip().lower()
                text_w_stops = text_w_stops.translate(str.maketrans('', '', string.punctuation))
                text_w_stops = [w for w in word_tokenize(text_w_stops) if w not in stops]

                
                #初始化字典：键：方面类别，值：语义候选词
                category_representatives={a['aspect']:[] for a in aspects if a['aspect'] in categories}
                aspect_category=[a['aspect'] for a in aspects if a['aspect'] in categories]   #提取当前样本的有效方面类别
                #记录情感极性:键："目标实体#类别"，值：情感极性
                aspect_polarity={a['target_entity']+'#'+a['aspect']: a['sentiment'] for a in aspects if a['aspect'] in categories}
                
                #为每个有效方面类别提取语义候选词
                for a in aspect_category:
                    representatives=[]
                    for w in text_w_stops:
                        if w not in model.wv or  w in representatives :   #跳过：词汇无词向量或已在候选词列表中 
                            continue
                            
                        #生成种子词向量
                        seed_vec = np.array([model.wv[w_] for w_ in category_seed_words.get(a) if w_ in model.wv])
                        #计算最大余弦相似度
                        score = np.max(cosine_similarity(seed_vec, np.expand_dims(model.wv[w], axis=0)), axis=0)[0]
                        
                        if score > threshold:   #超过阈值则保留
                            representatives.append(w)
                    category_representatives[a]= representatives   #保存当前类别的候选词

                    
                # -------------------- 构建输出结果 ------------------------
                current_data= dict()  #构建当前样本的最终结果字典
                #替换地点标记为特殊格式
                current_data['text']= text.replace('LOCATION1', 'LOCATION - 1 -').replace('LOCATION2', 'LOCATION - 2 -').lower()  
                asp_tem =[]   #存储每个"地点-类别"对
                loc=dict()   #存储地点标记映射（如"LOCATION1": "location - 1 -"）
                
                #处理地点标记：为每个地点标记后加空格，并存储映射关系
                for lc, v in zip(['LOCATION1', 'LOCATION2'], ['location - 1 -', 'location - 2 -']):
                    if lc in text:
                        text = text.replace(lc, lc+' ')   #地点标记后加空格
                        loc[lc] = v   #记录地点标记映射关系
                aspect_with_loc_dist=[]   #存储"地点标记-类别"对

                
                #处理多地点（LOCATION1和LOCATION2同时存在）的情况：通过词距（候选词与地点标记的距离）匹配类别对应的地点
                if len(loc)>1:   #包含两个地点（LOCATION1和LOCATION2）
                    text =  re.sub(r'[^\w\s]', ' ', text.strip().lower()).split()   #文本预处理：去除标点 → 分词
                    for aspect in categories:
                        if aspect not in  category_representatives:
                            continue   #跳过无候选词的类别
                        aux = category_representatives.get(aspect)    #当前类别的候选词
                        aux_loc={lc:[] for lc in loc.keys() }    #存储候选词到每个地点的距离
                        #遍历候选词
                        for w in aux:
                            if w not in text:
                                continue   #候选词不在文本中，跳过
                            for lc in loc.keys():
                                #计算候选词与地点标记在分词列表中的距离
                                aux_loc[lc].append(abs(text.index(w)- text.index(lc.lower()))) 

                        #跳过无有效距离的情况
                        if not len(aux_loc['LOCATION1']):
                            continue
                        #比较候选词到两个地点的平均距离，距离更近的地点即为该类别的目标地点
                        if np.mean(aux_loc['LOCATION1']) < np.mean(aux_loc['LOCATION2']):
                            aspect_with_loc_dist.append(('LOCATION1', aspect))
                        else:
                            aspect_with_loc_dist.append(('LOCATION2', aspect))

                            
                #构建每个“地点-类别“对的详细信息
                for lc,v in loc.items():
                    for aspect in categories:
                        #检查：当前类别是否有候选词，且对应的"地点#类别"有极性
                        if aspect in category_representatives and aspect_polarity.get(lc+'#'+aspect) in['Positive','Negative','Neutral']: 
                            #关联候选词：若匹配到的地点-类别对或仅有一个地点，则使用候选词；否则为空
                            if (lc, aspect) in aspect_with_loc_dist or len(loc)==1:
                                aux = category_representatives.get(aspect)
                            else:
                                aux=[]   #未匹配的地点-类别对：语义候选词为空
                            aux = list(set(aux))   #候选词去重
                            #构建当前"地点-类别"对的信息（含目标地点标记）
                            temp={'category':aspect, 
                                  'polarity':aspect_polarity.get(lc+'#'+aspect), 
                                  'auxiliary':aux, 
                                  'target':v
                            }
                        else:
                            #无有效标注的"地点-类别"对：极性设为"None"，候选词为空
                            temp = {'category': aspect, 
                                    'polarity': 'none',
                                    'auxiliary': [] , 
                                    'target':v
                            }
                        asp_tem.append(temp)
                        
                current_data['aspect']=asp_tem   #关联所有的地点-类别信息
                data_to_save.append(current_data)    #加入最终结果列表
                
        #保存结果
        with open('../datasets/sentihood/bert_{}.json'.format(phase), 'w') as f :
            json.dump(data_to_save, f, indent=3)
        f.close()



def main():
    parser = argparse.ArgumentParser()   #创建参数解析器
    parser.add_argument('--dataset', default='semeval', type=str, choices=['semeval','sentihood'],)  
    opt = parser.parse_args()


    #根据数据集调用对应函数
    if opt.dataset =='semeval':   
        semeval()
    else:
        sentihood()


if __name__ == '__main__':
    main()

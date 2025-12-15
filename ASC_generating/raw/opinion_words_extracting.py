# =============================================================================================================
# 生成句法候选词
# =============================================================================================================
# 功能：根据依存句法树，从句子中提取与每个 aspect 相关的 opinions （句法候选词）
# 模块说明：
# semeval 数据集：自动启动 CoreNLP Server（无需手动）
# sentihood 数据集：使用 CoreNLPDependencyParser(url='http://localhost:9000')，必须手动启动 CoreNLP Server
# （1） cd /hy-tmp/BERT-ASC-main/stanford-corenlp-4.5.10
# （2） java -mx16g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \    -port 9000 -timeout 600000
# ============================================================================================================


import os
os.environ["CORENLP_HOME"] = "/hy-tmp/BERT-ASC-main/stanford-corenlp-4.5.10"   #指定 Stanford CoreNLP 的路径
import nltk 
nltk.data.path.append("/hy-tmp/BERT-ASC-main/nltk_data")   #指定NLTK数据路径 （本地放 punkt 和 stopwords 的目录）
import  json  #处理JSON文件
import argparse   #命令行参数
from nltk.corpus import stopwords   #停用词
from nltk.parse.corenlp import CoreNLPDependencyParser,CoreNLPParser   #导入句法分析工具
from tqdm import tqdm   
stops = stopwords.words('english')   #加载英文停用词
#扩展停用词表（包含领域无关词和地点标记）
stops.extend([
    'us', "didnt",'im','couldnt','even','shouldnt','ive','make','today',
    'feel','sometimes','ive','whatever','never','although','anyway',
    'get','always','usually','want','go','would','one'
])
stops.extend(['location - 1 -', 'location - 2 -'])



def extract_opnion_words(dataset='semeval'):  
    """ 为semeval数据集提取句法候选词（使用Stanford CoreNLPClient）"""
    from stanfordnlp.server import CoreNLPClient  
    #启动CoreNLP服务（指定需要的注解器：分词、分句、词性标注、依存句法分析）
    with CoreNLPClient(
        annotators=['tokenize', 'ssplit', 'pos', 'depparse'], 
        timeout=60000, 
        memory='16G',
        start_server=True,    # ------自动启动服务器
    ) as client:   
        #定义数据集对应的子文件（训练集/测试集/验证集）和关注的方面类别
        files={'semeval':[ 'train', 'test'],'sentihood':[ 'train', 'test', 'dev'] }
        aspect_categories={'semeval':[ 'price', 'food', 'service', 'ambience'],
                           'sentihood':[ 'price', 'transit-location', 'safety','general'] 
                          }
        #遍历数据集的所有子文件
        for subFile in files.get(dataset):  
            with open('../datasets/{0}/bert_{1}.json'.format(dataset, subFile)) as f:   #加载语义候选词处理后的JSON数据
                data= json.load(f)  #加载原始文本
                new_data=[]    #存储处理后的结果（添加句法候选词后的新数据）
                #遍历每个样本
                for d in tqdm(data):  
                    text, aspects = d.get('text'), d.get('aspect')    #文本和方面信息
                    ann = client.annotate(text)   #对文本进行句法分析：获取分词、词性、依存关系等注解结果
                    dp_rel = get_parse(ann)   #从注解结果中提取依存关系三元组（源词、关系类型、目标词）

                    
                    #为每个方面类别提取句法侯选词
                    for i in range(len(aspects)):
                        current_aspect = aspects[i]    #当前方面
                        auxiliary= current_aspect['auxiliary'].copy()    #语义候选词
                        
                        #如果方面类别名称在文本中，则加入语义候选词列表
                        if current_aspect['category'] in text:
                            auxiliary.append(aspects[i]['category'])
                        opnions = []   #存储当前的句法候选词
                        
                        #遍历每个语义候选词，从依存关系中提取相关修饰词
                        for w in auxiliary:
                            if w not in text or w in opnions:   
                                continue    #跳过：候选词不在文本中，或已添加到句法候选词列表（去重）
                        
                            #遍历所有依存关系，筛选关键关系类型
                            for rel in dp_rel:
                                l, m, r = rel    # l=左词（源词），m=依存关系类型，r=右词（目标词）
                                candidates = [l, r]   #候选词为依存关系连接的两个词
                                #筛选重要依存关系
                                if m in ['nsubj', 'amod', 'advmod', 'ccomp', 'compound'] and w in candidates:    
                                    del candidates[candidates.index(w)]   #移除当前语义候选词本身（只保留关联的修饰词）
                                    opnions.extend(candidates)   #将修饰词加入句法候选词列表
                                    
                        #句法候选词去重+过滤：排除停用词、语义候选词、方面类别名称
                        opnions= list(set(opnions).difference(set(stops+auxiliary+aspect_categories.get(dataset))))  
                        opnions= sort_auxiliary(text, opnions)     #按候选词在文本中出现的顺序排序
                        aspects[i]['opinions']= opnions    #将句法候选词存入当前方面信息中
 
                    new_data.append({'text': text, 'aspect':aspects})    #将添加句法候选词后的样本加入新数据列表

                with open('../datasets/{0}/bert_{1}.json'.format(dataset, subFile), 'w') as f:   #保存包含句法候选词的结果
                    json.dump(new_data, f, indent=3)
                f.close()



def extract_opnion_words_sentihood(dataset='sentihood'):  
    """ 为sentihood数据集提取句法候选词（使用Stanford CoreNLPClient）"""
    parser = CoreNLPDependencyParser(url='http://localhost:9000')     #连接本地启动的CoreNLP服务（默认端口9000）------需手动启动服务
    files={'semeval':[ 'train', 'test'], 'sentihood':[ 'train', 'test', 'dev']}    #数据集文件
    #遍历当前数据集的所有子文件
    for subFile in files.get(dataset):   
        with open('../datasets/{0}/bert_{1}.json'.format(dataset, subFile)) as f :   #加载语义候选词处理后的JSON文件
            data= json.load(f)  
            new_data=[]   #存储处理后的新数据
            #遍历每个样本
            for d in tqdm(data):   
                text, aspects = d.get('text'), d.get('aspect')   #文本和方面信息
                dp_tree, = parser.raw_parse(text)   #对文本进行句法分析，生成依存树
                dp_rel = list(dp_tree.triples())    #提取依存关系三元组（(左词, 词性), 关系类型, (右词, 词性)）

                
                #为每个"地点-方面"对提取句法候选词
                for i in range(len(aspects)):  
                    current_aspect = aspects[i]  
                    auxiliary= current_aspect['auxiliary'].copy()   #复制语义候选词
                    if current_aspect['category'] in text:    #如果方面类别名称在文本中，加入语义候选词列表
                        auxiliary.append(aspects[i]['category'])
                    opnions = []   #存储句法候选词
                    
                    #遍历每个语义候选词，提取依存关系中的关联词
                    for w in auxiliary:
                        if w not in text or w in opnions:   
                            continue    #跳过：候选词不在文本中或已添加（去重）
                        #遍历所有依存关系
                        for rel in dp_rel:   
                            l, m, r = rel   # l和r是(词, 词性)元组，m是依存关系类型
                            candidates = [l[0], r[0]]    #候选词为左右词
                            #筛选关键依存关系
                            if m in ['nsubj', 'amod', 'advmod', 'ccomp', 'compound'] and w in candidates: 
                                del candidates[candidates.index(w)]    #移除当前语义候选词本身
                                opnions.extend(candidates)    #添加修饰词为候选词
                         
                    opnions= list(set(opnions).difference(set(stops)))   #句法候选词去重+过滤停用词
                    opnions= sort_auxiliary(text, opnions)   #按文本中出现顺序排序
                    aspects[i]['opinions']= opnions   #保存句法候选词
                new_data.append({'text': text, 'aspect':aspects})    #将处理后的样本加入新数据

                
            f.close()
            with open('../datasets/{0}/bert_{1}.json'.format(dataset, subFile), 'w') as f:  #保存包含句法候选词的结果
                json.dump(new_data, f, indent=3)
            f.close()


def sort_auxiliary(text_a, text_b):
    """
    按 text_a 的单词顺序对 text_b 排序：
    1. text_b 中存在于 text_a 的词 → 按 text_a 中的出现顺序排列
    2. text_b 中不存在于 text_a 的词 → 保留原始顺序，放结果末尾
    3. 保留 text_b 中的重复词
    """
    text_a_tokens = text_a.split()    #原始文本分词（基准顺序）
    #构建词→索引映射（O(n) 复杂度，比多次 text_a.index(w) 高效）
    word_to_index = {word: idx for idx, word in enumerate(text_a_tokens)}
    
    # 排序规则：
    # - 存在于 text_a 的词 → 用其在 text_a 中的索引排序
    # - 新增词 → 用“无穷大”作为索引（确保放末尾），同时保留原始顺序（sorted 是稳定排序）
    sorted_text_b = sorted(
        text_b,
        key=lambda w: word_to_index.get(w, float('inf'))
    )
    return sorted_text_b

    

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




def main():
    #命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='semeval', type=str,  choices=['semeval','sentihood'], help='semeval, sentihood', required=True)
    opt = parser.parse_args()
    
    #根据数据集选择对应的提取函数
    if opt.dataset == 'semeval':
        extract_opnion_words(dataset=opt.dataset)
    else:
        extract_opnion_words_sentihood(dataset=opt.dataset)



if __name__ == '__main__':
    main()





# ===========================================================================
# 语义候选词 auxiliary----Word2Vec + 种子词（enhanced seeds）
# ===========================================================================

import json   
import re     
import gensim     #用于加载Word2Vec模型
import pickle as pk 
import os   
import xml.etree.ElementTree     #解析semeval的XML文件
from nltk.tokenize import word_tokenize   #分词
from nltk.corpus import stopwords        #停用词
import logging   
import string     
import numpy as np   
from sklearn.metrics.pairwise import cosine_similarity    #计算余弦相似度
from tqdm import tqdm  
import argparse   
import nltk
nltk.data.path.append('/hy-tmp/BERT-ASC-main/nltk_data')    #指定NLTK数据路径 

#加载停用词
stops = stopwords.words('english')   
#扩展停用词表
stops.extend(['us', "didnt",'im','couldnt', 'make','today', 'feel', 'sometimes', 'ive', 'whatever','never','although','anyway','get','always', 'usually', 'want', 'go','would', 'one'])


# ---------------------- 配置日志 -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)



# ----------------------------- 为semeval数据集生成语义候选词 --------------------------
def semeval(model, threshold=.3):   
    """
    为SemEval数据集生成语义候选词
    model: 预加载的Word2Vec模型
    threshold: 余弦相似度阈值
    """
    data_root='../../datasets'   #数据集根目录
    #定义类别文件和种子词文件的路径（由run_LLDA_ConceptNet生成）
    categories_path = os.path.join(data_root, 'semeval', 'categories.pk')
    seeds_path = os.path.join(data_root, 'semeval', 'categories_seeds_lc.pk')
    #验证文件是否存在
    for path in [categories_path, seeds_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"必需的文件不存在: {path}")
    #加载类别和种子词
    categories = pk.load(open(categories_path, 'rb'))  
    category_seed_words = pk.load(open(seeds_path, 'rb'))   
    categories.append('anecdotes')     #补充'anecdotes'类别
    #原始数据路径
    path_base = os.path.join(data_root, 'raw', 'semeval')
    os.makedirs(os.path.join(data_root, 'semeval'), exist_ok=True)    #确保输出目录存在
   
    
    # ------------------- 处理训练集和测试集 --------------------------------
    for phase, file_src,  in zip(['train', 'test'], ['Restaurants_Train_v2.xml', 'Restaurants_Test_Gold.xml']): 
        data_to_save =[]     #存储当前阶段（训练集/测试集）的最终处理结果
        file_path = os.path.join(path_base, file_src)
        if not os.path.exists(file_path):
            logger.error(f"原始数据文件不存在: {file_path}")
            continue
        try:
            e = xml.etree.ElementTree.parse(file_path).getroot()   #解析XML（提取句子和标注信息）   
        except Exception as e:
            logger.error(f"解析XML文件失败: {str(e)}")
            continue

    
        reviews = e.findall('sentence')    #获取所有句子
        logger.info(f"开始处理SemEval {phase}集，共{len(reviews)}个句子")
        #遍历句子
        for review in tqdm(reviews, desc=f"SemEval {phase}处理进度"):  
            try:
                original_text = review.find('text').text    #存储原始文本
                if not original_text:    #如果文本为空，则跳过
                    continue   
                aspect_category = []      #存储所有方面类别
                aspect_polarity = dict()  #存储情感极性（键：类别，值：极性）
                options = review.findall('aspectCategories')    #提取方面类别相关标签
                #遍历标签信息
                for option in options:
                    suboptions = option.findall('aspectCategory')   #提取每个具体的方面类别
                    for suboption in suboptions:
                        current_aspect = suboption.get("category").replace('anecdotes/miscellaneous', 'anecdotes')    #简化类别名称
                        aspect_category.append(current_aspect)     #记录方面类别
                        aspect_polarity[current_aspect] = suboption.get("polarity")   #记录情感极性 
                
                #文本预处理：小写、去标点、分词、过滤停用词
                text = original_text.strip().lower()  
                text = text.translate(str.maketrans('', '', string.punctuation))    
                text_w_stops = [w for w in word_tokenize(text) if w not in stops]


                # ---------------------- 生成语义候选词 ----------------------
                #为当前句子中提到的每个方面类别生成候选词列表（键：类别，值：候选词列表）
                category_representatives={a:[] for a in aspect_category}
                for a in aspect_category:
                    if a =='anecdotes':
                        category_representatives[a].append('anecdotes')     #特殊处理"anecdotes"类别：直接添加类别名称作为候选词
                        continue  

                    representatives=[]    #存储当前类别的语义候选词
                    
                    # 1. 获取当前类别的有效种子词
                    valid_seeds = [w_ for w_ in category_seed_words.get(a, []) if w_ in model.wv]
                    if not valid_seeds: 
                        logger.debug(f"类别 {a} 没有有效的种子词，跳过")    #若无该类别的有效种子词，跳过
                        continue   
                        
                    # 2. 计算种子词的平均向量
                    seed_vec = np.array([model.wv[w_] for w_ in valid_seeds])  
                    
                    # 3. 遍历预处理后的句子中的每个词，计算与种子词的相似度
                    for w in text_w_stops:
                        if w not in model.wv or w in representatives :   #跳过：词不在Word2Vec模型中或已添加到候选词列表
                            continue   
                            
                        #计算当前词与种子词的余弦相似度
                        score = np.max(cosine_similarity(seed_vec, np.expand_dims(model.wv[w], axis=0)), axis=0)[0]
                        
                        #若得分超过阈值，则将当前词视为该类别的语义候选词
                        if score > threshold:   
                            representatives.append(w)
                            logger.debug(f"为类别 {a} 添加候选词: {w} (相似度: {score:.3f})")    

                    # 4. 候选词去重并去掉类别本身名称
                    representatives = list(set(representatives).difference(set(categories))) 
                    
                    category_representatives[a]= representatives    #保存候选词
                    logger.debug(f"类别 {a} 共生成 {len(representatives)} 个候选词")
                

                # ---------------------- 构建输出结构 ----------------------
                current_data = {
                    'text': original_text,    #原始文本
                    'aspect': []     #存储方面类别信息
                }
                #遍历所有可能的方面类别
                for aspect in categories:
                    if aspect in category_representatives: 
                        aux = list(set(category_representatives.get(aspect, [])))
                        if aspect not in aux:
                            aux.append(aspect)    # ★ 新增：确保加入类别名
                        #有标注的类别：填入极性和候选词
                        temp={'category':aspect,    #方面类别
                              'polarity':aspect_polarity.get(aspect),    #情感极性
                              'auxiliary': aux,
                              'validated_auxiliary_placeholder': ""    #预留字段：后续人工/自动验证候选词
                             }
                    else:    
                        #无标注的类别：极性设为"none"，候选词为空列表
                        temp = {'category': aspect, 
                                'polarity': 'none',   
                                'auxiliary': [aspect], 
                                'validated_auxiliary_placeholder': ""
                               }   
                    current_data['aspect'].append(temp)     #将该类别的信息加入当前句子的数据中
                data_to_save.append(current_data)    #将当前句子的处理结果加入到总结果列表中
                
            except Exception as e:
                logger.error(f"处理句子时出错: {str(e)}，跳过该句子")
                continue


        # ---------------------- 保存结果 ----------------------
        output_path = os.path.join(data_root, 'semeval', f'bert_{phase}_with_aux.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=3, ensure_ascii=False)
            
        # ----------------- 输出统计信息 -------------------------
        aux_counts = [len(a["auxiliary"]) for d in data_to_save for a in d["aspect"] if a["auxiliary"]]
        avg_aux = np.mean(aux_counts) if aux_counts else 0    #计算每个方面类别的平均候选词数量
        logger.info(f"SemEval平均候选词数: {avg_aux:.2f}")

        #按类别统计平均候选词数
        category_stats = {
            aspect['category']: len(aspect['auxiliary'])
            for d in data_to_save for aspect in d['aspect']
            if aspect['auxiliary']    #只统计有候选词的类别
        }
        if category_stats:
            #计算每个类别的平均候选词数
            avg_per_cat = {k: np.mean([v for kk, v in category_stats.items() if kk == k]) for k in set(category_stats.keys())}
            logger.info("[类别平均候选词数] " + ", ".join([f"{k}:{v:.2f}" for k, v in avg_per_cat.items()]))

        logger.info(f"SemEval {phase}集处理完成，保存 {len(data_to_save)} 条数据到 {output_path}")

       

# -------------------------- 为sentihood数据集生成语义候选词 --------------------------
def sentihood(model, threshold=.4):
    """
    为Sentihood数据集生成语义候选词
    model: 预加载的Word2Vec模型
    threshold: 余弦相似度阈值
    """
    # -------------------- 基础文件加载 --------------------
    data_root='../../datasets'
    #检查并加载种子词文件
    seeds_path = os.path.join(data_root, 'sentihood', 'categories_seeds_lc.pk')
    if not os.path.exists(seeds_path):
        raise FileNotFoundError(f"必需的文件不存在: {seeds_path}")
    category_seed_words = pk.load(open(seeds_path, 'rb'))    #加载种子词
    categories = {'general', 'price', 'transit-location', 'safety'}    #关注的类别
    os.makedirs(os.path.join(data_root, 'sentihood'), exist_ok=True)   #确保输出目录存在

    
    # -------------------- 遍历 train/test/dev --------------------
    for phase in ['train', 'test', 'dev']:  
        file_path = os.path.join(data_root, 'raw', 'sentihood', f'sentihood-{phase}.jsonl')    #原始数据集的路径
        if not os.path.exists(file_path):
            logger.error(f"原始数据文件不存在: {file_path}")
            continue
        data_to_save = []   #存储当前阶段的最终处理结果

        #逐行读取 jsonl 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = []     #存储解析后的原始样本（每个样本为字典）
            for line in f:
                line = line.strip()   #去除首尾空格
                if line:
                    try:
                        data.append(json.loads(line))   #解析每行的 JSON 数据
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析 JSON 行失败: {e}")
        logger.info(f"开始处理 Sentihood {phase} 集，共 {len(data)} 个样本")

        
        # =====================================================
        # 逐样本处理
        # =====================================================
        for d in tqdm(data, desc=f"Sentihood {phase} 处理进度"):
            try:
                original_text = d['text']    #提取原始文本
                aspects = d.get('opinions', [])    #提取方面信息

                #文本预处理：移除地点标记（LOCATION1/LOCATION2）→ 小写 → 去标点 → 分词 → 过滤停用词
                text = original_text.replace('LOCATION1', '').replace('LOCATION2', '')   
                text = text.strip().lower()
                text = text.translate(str.maketrans('', '', string.punctuation))
                text_w_stops = [w for w in word_tokenize(text) if w not in stops]

                
                # ---------------------- 提取方面类别和极性 ----------------------
                #初始化字典：键：方面类别，值：语义候选词
                category_representatives = {a['aspect']:[] for a in aspects if a['aspect'] in categories} 
                #记录方面类别
                aspect_category=[a['aspect'] for a in aspects if a['aspect'] in categories]   
                #记录极性:键："目标实体#类别"，值：情感极性
                aspect_polarity = {
                    a['target_entity'].lower() + '#' + a['aspect']: a['sentiment']
                    for a in aspects if a['aspect'] in categories
                }


                # =====================================================
                # 生成候选词
                # =====================================================
                for a in aspect_category:
                    representatives = []    #存储当前类别的候选词
                    
                    # 1. 获取当前类别的有效种子词
                    valid_seeds = [w_ for w_ in category_seed_words.get(a, []) if w_ in model.wv]
                    if not valid_seeds:
                        continue
                        
                    # 2. 提取种子词向量
                    seed_vec = np.array([model.wv[w_] for w_ in valid_seeds])  

                    # 3. 遍历预处理后的词，计算与种子词的相似度
                    for w in text_w_stops:   
                        if w not in model.wv or w in representatives:     #跳过：无词向量或已在候选词列表中
                            continue
                            
                        #计算最大余弦相似度
                        score = np.max(cosine_similarity(seed_vec, np.expand_dims(model.wv[w], axis=0)), axis=0)[0]
                        if score > threshold:    #超过阈值则视为候选词
                            representatives.append(w)
                            logger.debug(f"为类别 {a} 添加候选词: {w} (相似度: {score:.3f})")

                    # 4. 去重并保存候选词
                    category_representatives[a] = list(set(representatives))     #去重
                    logger.debug(f"类别 {a} 共生成 {len(representatives)} 个候选词")


                # =====================================================
                # 构建 LOCATION 映射
                # =====================================================
                loc_map = {}  #存储地点标记映射：键为原始地点标记（LOCATION1/LOCATION2），值为标准化名称（location - 1 -/location - 2 -）
                for lc, std in zip(["LOCATION1", "LOCATION2"], ["location - 1 -", "location - 2 -"]):
                    if lc in original_text:
                        loc_map[lc.lower()] = std
                
                #替换文本用于输出（替换地点标记为特殊格式）
                text_with_locations = (
                    original_text.replace('LOCATION1', 'LOCATION - 1 -')
                                 .replace('LOCATION2', 'LOCATION - 2 -')
                                 .lower()
                )
                
                #一次性构建 current_data
                current_data = {   
                    'text': text_with_locations,
                    'aspect': []
                }
                

                # ======================================================
                # Case A：没有 LOCATION → target="none"
                # ======================================================
                if len(loc_map) == 0:
                    for aspect in categories:
                        key_matches = [k for k in aspect_polarity.keys() if k.endswith(f"#{aspect}")]
                        polarity = aspect_polarity[key_matches[0]] if key_matches else "None"
                        aux = category_representatives.get(aspect, []).copy()
                        if aspect not in aux:
                            aux.append(aspect)

                        current_data["aspect"].append({
                            "category": aspect,
                            "polarity": polarity,
                            "auxiliary": aux,
                            "target": "none",
                            "validated_auxiliary_placeholder": ""
                        })

                    data_to_save.append(current_data)
                    continue
                

                # ======================================================
                # Case B：有 LOCATION1 和 LOCATION2 → 距离分配
                # ======================================================
                aspect_with_loc_dist = []   #存储"地点标记-类别"对
                if len(loc_map) > 1:    
                    # ------------------ 构建 tokens -----------------
                    text_for_dist = re.sub(r"[^\w\s]", " ", original_text.lower())
                    tokens = text_for_dist.split()    #文本分词
                    for aspect in categories:
                        if aspect not in category_representatives:    #该类别无候选词，跳过
                            continue 
                        aux = category_representatives.get(aspect, [])  #当前类别的候选词
                        if not aux:
                            continue
                        #存储每个地点与候选词的距离（keys：'location1', 'location2'  值：距离列表）
                        aux_loc = {lc: [] for lc in loc_map.keys()}    
                        #遍历候选词
                        for w in aux:
                            if w not in tokens:   #候选词不在文本中，跳过 
                                continue   
                            #遍历每个地点
                            for lc in loc_map.keys():   # lc：'location1'
                                if lc not in tokens:
                                    continue
                                try:
                                    distance = abs(tokens.index(w) - tokens.index(lc))   #计算候选词与地点标记的位置距离  
                                    aux_loc[lc].append(distance)   #记录距离
                                except ValueError:
                                    continue
                                    
                        #必须两个地点都有距离
                        if not aux_loc['location1'] or not aux_loc['location2']:
                            continue
                        #比较两个地点的平均距离，将类别分配到平均距离更小的地点
                        if np.mean(aux_loc['location1']) < np.mean(aux_loc['location2']):
                            aspect_with_loc_dist.append(('location1', aspect))
                        else:
                            aspect_with_loc_dist.append(('location2', aspect))


                # =====================================================
                # 构建最终 aspect 结构
                # =====================================================
                #遍历地点标记
                for lc, std in loc_map.items():   # lc：'location1'
                    for aspect in categories:
                        polarity = aspect_polarity.get(lc + '#' + aspect, 'None')
                        #检查类别是否有效且有极性标注
                        if aspect in category_representatives and polarity in ['Positive', 'Negative', 'Neutral']:
                            #关联候选词：若匹配到的地点-类别对或仅有一个地点，则使用候选词
                            if (lc, aspect) in aspect_with_loc_dist or len(loc_map) == 1:
                                aux = category_representatives.get(aspect, []).copy() 
                            else:   
                                aux = []    #未匹配的地点-类别对：语义候选词为空
                            aux = list(set(aux))   #候选词去重
                            if aspect not in aux:
                                aux.append(aspect)     #加入类别名
                                
                            #构建当前地点-类别的详细信息
                            temp = {
                                'category': aspect,
                                'polarity': polarity, 
                                'auxiliary': aux,
                                'target': std,     
                                'validated_auxiliary_placeholder': ""
                            }
                        else:
                            temp = {
                                'category': aspect,
                                'polarity': 'None',
                                'auxiliary': [aspect], 
                                'target': std,
                                'validated_auxiliary_placeholder': ""
                            }
                        current_data['aspect'].append(temp)   #加入当前样本的结果
                data_to_save.append(current_data)    #加入阶段总结果

            except Exception as e:
                logger.error(f"处理样本时出错: {str(e)}，跳过该样本")
                continue

        # ------------------ 保存结果 -------------------------------
        output_path = os.path.join(data_root, 'sentihood', f'bert_{phase}_with_aux.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=3, ensure_ascii=False)
            
        # ---------------------- 输出统计信息 ----------------------
        aux_counts = [len(a["auxiliary"]) for d in data_to_save for a in d["aspect"] if a["auxiliary"]]
        avg_aux = np.mean(aux_counts) if aux_counts else 0
        logger.info(f"Sentihood 平均候选词数: {avg_aux:.2f}，共生成 {sum(aux_counts)} 个候选词")
        
        #按类别统计候选词
        category_stats = {
            aspect['category']: len(aspect['auxiliary'])
            for d in data_to_save for aspect in d['aspect']
            if aspect['auxiliary']
        }
        if category_stats:
            avg_per_cat = {
                k: np.mean([v for kk, v in category_stats.items() if kk == k])
                for k in set(category_stats.keys())
            }
            logger.info("[类别平均候选词数] " + ", ".join([f"{k}:{v:.2f}" for k, v in avg_per_cat.items()]))
            
        logger.info(f"Sentihood {phase} 处理完成，保存 {len(data_to_save)} 条数据到 {output_path}")
                        


def main():
    parser = argparse.ArgumentParser(description='为SemEval/Sentihood数据集生成语义候选词') 
    parser.add_argument('--dataset', required=True, type=str, choices=['semeval', 'sentihood'],help='指定数据集：semeval 或 sentihood')
    parser.add_argument('--embedding_path', 
                        default='/hy-tmp/BERT-ASC-main/ASC_generating/embeddings/restaurant.bin', 
                        type=str, 
                        help='Word2Vec词向量模型路径'
    )
    parser.add_argument('--threshold', type=float, help='余弦相似度阈值（semeval默认0.3，sentihood默认0.4，不指定则使用默认值）')
    opt = parser.parse_args()

    
    #验证词向量模型是否存在
    if not os.path.exists(opt.embedding_path):
        raise FileNotFoundError(f"词向量模型不存在: {opt.embedding_path}")

        
    #加载Word2Vec词向量模型
    try:
        logger.info(f"开始加载词向量模型: {opt.embedding_path}")
        if opt.embedding_path.endswith('.vec'):
            model = gensim.models.KeyedVectors.load_word2vec_format(opt.embedding_path, binary=False)
        elif opt.embedding_path.endswith('.bin'):
            try:
                model = gensim.models.KeyedVectors.load_word2vec_format(opt.embedding_path, binary=True)
            except Exception:
                logger.info("检测到 Gensim Word2Vec 模型格式 (.bin pickle)，尝试使用 Word2Vec.load() 重新加载")
                model = gensim.models.Word2Vec.load(opt.embedding_path)
        else:
            model = gensim.models.Word2Vec.load(opt.embedding_path)
        logger.info("词向量模型加载完成")
    except Exception as e:
        logger.error(f"加载词向量模型失败: {str(e)}")
        return

    
    #设置默认阈值
    threshold = opt.threshold
    if threshold is None:
        threshold = 0.3 if opt.dataset == 'semeval' else 0.4
        logger.info(f"使用默认阈值: {threshold}")

        
    #根据数据集调用对应函数
    try:
        if opt.dataset == 'semeval':
            semeval(model, threshold)
        else:
            sentihood(model, threshold)
    except Exception as e:
        logger.error(f"处理数据集时发生错误: {str(e)}")
    logger.info(f"[INFO] {opt.dataset} 数据集语义候选词生成完毕")



if __name__ == '__main__':
    main()
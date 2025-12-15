import json
import argparse
from nltk.corpus import stopwords
# 新增SenticNet相关导入
from senticnet.senticnet import SenticNet

stops = stopwords.words('english')
stops.extend(['us', "didnt",'im','couldnt','even', 'shouldnt','ive','make','today', 'feel', 'sometimes', 'ive' 'whatever','never','although','anyway','get','always', 'usually', 'want', 'go','would', 'one'])
stops.extend(['location - 1 -', 'location - 2 -'])

from nltk.parse.corenlp import CoreNLPDependencyParser,CoreNLPParser
from tqdm import tqdm

# 初始化SenticNet
sn = SenticNet()

# 新增情感分数获取函数
def get_sentiment_score(word):
    """获取词的情感分数，范围[-1, 1]"""
    try:
        # 获取情感极性分数
        polarity = sn.polarity_value(word)
        return float(polarity)
    except:
        # 处理未找到的词
        return 0.0

# 新增带情感权重的字符串格式化函数
def format_with_sentiment(word, rel, score):
    """格式化带情感权重的句法关系字符串"""
    return f"{word}_{rel} ({score:.2f})"

def extract_opnion_words(dataset='semeval'):
    from stanfordnlp.server import CoreNLPClient

    with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'depparse'], timeout=60000, memory='16G') as client:
        files={'semeval':[ 'train', 'test'],'sentihood':[ 'train', 'test', 'dev'] }
        aspect_categories={'semeval':[ 'price', 'food', 'service', 'ambience'],'sentihood':[ 'price', 'test', 'dev'] }
        
        for subFile in files.get(dataset):
            with open('../datasets/{0}/bert_{1}_{2}.json'.format(dataset, subFile,'')) as f :
                data= json.load(f)
                new_data=[]
                for d in tqdm(data):
                    text, aspects = d.get('text'), d.get('aspect')
                    ann = client.annotate(text)
                    dp_rel = get_parse(ann)

                    for i in range(len(aspects)):
                        current_aspect = aspects[i]
                        auxiliary= current_aspect['auxiliary'].copy()
                        if current_aspect['category'] in text:
                            auxiliary.append(aspects[i]['category'])
                        opnions = []
                        for w in auxiliary:
                            if w not in text or w in opnions:
                                continue
                            # 提取修饰词并添加情感权重
                            for rel in dp_rel:
                                l, m, r = rel
                                # 只保留核心句法关系：amod/nsubj/advmod
                                if m not in ['nsubj', 'amod', 'advmod']:
                                    continue
                                    
                                candidates = [l, r]
                                if w in candidates:
                                    del candidates[candidates.index(w)]
                                    # 为每个候选词获取情感分数并格式化
                                    for candidate in candidates:
                                        score = get_sentiment_score(candidate)
                                        opnions.append(format_with_sentiment(candidate, m, score))

                        # 去重并过滤停用词
                        opnions= list(set(opnions).difference(set(stops+auxiliary+aspect_categories.get(dataset))))
                        opnions= sort_auxiliary(text, opnions)
                        aspects[i]['opinions']= opnions

                    new_data.append({'text': text, 'aspect':aspects})
                f.close()
                with open('../../datasets/{0}/bert_{1}.json'.format(dataset, subFile), 'w') as f:
                    json.dump(new_data, f, indent=3)
                f.close()

def extract_opnion_words_sentihood(dataset='sentihood'):
    parser = CoreNLPDependencyParser(url='http://localhost:9000')
    files={'semeval':[ 'train', 'test'], 'sentihood':[ 'train', 'test', 'dev']}

    for subFile in files.get(dataset):
        with open('../datasets/{0}/bert_{1}.json'.format(dataset, subFile)) as f :
            data= json.load(f)
            new_data=[]
            for d in tqdm(data):
                text, aspects = d.get('text'), d.get('aspect')
                dp_tree, = parser.raw_parse(text)
                dp_rel = list(dp_tree.triples())

                for i in range(len(aspects)):
                    opnions = []
                    current_aspect = aspects[i]
                    auxiliary= current_aspect['auxiliary'].copy()
                    if current_aspect['category'] in text:
                        auxiliary.append(aspects[i]['category'])
                    
                    for w in auxiliary:
                        if w not in text or w in opnions:
                            continue
                        # 提取修饰词并添加情感权重
                        for rel in dp_rel:
                            l, m, r = rel
                            # 只保留核心句法关系：amod/nsubj/advmod
                            if m not in ['nsubj', 'amod', 'advmod']:
                                continue
                                
                            candidates = [l[0], r[0]]
                            if w in candidates:
                                del candidates[candidates.index(w)]
                                # 为每个候选词获取情感分数并格式化
                                for candidate in candidates:
                                    score = get_sentiment_score(candidate)
                                    opnions.append(format_with_sentiment(candidate, m, score))
                    
                    opnions= list(set(opnions).difference(set(stops)))
                    opnions= sort_auxiliary(text, opnions)
                    aspects[i]['opinions']= opnions
                new_data.append({'text': text, 'aspect':aspects})
            f.close()
            with open('../../datasets/{0}/bert_{1}.json'.format(dataset, subFile), 'w') as f:
                json.dump(new_data, f, indent=3)
            f.close()

def sort_auxiliary( text_a, text_b):
    text_a = text_a.split()
    arr = [text_a.index(w.split('_')[0]) if w.split('_')[0] in text_a else len(text_a) for w in text_b]
    arr = sorted(arr)
    return [text_a[k] if k !=  len(text_a) else ' '.join(set(text_b).difference(set(text_a))) for k in arr]

def get_parse(ann):
    sentence = ann.sentence[0]
    dependency_parse = sentence.enhancedDependencies
    token_dict = {}
    for i in range(0, len(sentence.token)):
        token_dict[sentence.token[i].tokenEndIndex] = sentence.token[i].word

    list_dep = []
    for i in range(0, len(dependency_parse.edge)):
        source_node = dependency_parse.edge[i].source
        source_name = token_dict[source_node]

        target_node = dependency_parse.edge[i].target
        target_name = token_dict[target_node]

        dep = dependency_parse.edge[i].dep

        list_dep.append((source_name, dep, target_name))
    return list_dep

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='semeval', type=str,  choices=['semeval','sentihood'], help='semeval, sentihood', required=True)
    opt = parser.parse_args()
    extract_opnion_words(dataset=opt.dataset)

if __name__ == '__main__':
    main()
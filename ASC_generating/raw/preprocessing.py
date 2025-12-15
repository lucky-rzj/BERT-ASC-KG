#文本预处理脚本----对原始训练文本进行清洗、分词、去停用词和词形还原


from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import codecs
import os

def parseSentence(line):
    """
    单句预处理函数：接收一行文本，执行分词、去停用词、词形还原
    """
    lmtzr = WordNetLemmatizer()    #初始化词形还原器（Lemmatizer）
    stop = stopwords.words('english')
    
    text_token = CountVectorizer().build_tokenizer()(line.lower())   #转为小写
    text_rmstop = [i for i in text_token if i not in stop]   #去除停用词
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]   #词形还原
    return text_stem   #返回预处理后的单词列表


def preprocess_train(domain):
    """
    训练集预处理函数：读取原始训练文本，批量执行预处理并保存结果
    """
    in_path = codecs.open('datasets/{}/train.txt'.format(domain), 'r', 'utf-8')
    os.makedirs('preprocessed_data/{}'.format(domain), exist_ok=True)
    out_path = codecs.open('preprocessed_data/{}/train.txt'.format(domain), 'w', 'utf-8')
   
    for line in f:
        tokens = parseSentence(line)  #调用单句预处理函数
        if len(tokens) > 0:
            out.write(' '.join(tokens)+'\n')
    in_path.close()
    out_path.close()

    print(f"✅ 已生成预处理文件：{out_path}")



def preprocess(domain):
    """
    总预处理函数：统一调用训练集预处理（可扩展测试集、验证集预处理）
    """
    preprocess_train(domain)
   

if __name__ == '__main__':
    preprocess('restaurant')



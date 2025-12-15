# 用于预训练领域词向量（Word2Vec embeddings）的脚本
# 核心任务：读取指定领域的预处理文本（train.txt），用 Gensim 训练词向量模型，并保存为 .bin 文件

import gensim    #导入gensim库，用于训练词向量（Word2Vec）
import codecs
import os

class Sentence(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            yield line.split()


def main(domain):
    source = f'preprocessed_data/{domain}/train.txt'
    model_file = f'embeddings/{domain}.bin'
   
    os.makedirs('embeddings', exist_ok=True)
    
    sentences = Sentence(source)
    #训练Word2Vec模型
    model = gensim.models.Word2Vec(sentences, vector_size=200, window=5, min_count=10, workers=4)
    model.save(model_file)
    print(f"✅ Word2Vec 模型已保存至：{model_file}")



if __name__ == '__main__':
    print ('Pre-training word embeddings ...')
    main('restaurant')
  

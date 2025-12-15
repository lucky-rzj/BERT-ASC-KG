#“用预处理数据训练模型 + 用训练好的模型做主题推理” 的一体化脚本，核心产出是训练好的 L-LDA 模型和各方面类别的核心主题词

import random
import sys
import pickle as pk
sys.path.append('../')    #将上级目录添加到系统路径
import model.labeled_lda as llda


# -----------------------
# 训练L-LDA模型
# -----------------------
def train():  
    #初始化示例数据
    labeled_documents_train = [("example example example example example"*10, ["example"]),
                         ("test llda model test llda model test llda model"*10, ["test", "llda_model"]),
                         ("example test example test example test example test"*10, ["example", "test"]),
                         ("good perfect good good perfect good good perfect good "*10, ["positive"]),
                         ("bad bad down down bad bad down"*10, ["negative"])]
    #加载预处理后的训练集和测试集
    labeled_documents_train  = pk.load(open('data/semeval/train.pk', 'rb'))
    labeled_documents_test  = pk.load(open('data/semeval/test.pk', 'rb'))
    #转换格式为 L-LDA 模型要求的（文档内容，标签列表）
    labeled_documents_train = [(v,k.split()) for k,v in labeled_documents_train.items()]  
    labeled_documents_test = [(v,k.split()) for k,v in labeled_documents_test.items()]    
    
    #初始化L-LDA模型
    llda_model = llda.LldaModel(labeled_documents=labeled_documents_train, alpha_vector=0.01)
    print(llda_model)

    
    #训练模型----第一阶段训练
    while True:
        print("iteration %s sampling..." % (llda_model.iteration + 1))    #打印当前迭代采样信息
        llda_model.training(1)
        #打印训练指标（困惑度：越低拟合越好；delta beta：主题-词分布变化量）
        print("after iteration: %s, perplexity: %s" % (llda_model.iteration, llda_model.perplexity()))
        print("delta beta: %s" % llda_model.delta_beta)
        if llda_model.iteration > 600:
            break

    #模型更新----用测试集数更新（提升泛化能力）
    print("before updating: ", llda_model)
    update_labeled_documents =random.sample(labeled_documents_test, k=5)   #随机采样5个测试集文档
    llda_model.update(labeled_documents=update_labeled_documents)
    print("after updating: ", llda_model)

    #重新训练模型----第二阶段训练
    while True:
        print("iteration %s sampling..." % (llda_model.iteration + 1))
        llda_model.training(1)
        print("after 1 iteration: %s, perplexity: %s" % (llda_model.iteration, llda_model.perplexity()))
        print("delta beta: %s" % llda_model.delta_beta)
        if llda_model.iteration >1200:
            break
            
    
    #模型推理测试（验证模型效果）
    # note: the result topics may be different for difference training, because gibbs sampling is a random algorithm
    document = "example llda model example example good perfect good perfect good perfect" * 100
    document=random.sample(labeled_documents_test, k=1)[0][0]   #随机选1个测试集文档作为推理样本
    print(document)
    topics = llda_model.inference(document=document, iteration=100, times=10)    #执行推理：迭代100次，重复10次
    print(topics)

    
    #保存模型
    save_model_dir = "data/model"
    llda_model.save_model_to_dir(save_model_dir)
    print(f"\n模型已保存至：{save_model_dir}")

    
    #从磁盘加载模型
    llda_model_new = llda.LldaModel()
    llda_model_new.load_model_from_dir(save_model_dir, load_derivative_properties=False)
    

    #打印各主题的前10个核心词
    print("Top-5 terms of topic 'food': ", llda_model_new.top_terms_of_topic("food", 10, False))
    print("Top-5 terms of topic 'price': ", llda_model_new.top_terms_of_topic("price", 10, False))
    print("Top-5 terms of topic 'ambience': ", llda_model_new.top_terms_of_topic("ambience",10, False))
    print("Top-5 terms of topic 'service': ", llda_model_new.top_terms_of_topic("service", 10, False))
    print("Top-5 terms of topic 'anecdotes/miscellaneous': ", llda_model_new.top_terms_of_topic("anecdotes/miscellaneous", 10, False))
   
    

# -----------------------
# 独立推理函数
# -----------------------   
def inference():  
    save_model_dir = "data/model"
    #加载训练好的模型
    llda_model = llda.LldaModel()
    llda_model.load_model_from_dir(save_model_dir)
    print(f"已从 {save_model_dir} 加载模型！")
   
    n= 15  #显示主题的前15个词
    print("Top-5 terms of topic 'food': ", llda_model.top_terms_of_topic("food", 10, False))
    print("Top-5 terms of topic 'price': ", llda_model.top_terms_of_topic("price", n, False))
    print("Top-5 terms of topic 'ambience': ", llda_model.top_terms_of_topic("ambience", n, False))
    print("Top-5 terms of topic 'service': ", llda_model.top_terms_of_topic("service", n, False))
    print("Top-5 terms of topic 'anecdotes/miscellaneous': ",llda_model.top_terms_of_topic("anecdotes/miscellaneous", n, False))



if __name__ == '__main__':
    inference()     # ← 默认执行推理
    # train()       # ← 取消注释这一行可执行训练

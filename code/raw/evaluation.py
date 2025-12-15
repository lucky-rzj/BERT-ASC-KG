# =============================================================================
# 评估指标计算模块-----提供具体的指标计算逻辑
# =============================================================================

import argparse   #用于解析命令行参数
import collections   #提供容器数据类型，如有序字典等
import numpy as np  #用于数值计算的库
import pandas as pd  #用于数据处理和分析的库
from sklearn import metrics  #用于计算评估指标的工具库
from sklearn.preprocessing import label_binarize   #用于将标签二值化


# ----------------- Sentihood Strict Accuracy --------------------------
def sentihood_strict_acc(y_true, y_pred): 
    """
    Calculate "strict Acc" of aspect detection task of Sentihood.
    """
    total_cases = int(len(y_true) / 4)  #总样本数=标签长度/4（每个样本包含4个方面）
    true_cases = 0  #完全正确的样本数
    for i in range(total_cases):  #遍历总样本数，检查当前样本的4个方面是否全部预测正确
        if y_true[i * 4] != y_pred[i * 4]: continue
        if y_true[i * 4 + 1] != y_pred[i * 4 + 1]: continue
        if y_true[i * 4 + 2] != y_pred[i * 4 + 2]: continue
        if y_true[i * 4 + 3] != y_pred[i * 4 + 3]: continue
        true_cases += 1  #全部正确则计数+1
    aspect_strict_Acc = true_cases / total_cases   #计算严格准确率=完全正确的样本数/总样本数
    return aspect_strict_Acc


# ------------------- Sentihood  Micro-F1  --------------------------
def sentihood_macro_F1(y_true, y_pred):  
    """
    Calculate "Macro-F1" of aspect detection task of Sentihood.
    """
    p_all = 0   #所有样本的精确率总和
    r_all = 0   #所有样本的召回率总和
    count = 0   #有效样本数（真是标签非空的样本）
    
    s_all = 0   #所有预测的正例总数
    g_all = 0   #所有真实的正例总数
    s_g_all = 0  #预测与真实的交集总数（真正例）

    for i in range(len(y_pred) // 4):  #遍历每个样本（每个样本含4个方面）
        a = set()   #存储当前样本预测为非“无”的方面索引  （预测的标签集合）
        b = set()   #存储当前样本真实为非“无”的方面索引  （真实的标签集合）
        for j in range(4):  #遍历4个方面
            if y_pred[i * 4 + j] != 0:  # 0表示“无”，非0则视为预测到该方面
                a.add(j)
            if y_true[i * 4 + j] != 0:  # 真实标签非0则视为存在该方面
                b.add(j)
        if len(b) == 0: continue  #若真实标签无方面，则跳过该样本
        a_b = a.intersection(b)   #计算预测与真实的交集（真正例）
        if len(a_b) > 0:  #如果存在正确预测的标签
            p = len(a_b) / len(a)   #精确率=真正例/预测正例
            r = len(a_b) / len(b)   #召回率=真正例/真实正例
        else:
            p = 0
            r = 0

        s_g = a.intersection(b)  #存储模型正确预测的方面
        s_all += len(a)   #累计所有预测的方面总数
        g_all += len(b)   #累计所有真实的方面总数
        s_g_all += len(s_g)  #累计所有正确预测的方面总数


        count += 1  #有效样本数+1
        p_all += p  #累积精确率
        r_all += r  #累积召回率
    Ma_p = p_all / count   #平均精确率（宏精确率）
    Ma_r = r_all / count   #平均召回率（宏召回率）
    
    #避免除零错误
    # avoid zero division
    if Ma_p + Ma_r == 0:
        aspect_Macro_F1 = 0
    else:
        aspect_Macro_F1 = 2 * Ma_p * Ma_r / (Ma_p + Ma_r)  #宏F1=2*（宏精确率*宏召回率）/（宏精确率+宏召回率）


    # avoid zero division
    if s_all == 0:  
        p = 0.0
    else:    #如果存在预测的方面数
        p = s_g_all / s_all  #微精确率

    # avoid zero division
    if g_all == 0:
        r = 0.0
    else:   #如果存在真实的方面数
        r = s_g_all / g_all  #微召回率

    # avoid zero division
    if (p + r) == 0:
        f = 0.0
    else:
        f = 2 * p * r / (p + r)   #微F1

    return aspect_Macro_F1


def sentihood_AUC_Acc(y_true, score):
    """
    Calculate "Macro-AUC" of both aspect detection and sentiment classification tasks of Sentihood.
    Calculate "Acc" of sentiment classification task of Sentihood.
    """
    # ----------------------- aspect-Macro-AUC --------------------------
    aspect_y_true = []   #方面检测的真实标签（1表示“无此方面”，0表示“有此方面”）
    aspect_y_score = []  #方面检测的预测分数（“无此方面”的概率）
    aspect_y_trues = [[], [], [], []]   #按4个方面分别存储真实标签
    aspect_y_scores = [[], [], [], []]  #按4个方面分别存储预测分数
    for i in range(len(y_true)):  #遍历真实标签
        if y_true[i] > 0:    #若有此方面（标签>0）
            aspect_y_true.append(0)
        else:    #若无该方面（标签=1）
            aspect_y_true.append(1)  # "None": 1  
        tmp_score = score[i][0]  # probability of "None"   #“无此方面”的概率
        aspect_y_score.append(tmp_score)  #存储“无此方面”的概率
        #按方面索引（0-3）分组存储
        aspect_y_trues[i % 4].append(aspect_y_true[-1])    #将真实标签按方面类别（0-3）分别存入 aspect_y_trues 列表的对应子列表中
        aspect_y_scores[i % 4].append(aspect_y_score[-1])  #将预测概率按方面类别（0-3）分别存入 aspect_y_scores 列表的对应子列表中

    aspect_auc = []
    for i in range(4):  #计算每个方面的AUC
        aspect_auc.append(metrics.roc_auc_score(aspect_y_trues[i], aspect_y_scores[i]))
    aspect_Macro_AUC = np.mean(aspect_auc)    #宏AUC=4个方面AUC的平均值


    # --------------- sentiment-Macro-AUC ---------------------
    sentiment_y_true = []   #情感分类的真实标签（正面=0 负面=1）
    sentiment_y_pred = []   #情感分类的预测标签
    sentiment_y_score = []  #情感分类的预测分数（负面的概率）
    sentiment_y_trues = [[], [], [], []]   #按4个方面分别存储情感真实标签
    sentiment_y_scores = [[], [], [], []]  #按4个方面分别存储情感预测分数
    for i in range(len(y_true)):   #遍历情感真实标签
        if y_true[i] > 0:   #仅处理存在方面的样本（跳过“无”）
            sentiment_y_true.append(y_true[i] - 1)  # "Postive":0, "Negative":1   #转换标签1→0（正面），2→1（负面）
            tmp_score = score[i][2] / (score[i][1] + score[i][2])  # probability of "Negative"  #计算负面情感的概率（负面分数/（正面分数+负面分数））
            sentiment_y_score.append(tmp_score)   #存储负面情感的概率
            if tmp_score > 0.5:   #按概率阈值0.5预测标签
                sentiment_y_pred.append(1)  #预测为负面
            else:
                sentiment_y_pred.append(0)  #预测为正面
            #按方面索引分组存储
            sentiment_y_trues[i % 4].append(sentiment_y_true[-1])
            sentiment_y_scores[i % 4].append(sentiment_y_score[-1])

    sentiment_auc = []  
    for i in range(4):   #计算每个方面的情感AUC
        sentiment_auc.append(metrics.roc_auc_score(sentiment_y_trues[i], sentiment_y_scores[i]))
    sentiment_Macro_AUC = np.mean(sentiment_auc)   #情感宏AUC


    # ------------------------ sentiment Acc ------------------------
    sentiment_y_true = np.array(sentiment_y_true)   #将情感分类的真实标签列表转换为numpy数组
    sentiment_y_pred = np.array(sentiment_y_pred)   #将情感分类的预测标签列表转换为numpy数组
    sentiment_Acc = metrics.accuracy_score(sentiment_y_true, sentiment_y_pred)  #计算情感分类准确率=正确预测的样本数/总样本数

    return aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC  #返回三个评估指标：方面检测的宏AUC、情感分类的准确率、情感分类的宏AUC



from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
#计算SemEval-2014数据集的方面检测任务的“微P、R、F1分数 ”
def semeval_PRF(y_true, y_pred):
    """
    Calculate "Micro P R F" of aspect detection task of SemEval-2014.
    """
    s_all = 0  #所有预测的正例总数
    g_all = 0  #所有真实的正例总数
    s_g_all = 0   #预测与真实的交集总数（真正例）
    al_t, al_p=[],[]  
    for i in range(len(y_pred) // 5):  #遍历每个样本（每个样本含5个方面）
        s = set()  #存储当前样本预测为非“无”的方面索引
        g = set()  #存储当前样本真实为非“无”的方面索引
        for j in range(5):   
            if y_pred[i * 5 + j] != 4:    #4表示“无”，非4则视为预测到该方面
                s.add(j)
            if y_true[i * 5 + j] != 4:    #真实标签非4则视为存在该方面
                g.add(j)
        if len(g) == 0: continue   #若真实标签无方面，则跳过该样本
        s_g = s.intersection(g)   #真正例
        s_all += len(s)   #累计所有预测的方面总数
        g_all += len(g)   #累计所有真实的方面总数
        s_g_all += len(s_g)  #累计所有正确预测的方面总数

    # avoid zero division
    if s_all == 0:
        r = 0.0
    else:
        r = s_g_all / s_all  #微召回率

    # avoid zero division
    if g_all == 0:
        p = 0.0
    else:
        p = s_g_all / g_all  #微精确率

    # avoid zero division
    if (p + r) == 0:
        f = 0.0
    else:
        f = 2 * p * r / (p + r)   #微F1
    return p, r, f


    
#逐个 aspect 输出指标
def semeval_PRF_each_aspect(y_true, y_pred):
    """
    Calculate "Micro P R F" of aspect detection task of SemEval-2014.
    """
    aspects =['service', 'food', 'ambience', 'price', 'anecdotes']  #5个方面名称
    for _, a in enumerate(aspects):  #遍历每个方面
        s_all = 0  #所有预测的正例总数
        g_all = 0  #所有真实的正例总数
        s_g_all = 0   #真正例
        al_t=[]   #存储当前方面的真实标签（存在=1，不存在=0）
        al_p=[]   #存储当前方面的预测标签（存在=1，不存在=0）
        for i in range(len(y_pred) // 5):  #遍历每个样本
            s = set()   #存储当前样本中预测存在的方面索引
            g = set()   #存储当前样本中真实存在的方面索引
            #提取当前样本中当前方面的真实和预测标签
            true= y_true[i+_]  
            pred= y_pred[i+_]
            #转换为二值标签:非4→1（存在），4→0（无）
            true = 1 if true !=4 else 0
            pred = 1 if pred !=4 else 0

            al_t.append(true)
            al_p.append(pred)
            for j in range(5):  #遍历五个方面
                if j!=_:  #如果当前方面索引不等于目标方面索引，则跳过（只处理其他方面）
                    continue
                if y_pred[i * 5 + j] != 4:  #如果当前样本的第j个方面的预测标签不等于4（4表示“无此方面”），则将该方面索引加入预测集合s
                    s.add(j)
                if y_true[i * 5 + j] != 4:  #如果当前样本的第j个方面的真实标签不等于4，则将该方面索引加入真实集合g
                    g.add(j)
            if len(g) == 0: continue  #如果真实集合g为空，（即该样本在当前方面上实际不存在），则跳过后续计算
            s_g = s.intersection(g)  #计算预测集合s和真实集合的交集（即正确预测的方面）
            s_all += len(s)  #累加预测的方面总数
            g_all += len(g)  #累加真实的方面总数
            s_g_all += len(s_g)  #累加正确预测的方面总数
            
        #计算当前方面的微P、R、F1
        # avoid zero division
        if s_all == 0:
            p = 0.0
        else:
            p = s_g_all / s_all

        # avoid zero division
        if g_all == 0:
            r = 0.0
        else:
            r = s_g_all / g_all

        # avoid zero division
        if (p + r) == 0:
            f = 0.0
        else:
            f = 2 * p * r / (p + r)
        f1 = f1_score(al_t, al_p, average='macro')   
        print('{0}, p {1}  r {2} f1 {3}'.format(a, p, r,f,  f1))
   

        
# ------------------ SemEval P、R、F1 ------------------------------
def _semeval_PRF(y_true, y_pred):
    """
    Calculate "Micro P R F" of aspect detection task of SemEval-2014.
    """
    s_all = 0  #所有预测的正例总和
    g_all = 0  #所有真实的正例总和
    s_g_all = 0  #预测与真实的交集总数
    count = 0  #有效样本数
    p_all = 0  #所有样本的精确率总和
    r_all = 0  #所有样本的召回率总和

    for i in range(len(y_pred) // 5):  #遍历每个样本
        s = set()  #创建空集合s，用于存储当前样本中模型预测存在的方面索引
        g = set()  #创建空集合g，用于存储当前样本中真实存在的方面索引
        for j in range(5):  #遍历当前样本的5个方面
            if y_pred[i * 5 + j] != 4:   #如果第j个方面的预测标签不等于4（4表示“无此方面”），则将该方面索引加入预测集合s
                s.add(j)
            if y_true[i * 5 + j] != 4:   #如果第j个方面的真实标签不等于4，则将该方面索引加入真实集合g
                g.add(j)
        if len(g) == 0: continue  #如果真实集合g为空，则跳过当前样本的计算
        s_g = s.intersection(g)   #计算预测集合s和真实集合g的交集（即正确预测的方面）
        s_all += len(s)  #累加所有样本的预测方面总数
        g_all += len(g)  #累加所有样本的真实方面总数
        s_g_all += len(s_g)  #累加所有样本的正确预测方面总数

        a_b = s.intersection(g)  # 再次计算预测与真实的交集（与s_g相同，用于计算单样本指标）
        if len(a_b) > 0:  #若存在正确预测的方面
            p = len(a_b) / len(s)  #计算精确度=正确预测数/预测总数
            r = len(a_b) / len(g)  #计算召回率=正确预测数/真实总数
        else:  # 如果没有正确预测的方面（交集为空）
            p = 0 #精确率和召回率为0
            r = 0

        
        count += 1  #累加有效样本数（有真实方面的样本）
        p_all += p  #累加所有样本的精确率
        r_all += r  #累加所有样本的召回率
    Ma_p = p_all / count  #宏精确率
    Ma_r = r_all / count  #宏召回率
    # avoid zero division
    if Ma_p + Ma_r == 0:
        aspect_Macro_F1 = 0
    else:
        aspect_Macro_F1 = 2 * Ma_p * Ma_r / (Ma_p + Ma_r)  #计算宏F1
        
    #计算微P、R、F1（未返回，仅保留逻辑）
    # avoid zero division
    if s_all == 0:
        p = 0.0
    else:
        p = s_g_all / s_all

    # avoid zero division
    if g_all == 0:
        r = 0.0
    else:
        r = s_g_all / g_all

    # avoid zero division
    if (p + r) == 0:
        f = 0.0
    else:
        f = 2 * p * r / (p + r)
 
    return Ma_p, Ma_r, aspect_Macro_F1


    
# --------------------  SemEval Sentiment Acc（2、3、4类） ---------------------------
def semeval_Acc(y_true, y_pred, score, classes=4):  #参数说明：y_true（真实标签）、y_pred（预测标签）、score（模型预测分数）、classes（分类类别数，4类）
    """
    Calculate "Acc" of sentiment classification task of SemEval-2014.
    """
    assert classes in [2, 3, 4], "classes must be 2 or 3 or 4."  #断言:确保分类类别数只能是2、3、4中的一个，若不是则抛出错误提示

    if classes == 4: #若为4类（正面=0，负面=1，中性=2，冲突=3,无此方面=4）
        total = 0   #初始化有效样本总数（排除“无此方面”的样本）
        total_right = 0  #初始化预测正确的样本数
        for i in range(len(y_true)):  #遍历所有标签（每个标签对应一个方面的情感）
            if y_true[i] == 4: continue   #如果真实标签是4（表示“无此方面”），则跳过该样本（不参与情感分类准确率计算）
            total += 1   #有效样本总数加1
            tmp = y_pred[i]  #获取当前样本的预测标签，暂存到tmp中
            if tmp == 4:  #如果预测标签是4（表示“无此方面”），则根据模型预测分数重新确定情感分类
                # 若“正面”（索引0）的预测分数是4类中最高的，则将预测标签修正为0（正面）
                if score[i][0] >= score[i][1] and score[i][0] >= score[i][2] and score[i][0] >= score[i][3]:
                    tmp = 0
                # 若“负面”（索引1）的预测分数是4类中最高的，则将预测标签修正为1（负面）
                elif score[i][1] >= score[i][0] and score[i][1] >= score[i][2] and score[i][1] >= score[i][3]:
                    tmp = 1
                 # 若“中性”（索引2）的预测分数是4类中最高的，则将预测标签修正为2（中性）
                elif score[i][2] >= score[i][0] and score[i][2] >= score[i][1] and score[i][2] >= score[i][3]:
                    tmp = 2
                # 否则（“冲突”（索引3）的预测分数最高），将预测标签修正为3（冲突）
                else:
                    tmp = 3
            if y_true[i] == tmp:  #若修正后的预测标签与真实标签一致，则预测正确的样本数加1
                total_right += 1
        sentiment_Acc = total_right / total  #计算4类情感分类准确率：正确预测数/有效样本总数
    elif classes == 3:  #若为3类（正面=0，负面=1，中性=2，排除冲突=3和无此方面=4）
        total = 0  #初始化有效样本总数
        total_right = 0  #初始化预测正确的样本数
        for i in range(len(y_true)):   #遍历所有标签
            if y_true[i] >= 3: continue  #如果真实标签≥3（表示“冲突”（3）或“无此方面”（4）），则跳过该样本
            total += 1  #有效样本总数加1
            tmp = y_pred[i]  #获取当前样本的预测标签，暂存到tmp中
            if tmp >= 3:  #如果预测标签≥3，（模型误判为“冲突”或“无此方面”），则根据预测分数重新确定情感类别
                #若“正面”（索引0）的预测分数是3类中最高的，修正为0（正面）
                if score[i][0] >= score[i][1] and score[i][0] >= score[i][2]:
                    tmp = 0
                # 若“负面”（索引1）的预测分数是3类中最高的，修正为1（负面）
                elif score[i][1] >= score[i][0] and score[i][1] >= score[i][2]:
                    tmp = 1
                # 否则（“中性”（索引2）分数最高），修正为2（中性）
                else:
                    tmp = 2
            if y_true[i] == tmp: #若修正后的预测标签与真实标签一致，则预测正确的样本数加1
                total_right += 1
        sentiment_Acc = total_right / total  #计算3类情感分类准确率：正确预测数/有效样本总数
    else:  # #若为2类（正面=0，负面=1，排除中性=2和冲突=3和无此方面=4）
        total = 0  #初始化有效样本总数
        total_right = 0  #初始化预测正确的样本数
        for i in range(len(y_true)):  #遍历所有标签
            if y_true[i] >= 3 or y_true[i] == 1: continue  #如果真实标签≥3（冲突、无此方面）或等于1（负面），则跳过该样本
            total += 1   #有效样本总数加1
            tmp = y_pred[i]  #获取当前样本的预测标签，暂存到tmp中
            if tmp >= 3 or tmp == 1:  #如果预测标签≥3（冲突、无此方面）或等于1（负面），则根据预测分数重新确定类别
                #若“正面”（索引0）分数高于“中性”（索引2），修正为0（正面）
                if score[i][0] >= score[i][2]:
                    tmp = 0
                #否则，修正为2（中性）
                else:
                    tmp = 2
            if y_true[i] == tmp:  #若修正后的预测标签与真实标签一致，则预测正确的样本数加1
                total_right += 1
        sentiment_Acc = total_right / total  #计算2类情感分类准确率：正确预测数/有效样本总数

    return sentiment_Acc  #返回计算得到的情感分类准确率



def main():
    parser = argparse.ArgumentParser()  #创建命令行参数解析器
    #添加“任务名称”参数：必填，指定要评估的任务，可选值包含Sentihood和SemEval数据集的各类子任务
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        choices=["sentihood_single", "sentihood_NLI_M", "sentihood_QA_M", \
                                 "sentihood_NLI_B", "sentihood_QA_B", "semeval_single", \
                                 "semeval_NLI_M", "semeval_QA_M", "semeval_NLI_B", "semeval_QA_B"],
                        help="The name of the task to evaluation.")
    #添加“预测数据目录”参数：必填，指定存储模型预测结果数据的目录路径
    parser.add_argument("--pred_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The pred data dir.")
    args = parser.parse_args()  #解析命令行参数，得到参数对象args
    result = collections.OrderedDict()  #创建有序字典result，用于存储评估结果（保持键的插入顺序）
    
    #判断任务名称是否属于Sentihood数据集的任务（5类子任务）
    if args.task_name in ["sentihood_single", "sentihood_NLI_M", "sentihood_QA_M", "sentihood_NLI_B", "sentihood_QA_B"]:
        y_true = get_y_true(args.task_name)  #调用get_y_true函数获取该任务的真实标签
        y_pred, score = get_y_pred(args.task_name, args.pred_data_dir)  #调用get_y_pred函数获取该任务的预测标签和模型预测分数
        aspect_strict_Acc = sentihood_strict_acc(y_true, y_pred)  #计算Sentihood数据集的方面检测严格准确率
        aspect_Macro_F1 = sentihood_macro_F1(y_true, y_pred)  #计算Sentihood数据集的方面检测宏F1分数
        aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC = sentihood_AUC_Acc(y_true, score)  #计算Sentihood数据集的方面检测宏AUC、情感分类准确率、情感分类宏AUC
        #将Sentihood任务的评估指标存入有序字典
        result = {'aspect_strict_Acc': aspect_strict_Acc,
                  'aspect_Macro_F1': aspect_Macro_F1,
                  'aspect_Macro_AUC': aspect_Macro_AUC,
                  'sentiment_Acc': sentiment_Acc,
                  'sentiment_Macro_AUC': sentiment_Macro_AUC}
    else:  #否则（任务名称属于SemEval数据集的任务）
        y_true = get_y_true(args.task_name)  #调用get_y_true函数获取该任务的真实标签
        y_pred, score = get_y_pred(args.task_name, args.pred_data_dir)   #调用get_y_pred函数获取该任务的预测标签和模型预测分数
        aspect_P, aspect_R, aspect_F = semeval_PRF(y_true, y_pred)  #计算SemEval数据集的方面检测精确率（P）、召回率（R）、F1分数
        #计算SemEval数据集4类、3类、2类情感分类的准确率
        sentiment_Acc_4_classes = semeval_Acc(y_true, y_pred, score, 4)  
        sentiment_Acc_3_classes = semeval_Acc(y_true, y_pred, score, 3)  
        sentiment_Acc_2_classes = semeval_Acc(y_true, y_pred, score, 2)
        #将SemEval任务的评估指标存入有序字典
        result = {'aspect_P': aspect_P,
                  'aspect_R': aspect_R,
                  'aspect_F': aspect_F,
                  'sentiment_Acc_4_classes': sentiment_Acc_4_classes,
                  'sentiment_Acc_3_classes': sentiment_Acc_3_classes,
                  'sentiment_Acc_2_classes': sentiment_Acc_2_classes}

    for key in result.keys():  #遍历有序字典的所有键，打印每个评估指标的名称和对应值
        print(key, "=", str(result[key]))


if __name__ == "__main__":  
    main()
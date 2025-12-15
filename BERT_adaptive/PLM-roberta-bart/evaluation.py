# =============================================================================
# 评估指标计算模块
# =============================================================================
# 功能：文件内部实现了适用于 SemEval 和 Sentihood 数据集的方面检测与情感分类指标；
# 包含的指标：
#            数据集	          任务	                指标
#            Sentihood	   方面检测 + 情感分类	    Strict Acc, Macro F1, Macro AUC，Acc
#            SemEval	   方面检测 + 情感分类	    P/R/F1, Acc (4/3/2类)
# =============================================================================


import argparse
import collections
import numpy as np
import pandas as pd
from sklearn import metrics     #用于计算评估指标的工具库
from sklearn.preprocessing import label_binarize    #用于将标签二值化



# ---------------------------------------------------------------------------
# ✅ Sentihood Strict Accuracy
# ---------------------------------------------------------------------------
def sentihood_strict_acc(y_true, y_pred):
    """
    计算sentihood----方面检测任务----严格准确率
    """
    total_cases = int(len(y_true) / 4)    #总样本数=标签长度/4（每个样本包含4个方面）
    true_cases = 0     #完全正确的样本数
    for i in range(total_cases):
        if y_true[i * 4] != y_pred[i * 4]: continue
        if y_true[i * 4 + 1] != y_pred[i * 4 + 1]: continue
        if y_true[i * 4 + 2] != y_pred[i * 4 + 2]: continue
        if y_true[i * 4 + 3] != y_pred[i * 4 + 3]: continue
        true_cases += 1
    aspect_strict_Acc = true_cases / total_cases     #计算严格准确率=完全正确的样本数/总样本数
    return aspect_strict_Acc


# ---------------------------------------------------------------------------
# ✅ Sentihood  Micro-F1 和 Micro-F1
# ---------------------------------------------------------------------------
def sentihood_macro_F1(y_true, y_pred):
    """
    计算sentihood-----方面检测任务---- Macro-F1 和 Micro-F1
    """
    p_all = 0   #所有样本的精确率总和
    r_all = 0   #所有样本的召回率总和
    count = 0   #有效样本数（真是标签非空的样本）
    
    s_all = 0    #所有预测的正例总数   
    g_all = 0    #所有真实的正例总数
    s_g_all = 0   #真正例

    for i in range(len(y_pred) // 4):
        a = set()    #预测的标签集合
        b = set()    #真实的标签集合
        for j in range(4):
            if y_pred[i * 4 + j] != 0:  #若预测到该方面
                a.add(j)
            if y_true[i * 4 + j] != 0:  #若存在该方面
                b.add(j)
        if len(b) == 0: continue

        a_b = a.intersection(b)    #计算真正例
        if len(a_b) > 0:
            p = len(a_b) / len(a)   #精确率=真正例/预测正例
            r = len(a_b) / len(b)   #召回率=真正例/真实正例
        else:
            p = 0
            r = 0

        s_g = a.intersection(b)
        s_all += len(a)
        g_all += len(b)
        s_g_all += len(s_g)

        count += 1
        p_all += p
        r_all += r
    Ma_p = p_all / count   #宏精确率
    Ma_r = r_all / count   #宏召回率

    # avoid zero division
    if Ma_p + Ma_r == 0:  
        aspect_Macro_F1 = 0
    else:
        aspect_Macro_F1 = 2 * Ma_p * Ma_r / (Ma_p + Ma_r)    #宏F1=2*（宏精确率*宏召回率）/（宏精确率+宏召回率）


    if s_all == 0:
        p = 0.0
    else:
        p = s_g_all / s_all  #微精确率

    if g_all == 0:
        r = 0.0
    else:
        r = s_g_all / g_all   #微召回率

    if (p + r) == 0:
        f = 0.0
    else:
        f = 2 * p * r / (p + r)   #微F1
        
    return aspect_Macro_F1



# ---------------------------------------------------------------------------
# ✅ Sentihood AUC + Accuracy
# ---------------------------------------------------------------------------
def sentihood_AUC_Acc(y_true, score):
    """
    计算sentihod----方面检测和情感分类任务------宏AUC
    计算sentihod----情感分类任务----Accuracy
    """
    # ------------------ aspect-Macro-AUC ------------------
    aspect_y_true = []
    aspect_y_score = []
    aspect_y_trues = [[], [], [], []]
    aspect_y_scores = [[], [], [], []]
    for i in range(len(y_true)):
        if y_true[i] > 0:
            aspect_y_true.append(0)   #若有该方面（标签=0）
        else:
            aspect_y_true.append(1)   #若无该方面（标签=1）
        tmp_score = score[i][0]  
        aspect_y_score.append(tmp_score)
        aspect_y_trues[i % 4].append(aspect_y_true[-1])
        aspect_y_scores[i % 4].append(aspect_y_score[-1])

    aspect_auc = []
    for i in range(4):
        aspect_auc.append(metrics.roc_auc_score(aspect_y_trues[i], aspect_y_scores[i]))   #计算每个方面的AUC
    aspect_Macro_AUC = np.mean(aspect_auc)    #宏AUC=4个方面AUC的平均值

    # ------------------------ sentiment-Macro-AUC -------------------------
    sentiment_y_true = []
    sentiment_y_pred = []
    sentiment_y_score = []
    sentiment_y_trues = [[], [], [], []]
    sentiment_y_scores = [[], [], [], []]
    for i in range(len(y_true)):
        if y_true[i] > 0:
            sentiment_y_true.append(y_true[i] - 1)   # "Postive":0, "Negative":1 
            tmp_score = score[i][2] / (score[i][1] + score[i][2])   # probability of "Negative"
            sentiment_y_score.append(tmp_score)

            if tmp_score > 0.5:
                sentiment_y_pred.append(1)     # "Negative": 1
            else:
                sentiment_y_pred.append(0)
            sentiment_y_trues[i % 4].append(sentiment_y_true[-1])
            sentiment_y_scores[i % 4].append(sentiment_y_score[-1])

    sentiment_auc = []
    for i in range(4):
        sentiment_auc.append(metrics.roc_auc_score(sentiment_y_trues[i], sentiment_y_scores[i]))  #计算每个方面的情感AUC
    sentiment_Macro_AUC = np.mean(sentiment_auc)    #情感宏AUC

    # -------------------- sentiment Acc ----------------------
    sentiment_y_true = np.array(sentiment_y_true)
    sentiment_y_pred = np.array(sentiment_y_pred)
    sentiment_Acc = metrics.accuracy_score(sentiment_y_true, sentiment_y_pred)   #计算情感分类准确率=正确预测的样本数/总样本数
    
    return aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC


from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
#微平均---统计样本层面的精确率、召回率、F1
def semeval_PRF(y_true, y_pred):
    """
    计算SemEval----方面检测任务----微P、R、F1
    """
    s_all = 0
    g_all = 0
    s_g_all = 0
    al_t, al_p=[],[]
    for i in range(len(y_pred) // 5):
        s = set()
        g = set()
        for j in range(5):
            if y_pred[i * 5 + j] != 4:
                s.add(j)
            if y_true[i * 5 + j] != 4:
                g.add(j)
        if len(g) == 0: continue
            
        s_g = s.intersection(g)
        s_all += len(s)
        g_all += len(g)
        s_g_all += len(s_g)

    # avoid zero division
    if s_all == 0:
        r = 0.0
    else:
        r = s_g_all / s_all   #微召回率

  
    if g_all == 0:
        p = 0.0
    else:
        p = s_g_all / g_all   #微精确率

   
    if (p + r) == 0:
        f = 0.0
    else:
        f = 2 * p * r / (p + r)   #微F1
    return p, r, f



#逐个 aspect 输出指标
def semeval_PRF_each_aspect(y_true, y_pred):
    """
    计算SemEval----方面检测任务(每个方面)----微P、R、F1
    """
    aspects =['service', 'food', 'ambience', 'price', 'anecdotes']
    for _, a in enumerate(aspects):
        s_all = 0
        g_all = 0
        s_g_all = 0
        al_t=[]
        al_p=[]
        for i in range(len(y_pred) // 5):
            s = set()
            g = set()
            true= y_true[i+_]
            pred= y_pred[i+_]
            true = 1 if true !=4 else 0
            pred = 1 if pred !=4 else 0

            al_t.append(true)
            al_p.append(pred)
            for j in range(5):
                if j!=_:
                    continue
                if y_pred[i * 5 + j] != 4:
                    s.add(j)
                if y_true[i * 5 + j] != 4:
                    g.add(j)
            if len(g) == 0: continue
            s_g = s.intersection(g)
            s_all += len(s)
            g_all += len(g)
            s_g_all += len(s_g)

        # avoid zero division
        if s_all == 0:
            p = 0.0
        else:
            p = s_g_all / s_all

        if g_all == 0:
            r = 0.0
        else:
            r = s_g_all / g_all

        if (p + r) == 0:
            f = 0.0
        else:
            f = 2 * p * r / (p + r)   #微F1
        f1 = f1_score(al_t, al_p, average='macro')    #宏F1
        print('{0}, p {1}  r {2} f1 {3}'.format(a, p, r, f, f1))
  

        
#SemEval-14 任务要求的官方指标
def _semeval_PRF(y_true, y_pred):
    """
    计算SemEval----方面检测任务-----宏P、R、F1
    """
    s_all = 0
    g_all = 0
    s_g_all = 0
    
    count = 0
    p_all = 0
    r_all = 0
    for i in range(len(y_pred) // 5):
        s = set()
        g = set()
        for j in range(5):
            if y_pred[i * 5 + j] != 4:
                s.add(j)
            if y_true[i * 5 + j] != 4:
                g.add(j)
        if len(g) == 0: continue
        s_g = s.intersection(g)
        s_all += len(s)
        g_all += len(g)
        s_g_all += len(s_g)

        a_b = s.intersection(g)
        if len(a_b) > 0:
            p = len(a_b) / len(s)    #精确度=正确预测数/预测总数
            r = len(a_b) / len(g)    #召回率=正确预测数/真实总数
        else:
            p = 0
            r = 0

        count += 1
        p_all += p
        r_all += r
    Ma_p = p_all / count    #宏精确率
    Ma_r = r_all / count    #宏召回率

    # avoid zero division
    if Ma_p + Ma_r == 0:
        aspect_Macro_F1 = 0
    else:
        aspect_Macro_F1 = 2 * Ma_p * Ma_r / (Ma_p + Ma_r)   #宏F1
        
    if s_all == 0:
        p = 0.0
    else:
        p = s_g_all / s_all

    if g_all == 0:
        r = 0.0
    else:
        r = s_g_all / g_all

    if (p + r) == 0:
        f = 0.0
    else:
        f = 2 * p * r / (p + r)
        
    return Ma_p, Ma_r, aspect_Macro_F1


# ---------------------------------------------------------------------------
# ✅ SemEval Sentiment Acc 
# ---------------------------------------------------------------------------
def semeval_Acc(y_true, y_pred, score, classes=4):
    """
    计算SemEval----情感分类任务----准确率Acc
    """
    assert classes in [2, 3, 4], "classes must be 2 or 3 or 4."
    if classes == 4:
        total = 0
        total_right = 0
        for i in range(len(y_true)):
            if y_true[i] == 4: continue
            total += 1
            tmp = y_pred[i]
            if tmp == 4:
                if score[i][0] >= score[i][1] and score[i][0] >= score[i][2] and score[i][0] >= score[i][3]:
                    tmp = 0
                elif score[i][1] >= score[i][0] and score[i][1] >= score[i][2] and score[i][1] >= score[i][3]:
                    tmp = 1
                elif score[i][2] >= score[i][0] and score[i][2] >= score[i][1] and score[i][2] >= score[i][3]:
                    tmp = 2
                else:
                    tmp = 3
            if y_true[i] == tmp:
                total_right += 1
        sentiment_Acc = total_right / total   #计算4类情感分类准确率：正确预测数/有效样本总数
        
    elif classes == 3:
        total = 0
        total_right = 0
        for i in range(len(y_true)):
            if y_true[i] >= 3: continue
            total += 1
            tmp = y_pred[i]
            if tmp >= 3:
                if score[i][0] >= score[i][1] and score[i][0] >= score[i][2]:
                    tmp = 0
                elif score[i][1] >= score[i][0] and score[i][1] >= score[i][2]:
                    tmp = 1
                else:
                    tmp = 2
            if y_true[i] == tmp:
                total_right += 1
        sentiment_Acc = total_right / total   #计算3类情感分类准确率：正确预测数/有效样本总数
        
    else:
        total = 0
        total_right = 0
        for i in range(len(y_true)):
            if y_true[i] >= 3 or y_true[i] == 1: continue
            total += 1
            tmp = y_pred[i]
            if tmp >= 3 or tmp == 1:
                if score[i][0] >= score[i][2]:
                    tmp = 0
                else:
                    tmp = 2
            if y_true[i] == tmp:
                total_right += 1
        sentiment_Acc = total_right / total   #计算2类情感分类准确率：正确预测数/有效样本总数

    return sentiment_Acc  #返回计算得到的情感分类准确率



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        choices=["sentihood_single", "sentihood_NLI_M", "sentihood_QA_M", \
                                 "sentihood_NLI_B", "sentihood_QA_B", "semeval_single", \
                                 "semeval_NLI_M", "semeval_QA_M", "semeval_NLI_B", "semeval_QA_B"],
                        help="The name of the task to evalution.")
    parser.add_argument("--pred_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The pred data dir.")
    args = parser.parse_args()
    result = collections.OrderedDict()
    
    #判断任务名称是否属于Sentihood数据集的任务（5类子任务）
    if args.task_name in ["sentihood_single", "sentihood_NLI_M", "sentihood_QA_M", "sentihood_NLI_B", "sentihood_QA_B"]:
        y_true = get_y_true(args.task_name)
        y_pred, score = get_y_pred(args.task_name, args.pred_data_dir)
        aspect_strict_Acc = sentihood_strict_acc(y_true, y_pred)
        aspect_Macro_F1 = sentihood_macro_F1(y_true, y_pred)
        aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC = sentihood_AUC_Acc(y_true, score)
        #将Sentihood任务的评估指标存入有序字典
        result = {'aspect_strict_Acc': aspect_strict_Acc,
                  'aspect_Macro_F1': aspect_Macro_F1,
                  'aspect_Macro_AUC': aspect_Macro_AUC,
                  'sentiment_Acc': sentiment_Acc,
                  'sentiment_Macro_AUC': sentiment_Macro_AUC}
    else:  #判断任务名称属于SemEval数据集的任务
        y_true = get_y_true(args.task_name)
        y_pred, score = get_y_pred(args.task_name, args.pred_data_dir)
        aspect_P, aspect_R, aspect_F = semeval_PRF(y_true, y_pred)
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

    for key in result.keys():
        print(key, "=", str(result[key]))  #遍历有序字典的所有键，打印每个评估指标的名称和对应值
  

if __name__ == "__main__":
    main()

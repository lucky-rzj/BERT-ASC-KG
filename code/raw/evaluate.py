# ===========================================================================================
# 模型评估入口----加载模型 + 数据 + 调用指标
# ===========================================================================================
# 功能：用于加载训练好的模型并在测试集上运行预测，然后调用 evaluation.py 中的指标函数输出最终性能
# ===========================================================================================

import json  #用于JSON数据处理
import logging  #用于日志记录
import argparse  #用于解析命令行参数
import os  #用于文件路径操作
import sys  #用于系统相关操作
import random  #用于随机数生成
import numpy   #导入numpy库，用于数值计算
import pandas as pd   #导入pandas库，并简写为pd，用于数据处理
from torch.utils.data.sampler import  WeightedRandomSampler   #用于加权随机采样器
import torch  #导入pytorch库，用于深度学习相关操作
from torch.utils.data import DataLoader, random_split, TensorDataset  #用于数据加载和处理
import torch.nn.functional as F   #pytorch的函数式接口
import  numpy as np 
import copy  #用于对象复制
from  tqdm import tqdm   #用于显示进度条
from transformers import AdamW   #从transformers库导入AdamW优化器
from transformers import AutoTokenizer   #用于加载预训练模型的分词器

from evaluation import *  #导入评估指标函数
from data_utils import  ABSADataset_absa_bert_semeval_json, ABSADataset_absa_bert_sentihood_json   
from MyModel import BERT_ASC_vanila   


logger = logging.getLogger()   #创建日志记录器
logger.setLevel(logging.INFO)  #设置日志级别为INFO
logger.addHandler(logging.StreamHandler(sys.stdout))   #将日志输出到控制台



class Instructor:
    def __init__(self, opt): 
        self.opt = opt 
        #初始化模型并移动到指定设备
        self.model = BERT_ASC_vanila(opt)  
        self.model.to(self.opt.device)
        
        #加载预训练分词器
        tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_bert_name)   

        #根据数据集的类型，加载测试集    
        if self.opt.dataset=='semeval':  
            self.testset = ABSADataset_absa_bert_semeval_json(opt.dataset_file['test'], tokenizer, opt)
        else:
            self.testset = ABSADataset_absa_bert_sentihood_json(opt.dataset_file['test'], tokenizer, opt)
        logger.info(' test {}'.format( len(self.testset)))    #打印测试集的大小

        #若使用GPU，打印显存使用情况
        if opt.device.type == 'cuda':   
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        if torch.cuda.device_count() > 1:   #如果有多个GPU，使用数据并行
            logger.info(f'Using {torch.cuda.device_count()} GPUs.')
            self.model = nn.DataParallel(self.model)
        self._print_args()   #打印参数信息

        
    #计算可训练参数和不可训练参数
    def _print_args(self):   
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():   #遍历模型参数
            n_params = torch.prod(torch.tensor(p.shape))   #计算参数总数
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))


    #评估准确率和F1分数的方法
    def _evaluate_acc_f1(self, data_loader):  
        n_correct, n_total = 0, 0   #正确预测数和总样本数
        t_targets_all, t_outputs_all = None, None   #存储所有真实标签和输出
        score = []  #存储预测分数
        self.model.eval()   #模型设为评估模型
        with torch.no_grad():   #关闭梯度计算
            for t_batch, t_sample_batched in enumerate(tqdm(data_loader)):  #遍历数据加载器
                #将数据移动到指定设备
                t_sample_batched = [b.to(self.opt.device) for b in t_sample_batched]
                input_ids, token_type_ids, attention_mask, labels = t_sample_batched

                logits = self.model(input_ids, token_type_ids, attention_mask, labels=None)   #模型向前传播，获取输出
                score.append(F.softmax(logits, dim=-1).detach().cpu().numpy())   #存储softmax后的概率分数
                #计算正确预测数
                n_correct += (torch.argmax(logits, -1) == labels).sum().item()
                n_total += len(logits)

                if t_targets_all is None:   #累积所有标签和输出
                    t_targets_all = labels
                    t_outputs_all = logits
                else:
                    t_targets_all = torch.cat((t_targets_all, labels), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, logits), dim=0)
        #返回真实标签、预测标签和分数
        return t_targets_all.cpu().numpy(), torch.argmax(t_outputs_all, -1).cpu().numpy(), np.concatenate(score, axis=0)


        
    # ---------------------- 测试集上评估 -------------------------
    def run(self):  
        #转换测试集为TensorDataset格式
        testset = TensorDataset(torch.tensor([f['text_bert_indices'] for f in self.testset], dtype=torch.long),
                                      torch.tensor([f['bert_segments_ids'] for f in self.testset], dtype=torch.long),
                                      torch.tensor([f['input_mask'] for f in self.testset], dtype=torch.long),
                                      torch.tensor([f['label'] for f in self.testset], dtype=torch.long)
        )
        #创建测试集数据加载器
        test_data_loader = DataLoader(dataset=testset, batch_size=self.opt.eval_batch_size, shuffle=False)  

        
        # --------------------------------------------------
        # 🔧 统一 SEEDS（应与你训练端使用的完全一致）
        # --------------------------------------------------
        if self.opt.dataset == 'semeval':
            SEEDS = [42, 21, 7, 13, 87]
        else:
            SEEDS = [42, 101, 735, 2025, 12345]
            
        all_results = []    #保存每个 seed 的结果


        # --------------------------------------------------
        # ⭐ 依次加载每个 seed 的模型 + 做测试
        # --------------------------------------------------
        for seed in SEEDS:
            best_model_path = f"state_dict/{self.opt.dataset}/seed{seed}.bm"
            logger.info(f"\n============ Evaluating seed {seed} ============\n")
            #加载模型参数
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.opt.device))

            #------------------ 测试集上评估并打印指标 --------------------------------
            self.model.eval()  
            y_true, y_pred, score = self._evaluate_acc_f1(test_data_loader)  #评估模型，获取真实标签、预测标签和分数
            if self.opt.dataset=='semeval':
                aspect_P, aspect_R, aspect_F = semeval_PRF(y_true, y_pred)
                sentiment_Acc_4_classes = semeval_Acc(y_true, y_pred, score, 4)
                sentiment_Acc_3_classes = semeval_Acc(y_true, y_pred, score, 3)
                sentiment_Acc_2_classes = semeval_Acc(y_true, y_pred, score, 2)

                logger.info("*************************************************************************")
                logger.info('>> P: {:.4f} , R: {:.4f} , F: {:.4f} '.format(aspect_P, aspect_R, aspect_F))
                logger.info('>> 4 classes acc: {:.4f} '.format(sentiment_Acc_4_classes))
                logger.info('>> 3 classes acc: {:.4f} '.format(sentiment_Acc_3_classes))
                logger.info('>> 2 classes acc: {:.4f} '.format(sentiment_Acc_2_classes))
                logger.info("*************************************************************************")

                #记录
                all_results.append({
                    "P": aspect_P,
                    "R": aspect_R,
                    "F": aspect_F,
                    "Acc4": sentiment_Acc_4_classes,
                    "Acc3": sentiment_Acc_3_classes,
                    "Acc2": sentiment_Acc_2_classes,
                })
            else:
                aspect_strict_Acc = sentihood_strict_acc(y_true, y_pred)
                aspect_Macro_F1 = sentihood_macro_F1(y_true, y_pred)
                aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC = sentihood_AUC_Acc(y_true, score)

                logger.info("*************************************************************************")
                logger.info(())
                logger.info('>> aspect_strict_Acc: {:.4f} , aspect_Macro_F1: {:.4f} , aspect_Macro_AUC: {:.4f} '.format(
                    aspect_strict_Acc, 
                    aspect_Macro_F1, 
                    aspect_Macro_AUC
                ))
                logger.info('>> sentiment_Acc: {:.4f} '.format(sentiment_Acc))
                logger.info('>> sentiment_Macro_AUC: {:.4f} '.format(sentiment_Macro_AUC))
                logger.info("*************************************************************************")

                all_results.append({
                    "aspect_strict_Acc": aspect_strict_Acc,
                    "aspect_Macro_F1": aspect_Macro_F1,
                    "aspect_Macro_AUC": aspect_Macro_AUC,
                    "sentiment_Acc": sentiment_Acc,
                    "sentiment_Macro_AUC": sentiment_Macro_AUC,
                })

                
        # --------------------------------------------------
        # ⭐⭐ 输出 5 次的平均结果（复现论文）
        # --------------------------------------------------
        logger.info("\n==================== 5-SEED AVERAGE ====================")
        avg = {k: sum(r[k] for r in all_results) / len(all_results) for k in all_results[0]}
        if self.opt.dataset == 'semeval':
            logger.info(f"Avg P     = {avg['P']:.4f}")
            logger.info(f"Avg R     = {avg['R']:.4f}")
            logger.info(f"Avg F     = {avg['F']:.4f}")
            logger.info(f"Avg Acc-4 = {avg['Acc4']:.4f}")
            logger.info(f"Avg Acc-3 = {avg['Acc3']:.4f}")
            logger.info(f"Avg Acc-2 = {avg['Acc2']:.4f}")
        else:
            logger.info(f"Avg strict_acc = {avg['aspect_strict_Acc']:.4f}")
            logger.info(f"Avg macro_F1   = {avg['aspect_Macro_F1']:.4f}")
            logger.info(f"Avg macro_AUC  = {avg['aspect_Macro_AUC']:.4f}")
            logger.info(f"Avg sent_Acc   = {avg['sentiment_Acc']:.4f}")
            logger.info(f"Avg sent_AUC   = {avg['sentiment_Macro_AUC']:.4f}")
        logger.info("=======================================================================")




def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()  #解析命令行参数
    parser.add_argument('--dataset', default='semeval', type=str,  choices=['semeval','sentihood'], help='semeval, sentihood', required=True)  
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)    #
    parser.add_argument('--learning_rate', default=3e-5, type=float, help='try 5e-5, 2e-5')  #学习率
    parser.add_argument('--dropout', default=0.1, type=float)  
    parser.add_argument('--l2reg', default=0.001, type=float)  #L2正则化
    parser.add_argument('--warmup_proportion', default=0.01, type=float)  #学习率预热比例
    parser.add_argument('--num_epoch', default=5, type=int, help='')  #训练轮数
    parser.add_argument("--train_batch_size", default=32,type=int, help="Total batch size for training.")  #训练的总批次大小
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for eval.")  #评估的总批次大小
    parser.add_argument('--log_step', default=50, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=120, type=int)  #文本的最大序列长度
    parser.add_argument('--label_dim', default=5, type=int)  #标签维度
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--save_model', default=0, type=int)   #保存最佳模型的设置
    parser.add_argument('--device', default='cuda', type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=42, type=int, help='set seed for reproducibility')  #随机种子（设置随机种子以保证结果可复现）
    parser.add_argument('--valset_ratio', default=0, type=float,
                        help='set ratio between 0 and 1 for validation support')    #验证比例（设置0到1之间的比例作为验证集支持）
    opt = parser.parse_args()

    
    if opt.dataset=='sentihood':  #根据数据集设置标签维度（Sentihood为3类，SemEval为5类）
        opt.label_dim =3
        
    #设置随机种子以保证可复现性
    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    #定义数据集的文件路径
    dataset_files = {
        'train': '../../datasets/{}/bert_train.json'.format(opt.dataset),
        'test': '../../datasets/{}/bert_test.json'.format(opt.dataset),
        'val': '../../datasets/{}/bert_dev.json'.format(opt.dataset)
    }
    #定义参数初始化的方法
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }


    #设置预训练模型路径和其他参数
    logger.info(opt.pretrained_bert_name)  #打印预训练BERT模型的名称
    opt.dataset_file = dataset_files
    opt.inputs_cols = ['text_bert_indices', 'bert_segments_ids', 'input_mask', 'label']
    opt.initializer = initializers[opt.initializer]
    opt.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu') 
    
    ins = Instructor(opt)   # # 创建Instructor实例并执行评估
    ins.run()


if __name__ == '__main__':
    main()

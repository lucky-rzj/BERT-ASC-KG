# =============================================================================
# 训练主程序---标准版
# ======================================================================================
# 功能:面向 BERT-CLS 分类流程的训练脚本，用 HuggingFace 标准序列分类头做 ABSA 训练/验证/测试
# ======================================================================================

import logging   
import argparse  
import os 
import sys  
import random   #导入随机数模块，用于随机操作
import numpy  
from transformers import AdamW    #从transformers库导入AdamW优化器
from torch.utils.data.sampler import  WeightedRandomSampler   #导入加权随机采样器，用于不均衡数据采样
import torch   
from torch.utils.data import DataLoader, random_split, TensorDataset   #导入数据加载相关类
import torch.nn.functional as F   #导入pytorch的函数式接口，用于激活函数等操作
import  numpy as np 
import copy   
from  tqdm import tqdm   
from transformers import AutoTokenizer, AutoModel   #从transformers库导入自动分词器和自动模型
import torch.nn as nn   #导入pytorch的神经网络模块

from evaluation import *   #从evaluation模块导入所有函数，用于模型评估
from data_utils import  ABSADataset_absa_bert_semeval_json, ABSADataset_absa_bert_sentihood_json   #从data_utils模块导入数据集类
from MyModel import  BERT_ASC_vanila   #从MyModel模块导入BERT_ASC_vanila模型类


# ------------------- logger --------------------------- 
logger = logging.getLogger()   #配置日志记录器
logger.setLevel(logging.INFO)  #设置日志级别为INFO
logger.addHandler(logging.StreamHandler(sys.stdout))    #向标准输出添加日志处理器


class Instructor:   
    def __init__(self, opt): 
        self.opt = opt   #保存配置选项
        self.model = BERT_ASC_vanila(opt)   #初始化BERT_ASC_vanila模型
        self.model.to(self.opt.device)  #将模型移动到指定的设备（CPU或GPU）
        tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_bert_name)   #加载预训练的分词器
        
        if self.opt.dataset=='semeval':   
            #加载semeval数据集的训练集和测试集
            self.trainset = ABSADataset_absa_bert_semeval_json(opt.dataset_file['train'], tokenizer, opt)  
            self.testset = ABSADataset_absa_bert_semeval_json(opt.dataset_file['test'], tokenizer, opt)
            assert 0 <= opt.valset_ratio < 1   #确保验证集比例在合理范围内
            if opt.valset_ratio > 0:  #如果需要划分验证集
                valset_len = int(len(self.trainset) * opt.valset_ratio)   #计算验证集长度
                self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))  #划分为训练集和验证集
            else:    #如果不需要划分验证集，使用测试集作为验证集
                self.valset = self.testset 
        else:  
            #加载sentihood数据集的训练集和测试集
            self.trainset = ABSADataset_absa_bert_sentihood_json(opt.dataset_file['train'], tokenizer, opt)  
            self.testset = ABSADataset_absa_bert_sentihood_json(opt.dataset_file['test'], tokenizer, opt)
            self.valset =self.testset   #使用测试集作为训练集
        logger.info('train {0}: dev {1}: test {2}'.format(len(self.trainset), len(self.valset), len(self.testset)))  
        
        if opt.device.type == 'cuda':    #如果是使用GPU
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))   #记录GPU的内存分配情况

            
    #定义学习率预热函数
    def warmup_linear(self, x, warmup=0.002):  
        if x < warmup:  #如果在预热阶段
            return x / warmup  #线性增加学习率
        return 1.0 - x   #预热后线性衰减学习率


        
    #-------------------------------------- 训练方法  ------------------------------------
    def _train(self, optimizer,criterion, train_data_loader, val_data_loader, t_total):   
        max_val_f1 = 0   #记录验证集最佳性能指标（根据数据集类型动态确定）
        global_step = 0   #全局步数计数器
        best_model_state = None     #存储最佳模型的参数字典
        best_epoch = -1     #记录最优轮次
        best_model_path = None    #保存路径  
        
        #循环训练轮次
        for epoch in range(self.opt.num_epoch):  
            loss_total = 0
            setp_total = 0
            logger.info('>' * 100)     #打印分割线
            logger.info('epoch: {}'.format(epoch))   #打印当前轮次
            self.model.train()     #将模式设置为训练模式
            # ------------------- 遍历训练数据 ---------------
            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):   
                optimizer.zero_grad()   #梯度清零
                sample_batched= [b.to(self.opt.device) for b in sample_batched]    #将批次数据移动到指定设备
                input_ids, token_type_ids, attention_mask, labels= sample_batched    #解包批次数据
                outputs= self.model(input_ids, token_type_ids, attention_mask, labels)   #模型向前传播，得到输出
                loss = criterion(outputs, labels)   #计算损失
                loss.sum().backward()    #反向传播计算梯度
                
                #累加损失和样本数（无梯度计算）
                with torch.no_grad():
                    loss_total+= loss.item()
                    setp_total+=len(labels)
                    
                #计算当前步骤的学习率
                lr_this_step = self.opt.learning_rate * self.warmup_linear(global_step / t_total, self.opt.warmup_proportion) 
                #更新优化器的学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()  #优化器更新参数
                global_step += 1  #全局步数加1


                
            # -------------------------- 验证集上评估 ---------------------
            y_true, y_pred, score = self._evaluate_acc_f1(val_data_loader)   #在验证集上评估模型，得到真实标签、预测标签和分数
            if self.opt.dataset  == 'semeval':  
                #计算semeval数据集的P、R、F分数和不同类别的2类、3类、4类准确率
                aspect_P, aspect_R, aspect_F = semeval_PRF(y_true, y_pred)  
                sentiment_Acc_4_classes = semeval_Acc(y_true, y_pred, score, 4)
                sentiment_Acc_3_classes = semeval_Acc(y_true, y_pred, score, 3)
                sentiment_Acc_2_classes = semeval_Acc(y_true, y_pred, score, 2)
                max_per= sentiment_Acc_4_classes    #以4类情感准确率作为最佳指标
            else: 
                #计算sentihood数据集的strict_Acc（严格准确率）、Macro_F1、Macro_AUC
                aspect_strict_Acc = sentihood_strict_acc(y_true, y_pred) 
                aspect_Macro_F1 = sentihood_macro_F1(y_true, y_pred)
                aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC = sentihood_AUC_Acc(y_true, score)
                max_per=aspect_strict_Acc     #以严格的aspect准确率作为最佳指标
            logger.info(" epoch : {0}, training loss: {1} ".format(str(epoch), loss_total/setp_total  ))  #打印当前轮次的训练损失
                        
            # -------------------------- 打印验证集指标 ---------------------
            if self.opt.dataset == 'semeval':  
                logger.info('')
                logger.info('>> P: {:.4f} , R: {:.4f} , F: {:.4f} '.format(aspect_P, aspect_R, aspect_F))
                logger.info('>> 2 classes acc: {:.4f} '.format(sentiment_Acc_2_classes))
                logger.info('>> 3 classes acc: {:.4f} '.format(sentiment_Acc_3_classes))
                logger.info('>> 4 classes acc: {:.4f} '.format(sentiment_Acc_4_classes))
            else:  
                logger.info('')
                logger.info('>> aspect_strict_Acc: {:.4f} , aspect_Macro_F1: {:.4f} , aspect_Macro_AUC: {:.4f} '.format(
                    aspect_strict_Acc, 
                    aspect_Macro_F1, 
                    aspect_Macro_AUC
                ))
                logger.info('>> sentiment_Acc: {:.4f} '.format(sentiment_Acc))
                logger.info('>> sentiment_Macro_AUC: {:.4f} '.format(sentiment_Macro_AUC))

                
            # --------------------- 保存最优模型 ------------------------------
            if max_per > max_val_f1:  #如果当前评估指标优于历史最佳
                max_val_f1 = max_per   #更新最佳指标
                best_epoch = epoch    #记录最优轮次
                #创建目录(根目录、子目录)
                if not os.path.exists('state_dict'):  #如果保存目录不存在则创建（根目录）
                    os.mkdir('state_dict')
                save_dir = f'state_dict/{self.opt.dataset}'    #创建数据集对应的子目录
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                #定义保存路径
                best_model_path = f'{save_dir}/seed{self.opt.seed}.bm'
                #保存模型到文件
                torch.save(self.model.state_dict(), best_model_path)
                #记录最佳模型状态
                best_model_state = copy.deepcopy(self.model.state_dict())
               
            self.model.train()   #将模型重新设置为训练模式
        logger.info(f"🔥 Training Finished. Best Epoch = {best_epoch}")
        return best_model_state,best_epoch   #返回模型状态字典、最优轮次

        
    
    def _evaluate_acc_f1(self, data_loader):   
        """ 核心评估函数：获取评估数据集的真实标签、预测标签、预测分数（用于后续计算各类指标） """
        n_correct, n_total = 0, 0  #正确预测数和总样本数
        t_targets_all, t_outputs_all = None, None   #存储所有目标标签和输出
        score = []  #存储预测分数
        self.model.eval()  #将模型设置为评估模式
        with torch.no_grad():  #关闭梯度计算
            for t_batch, t_sample_batched in enumerate(data_loader):  #遍历数据加载器
                t_sample_batched = [b.to(self.opt.device) for b in t_sample_batched]   #将批次数据移动到指定设备
                input_ids, token_type_ids, attention_mask, labels = t_sample_batched   #解包批次数据
                logits = self.model(input_ids, token_type_ids, attention_mask, labels=None)   #模型向前传播，得到logits
                score.append(F.softmax(logits, dim=-1).detach().cpu().numpy())  #将softmax后的分数添加到列表
                #计算正确预测数和总样本数
                n_correct += (torch.argmax(logits, -1) == labels).sum().item()
                n_total += len(logits)
                #累积目标标签和输出
                if t_targets_all is None:
                    t_targets_all = labels
                    t_outputs_all = logits
                else:
                    t_targets_all = torch.cat((t_targets_all, labels), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, logits), dim=0)
        return t_targets_all.cpu().numpy(), torch.argmax(t_outputs_all, -1).cpu().numpy(), np.concatenate(score, axis=0)  


    
    def make_weights_for_balanced_classes(self, labels, nclasses, fixed=False):  
        """ 为不平衡数据集生成样本权重（用于WeightedRandomSampler，解决类别分布不均问题）"""
        if fixed:  #如果使用固定权重
            weight = [0] * len(labels)   #初始化权重列表
            if nclasses == 3:   #如果是3分类 
                for idx, val in enumerate(labels):  #为每个样本设置权重
                    if val == 0:
                        weight[idx] = 0.2
                    elif val == 1:
                        weight[idx] = 0.4
                    elif val == 2:
                        weight[idx] = 0.4
                return weight  #返回权重列表
            else:  #如果是其他分类情况
                for idx, val in enumerate(labels):  #为每个样本设置权重
                    if val == 0:
                        weight[idx] = 0.2
                    else:
                        weight[idx] = 0.4
                return weight   #返回权重列表
        else:  #如果根据类别频率生成权重
            count = [0] * nclasses  #初始化类别计数列表
            for item in labels:  #统计每个类别的样本数
                count[item] += 1
            weight_per_class = [0.] * nclasses  #初始化每个类别的权重
            N = float(sum(count))   #总样本数
            
            for i in range(nclasses):  #计算每个类别的权重（总样本数/类别样本数）
                weight_per_class[i] = N / float(count[i])
            weight = [0] * len(labels)   #初始化样本权重列表
            for idx, val in enumerate(labels):  #为每个样本分配对应类别的权重
                weight[idx] = weight_per_class[val]
            return weight  #返回样本权重列表

            
    
    # =========================================================
    # ✅ RUN = Train best + Test best
    # =========================================================  
    def run(self):  
        #将训练集的标签转化为张量
        all_label_ids= torch.tensor([f['label'] for f in self.trainset], dtype=torch.long)   
        
        #将训练集转化为张量数据集，包含文本的BERT索引、段落ID、输入掩码和标签
        self.trainset = TensorDataset(torch.tensor([f['text_bert_indices'] for f in self.trainset], dtype=torch.long), 
                                      torch.tensor([f['bert_segments_ids'] for f in self.trainset], dtype=torch.long), 
                                      torch.tensor([f['input_mask'] for f in self.trainset], dtype=torch.long),
                                      all_label_ids)  
        
        if self.opt.dataset == "semeval": 
            sampler_weights = self.make_weights_for_balanced_classes(all_label_ids, 5)   #为semeval数据集创建平衡类别的采样权重（5个类别）
        else:  
            sampler_weights = self.make_weights_for_balanced_classes(all_label_ids, 3)   #为sentihood数据集创建平衡类别的采样权重（3个类别）
            
        #创建加权随机采样器，用于平衡训练集中的类别
        train_sampler = WeightedRandomSampler(sampler_weights, len(self.trainset), replacement=True)   
        #创建训练数据加载器，使用指定的训练集、批次大小和采样器
        train_data_loader= DataLoader(dataset=self.trainset, batch_size=self.opt.train_batch_size,sampler=train_sampler)

        
        #将测试集转换为张量数据集
        self.testset = TensorDataset(torch.tensor([f['text_bert_indices'] for f in self.testset], dtype=torch.long),
                                      torch.tensor([f['bert_segments_ids'] for f in self.testset], dtype=torch.long),
                                      torch.tensor([f['input_mask'] for f in self.testset], dtype=torch.long),
                                      torch.tensor([f['label'] for f in self.testset], dtype=torch.long))
        
        #将验证集转换为张量数据集
        self.valset = TensorDataset(torch.tensor([f['text_bert_indices'] for f in self.valset], dtype=torch.long),
                                     torch.tensor([f['bert_segments_ids'] for f in self.valset], dtype=torch.long),
                                     torch.tensor([f['input_mask'] for f in self.valset], dtype=torch.long),
                                     torch.tensor([f['label'] for f in self.valset], dtype=torch.long))
        #创建测试数据加载器，使用测试集和评估批次大小，不打乱数据
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.eval_batch_size, shuffle=False)  
        #创建验证数据加载器
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.eval_batch_size, shuffle=False)


        #计算总的训练步长：训练数据加载器的长度*训练轮次
        num_train_steps = int(len(train_data_loader) * self.opt.num_epoch)   
        t_total = num_train_steps   # 将总的训练步数赋值给t_total
        
        #初始化优化器：使用指定的优化器、模型参数、学习率和L2正则化
        optimizer= self.opt.optimizer(
            self.model.parameters(), 
            lr=self.opt.learning_rate,   
            weight_decay=self.opt.l2reg
        )  
        
        criterion = nn.CrossEntropyLoss()  #定义交叉熵损失函数

        #训练模型并获取最佳模型的参数
        best_model_state,best_epoch = self._train(optimizer,criterion, train_data_loader, val_data_loader, t_total)    
        #判断最佳模型状态是否为空，避免加载None报错
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)  #加载最佳模型的参数
        else:
            logger.warning("未找到最佳模型（验证集指标未超过初始值），将使用训练最后一轮的模型进行测试")

        
        self.model.eval()   #将模型设置为评估模式
        # -------------------------- 测试集上评估并输出指标 ---------------------
        logger.info(f"🔥 Testing Best Epoch = {best_epoch}")
        y_true, y_pred, score = self._evaluate_acc_f1(test_data_loader)  #在测试集上评估模型，获取真实标签、预测标签和分数
        if self.opt.dataset=='semeval':  #如果数据集是semeval，计算并输出相应的评估指标
            aspect_P, aspect_R, aspect_F = semeval_PRF(y_true, y_pred)
            sentiment_Acc_4_classes = semeval_Acc(y_true, y_pred, score, 4)
            sentiment_Acc_3_classes = semeval_Acc(y_true, y_pred, score, 3)
            sentiment_Acc_2_classes = semeval_Acc(y_true, y_pred, score, 2)
            logger.info('>> P: {:.4f} , R: {:.4f} , F: {:.4f} '.format(aspect_P, aspect_R, aspect_F))
            logger.info('>> 4 classes acc: {:.4f} '.format(sentiment_Acc_4_classes))
            logger.info('>> 3 classes acc: {:.4f} '.format(sentiment_Acc_3_classes))
            logger.info('>> 2 classes acc: {:.4f} '.format(sentiment_Acc_2_classes))
        else:   #如果数据集是sentihood，计算并输出相应的评估指标
            aspect_strict_Acc = sentihood_strict_acc(y_true, y_pred)
            aspect_Macro_F1 = sentihood_macro_F1(y_true, y_pred)
            aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC = sentihood_AUC_Acc(y_true, score)
            logger.info(())
            logger.info('>> aspect_strict_Acc: {:.4f} , aspect_Macro_F1: {:.4f} , aspect_Macro_AUC: {:.4f} '.format(
                aspect_strict_Acc, 
                aspect_Macro_F1, 
                aspect_Macro_AUC
            ))
            logger.info('>> sentiment_Acc: {:.4f} '.format(sentiment_Acc))
            logger.info('>> sentiment_Macro_AUC: {:.4f} '.format(sentiment_Macro_AUC))
    


def main():
    # Hyper Parameters（超参数设置）
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='semeval',  choices=['semeval','sentihood'], type=str, required=True)   #数据集
    parser.add_argument('--learning-rate', default=3e-5, type=float, help='try 5e-5, 2e-5')  #学习率
    parser.add_argument('--dropout', default=0.1, type=float)  #dropout率
    parser.add_argument('--l2reg', default=0.001, type=float)  #L2正则化系数
    parser.add_argument('--warmup-proportion', default=0.01, type=float)   #学习率预热比例
    parser.add_argument('--num_epoch', default=5, type=int, help='')  #训练轮数
    parser.add_argument("--train-batch-size", default=32,type=int, help="Total batch size for training.")  #训练批次大小
    parser.add_argument("--eval-batch-size", default=64, type=int, help="Total batch size for eval.")      #评估批次大小
    parser.add_argument('--log-step', default=50, type=int)    #日志输出间隔
    #预训练BERT模型名称：默认bert-base-uncased（基础版无大小写区分模型）
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)  
    
    parser.add_argument('--max_seq_len', default=120, type=int)  #文本的最大序列长度
    parser.add_argument('--label-dim', default=5, type=int)   #标签维度
    parser.add_argument('--hops', default=3, type=int)   #跳转次数
    parser.add_argument('--save_model', default=0, type=int)   #是否保存最佳模型
    parser.add_argument('--device', default='cuda', type=str, help='e.g. cuda:0')   #训练设备
    parser.add_argument('--seed', default=42, type=int, help='set seed for reproducibility')  #随机种子（设置随机种子以保证结果可复现）
    parser.add_argument('--valset_ratio', default=0, type=float,   
                        help='set ratio between 0 and 1 for validation support')   #验证集比例（0到1之间）
    opt = parser.parse_args()


    #如果数据集是sentihood，将标签维度设置为3
    if opt.dataset=='sentihood':
        opt.label_dim =3

        
    #如果设置了随机种子，为各种随机数生成器设置种子
    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    #定义数据集文件路径
    dataset_files = {
        'train': '../../datasets/{}/bert_train.json'.format(opt.dataset),
        'test': '../../datasets/{}/bert_test.json'.format(opt.dataset),
        'val': '../../datasets/{}/bert_dev.json'.format(opt.dataset)
    }

    
    logger.info(opt.pretrained_bert_name)    #打印预训练BERT模型的名称
    opt.optimizer = AdamW    #设置优化器为AdamW
    opt.dataset_file = dataset_files    #设置数据集文件路径 
    opt.inputs_cols = ['text_bert_indices', 'bert_segments_ids', 'input_mask', 'label']    #设置输入列名（BERT索引、段落ID、输入掩码、标签）
    opt.initializer = torch.nn.init.xavier_uniform_    # 设置初始化方法为xavier均匀初始化
    opt.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')   #设置设备为指定的cuda或cpu

    ins = Instructor(opt)   
    ins.run()


if __name__ == '__main__':
    main()

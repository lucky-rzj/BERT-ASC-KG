# ======================================================================
# 训练主程序----执行训练 → 验证 → 测试 → 保存模型
# ======================================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import logging
import argparse
import sys
import random
import numpy
import  numpy as np
from transformers import AdamW
import torch
from torch.utils.data.sampler import  WeightedRandomSampler    #导入加权随机采样器
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.nn.functional as F
import copy
from  tqdm import tqdm


# --------------------- 自定义模块导入 --------------------------
from data_utils_plm import(ABSATokenizer, ABSADataset_absa_bert_semeval_json, ABSADataset_absa_bert_sentihood_json)  
from evaluation import *      
from MyModel_plm import PLM_ASC    


# ------------------------- 日志记录 ------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


# ------------------------- 核心训练类 ------------------------------
class Instructor:
    """训练指导类：整合ABSA任务的全流程（数据加载→模型初始化→训练→评估→保存）"""
    def __init__(self, opt):
        """初始化函数：接收配置参数，完成数据集加载、模型初始化"""
        self.opt = opt
        
        # 1. 初始化模型
        self.model = PLM_ASC(opt).to(opt.device)
        
        # 2. 初始化分词器
        tokenizer = ABSATokenizer.from_pretrained(opt.pretrained_model)
        
        # 3. 加载数据集
        if self.opt.dataset=='semeval':
            self.trainset = ABSADataset_absa_bert_semeval_json(opt.dataset_file['train'], tokenizer, opt)
            self.testset = ABSADataset_absa_bert_semeval_json(opt.dataset_file['test'], tokenizer, opt)
            assert 0 <= opt.valset_ratio < 1
            if opt.valset_ratio > 0:
                valset_len = int(len(self.trainset) * opt.valset_ratio)
                self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))
            else:
                self.valset = self.testset
        else:
            self.trainset = ABSADataset_absa_bert_sentihood_json(opt.dataset_file['train'], tokenizer, opt)
            self.testset = ABSADataset_absa_bert_sentihood_json(opt.dataset_file['test'], tokenizer, opt)
            self.valset =self.testset
        logger.info('train {0}: dev {1}: test {2}'.format(len(self.trainset), len(self.valset), len(self.testset)))
        
        # 4. GPU显存监控
        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()


    def _print_args(self):
        """辅助函数：打印模型可训练参数数量和所有训练配置参数"""
        n_trainable_params, n_nontrainable_params = 0, 0   
        for p in self.model.parameters():    #遍历模型所有参数
            n_params = torch.prod(torch.tensor(p.shape))   #计算单个参数张量的元素个数
            if p.requires_grad:    #若参数需要梯度更新（可训练）
                n_trainable_params += n_params
            else:    #若参数固定（不可训练）
                n_nontrainable_params += n_params
        #输出参数数量
        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')   
        for arg in vars(self.opt):  
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))  


    def warmup_linear(self, x, warmup=0.002):
        """线性学习率预热函数：训练初期逐步提升学习率，避免梯度震荡"""
        if x < warmup:
            return x / warmup   #预热阶段：学习率随步数线性增长
        else:
            return max((1.0 - x), 0.0)   #预热后：学习率随步数线性衰减


    
    # =====================================================================
    # ✅ TRAIN (with validation and best epoch selection)
    # =====================================================================
    def _train(self, optimizer, train_data_loader, val_data_loader, t_total):
        """核心训练函数：执行多轮训练、学习率调度、验证集评估、最佳模型保存"""
        max_val_f1 = 0   #记录验证集最佳性能指标
        global_step = 0
        path = None       #最佳模型参数保存路径
        best_epoch = -1   #记录最优轮次
        
        for epoch in range(self.opt.num_epoch):
            loss_total= 0
            step_total= 0   
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            self.model.train()
            
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            
            # ---------------------- 遍历训练集 -------------------------
            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):
                optimizer.zero_grad()   #梯度清零
                sample_batched= [b.to(self.opt.device) for b in sample_batched]
                input_ids, attention_mask, labels= sample_batched

                # ----------- AMP 自动混合精度开始 -----------
                with autocast():
                    loss = self.model(input_ids, attention_mask, labels)   #模型前向传播：计算损失
                    
                # ----------- AMP 自动混合精度结束 -----------
                    
                #反向传播（使用混合精度 scaler）
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
      
                #累计损失和样本数
                with torch.no_grad():
                    loss_total+= loss.item()
                    step_total+=len(labels)

                    
                #计算当前步数的学习率
                lr_this_step = self.opt.learning_rate * self.warmup_linear(global_step / t_total, self.opt.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step   #更新所有参数组的学习率
                optimizer.step()   #梯度下降：更新模型参数
                global_step += 1

            
            logger.info(" epoch : {0}, training loss: {1} ".format(str(epoch), loss_total / step_total))  
            #-------------------------- 在验证集上评估并打印验证集指标 ---------------------
            y_true, y_pred, score = self._evaluate_acc_f1(val_data_loader)    #验证集评估：获取真实标签、预测标签、预测分数
            if self.opt.dataset  == 'semeval':
                aspect_P, aspect_R, aspect_F = semeval_PRF(y_true, y_pred)
                sentiment_Acc_4_classes = semeval_Acc(y_true, y_pred, score, 4)
                sentiment_Acc_3_classes = semeval_Acc(y_true, y_pred, score, 3)
                sentiment_Acc_2_classes = semeval_Acc(y_true, y_pred, score, 2)
                max_per = sentiment_Acc_4_classes    #以4类情感准确率作为最佳指标

                logger.info('')
                logger.info('>> P: {:.4f} , R: {:.4f} , F: {:.4f} '.format(aspect_P, aspect_R, aspect_F))
                logger.info('>> 2 classes acc: {:.4f} '.format(sentiment_Acc_2_classes))
                logger.info('>> 3 classes acc: {:.4f} '.format(sentiment_Acc_3_classes))
                logger.info('>> 4 classes acc: {:.4f} '.format(sentiment_Acc_4_classes))
            else:
                aspect_strict_Acc = sentihood_strict_acc(y_true, y_pred)
                aspect_Macro_F1 = sentihood_macro_F1(y_true, y_pred)
                aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC = sentihood_AUC_Acc(y_true, score)
                max_per = aspect_strict_Acc   #以严格准确率作为最佳指标

                logger.info('')
                logger.info('>> aspect_strict_Acc: {:.4f} , aspect_Macro_F1: {:.4f} , aspect_Macro_AUC: {:.4f} '.format(
                    aspect_strict_Acc,    
                    aspect_Macro_F1,
                    aspect_Macro_AUC
                ))
                logger.info('>> sentiment_Acc: {:.4f} '.format(sentiment_Acc))
                logger.info('>> sentiment_Macro_AUC: {:.4f} '.format(sentiment_Macro_AUC))


            # ------------------ Save best model ------------------            
            if max_per > max_val_f1:   #当前验证集性能 > 历史最佳
                max_val_f1 = max_per   #更新最佳指标
                best_epoch = epoch    #记录最优轮次
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = copy.deepcopy(self.model.state_dict())   #深拷贝模型当前参数
                
    
            self.model.train()  
        logger.info(f"🔥 Training Finished. Best Epoch = {best_epoch}")
        return path, best_epoch    #返回最佳模型的参数字典、最优轮次


        
    # ------------------------------ 核心评估函数 ------------------------------
    def _evaluate_acc_f1(self, data_loader):
        """核心评估函数：获取评估数据集的真实标签、预测标签、预测分数（用于后续计算各类指标）"""
        n_correct, n_total = 0, 0    #正确预测数、总样本数
        t_targets_all, t_outputs_all = None, None    #累计所有批次的真实标签、模型输出logits
        score = []   #存储每个批次的预测分数
        self.model.eval()
        with torch.no_grad():
            #遍历评估数据集的所有批次
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_sample_batched = [b.to(self.opt.device) for b in t_sample_batched]
                input_ids, attention_mask, labels = t_sample_batched
                
                #模型前向传播（预测时不传入labels，返回logits）
                logits = self.model(input_ids, attention_mask)

                score.append(F.softmax(logits, dim=-1).detach().cpu().numpy())  #计算预测分数并添加到score列表

                n_correct += (torch.argmax(logits, -1) == labels).sum().item()
                n_total += len(logits)
                
                #拼接所有批次的真实标签和logits
                if t_targets_all is None:
                    t_targets_all = labels
                    t_outputs_all = logits
                else:
                    t_targets_all = torch.cat((t_targets_all, labels), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, logits), dim=0)
        #转换为numpy数组并返回：真实标签、预测标签、预测分数
        return t_targets_all.cpu().numpy(), torch.argmax(t_outputs_all, -1).cpu().numpy(), np.concatenate(score, axis=0)


    def make_weights_for_balanced_classes(self, labels, nclasses, fixed=False):
        """ 为不平衡数据集生成样本权重（用于WeightedRandomSampler，解决类别分布不均问题）"""
        if fixed:   #手动固定权重模式
            weight = [0] * len(labels)
            if nclasses == 3:   #3分类场景：标签0权重0.2，标签1和2权重0.4
                for idx, val in enumerate(labels):
                    if val == 0:
                        weight[idx] = 0.2
                    elif val == 1:
                        weight[idx] = 0.4
                    elif val == 2:
                        weight[idx] = 0.4
                return weight
            else:   #其他分类场景：标签0权重0.2，其他标签权重0.4
                for idx, val in enumerate(labels):
                    if val == 0:
                        weight[idx] = 0.2
                    else:
                        weight[idx] = 0.4
                return weight
        else:   #自动计算权重模式
            count = [0] * nclasses
            for item in labels:
                idx = int(item)   
                count[idx] += 1
            weight_per_class = [0.] * nclasses  
            N = float(sum(count))  
            
            #计算每个类别的权重：总样本数 / 该类样本数（频率越高，权重越低）
            for i in range(nclasses):
                weight_per_class[i] = N / float(count[i])
            weight = [0] * len(labels)  
            for idx, val in enumerate(labels):
                weight[idx] = weight_per_class[val]  
            return weight


    
    # ======================================================
    # ✅ RUN = Train best + Test best
    # ======================================================
    def run(self):
        """
        训练流程主函数：整合ABSA任务全流程（数据集预处理→数据加载→优化器初始化→训练→评估→保存）
        是Instructor类的核心入口，调用其他函数完成端到端训练
        """
        # 1. 提取训练集所有标签并转换为PyTorch长整型张量
        all_label_ids = torch.tensor([f['label'] for f in self.trainset], dtype=torch.long)   

        # 2. 将训练集转换为TensorDataset格式，包含3个张量：token索引、注意力掩码、标签张量
        self.trainset = TensorDataset(
            torch.tensor([f['text_bert_indices'] for f in self.trainset], dtype=torch.long), 
            torch.tensor([f['input_mask'] for f in self.trainset], dtype=torch.long), 
            all_label_ids
        )
        
        # 3. 生成训练集样本权重
        if self.opt.dataset == "semeval":
            sampler_weights = self.make_weights_for_balanced_classes(all_label_ids, 5)
        else:
            sampler_weights = self.make_weights_for_balanced_classes(all_label_ids, 3)

        # 4. 初始化加权随机采样器
        train_sampler = WeightedRandomSampler(sampler_weights, len(self.trainset), replacement=True)
        # 5. 构建训练集数据加载器
        train_data_loader= DataLoader(dataset=self.trainset, batch_size=self.opt.train_batch_size,sampler=train_sampler)

        # 6. 转换测试集为TensorDataset格式
        self.testset = TensorDataset(torch.tensor([f['text_bert_indices'] for f in self.testset], dtype=torch.long),
                                      torch.tensor([f['input_mask'] for f in self.testset], dtype=torch.long),
                                      torch.tensor([f['label'] for f in self.testset], dtype=torch.long))
        # 7. 转换验证集为TensorDataset格式
        self.valset = TensorDataset(torch.tensor([f['text_bert_indices'] for f in self.valset], dtype=torch.long),
                                     torch.tensor([f['input_mask'] for f in self.valset], dtype=torch.long),
                                     torch.tensor([f['label'] for f in self.valset], dtype=torch.long))
        
        # 8. 构建测试集/验证集数据加载器
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.eval_batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.eval_batch_size, shuffle=False)

        # 9. 计算总训练步数 = 每个epoch的批次数量 × 训练轮数
        num_train_steps = int(len(train_data_loader) * self.opt.num_epoch)
        t_total = num_train_steps
        
        # 10. 初始化优化器：传入模型参数、学习率、L2正则化系数
        optimizer= self.opt.optimizer(
            self.model.parameters(), 
            lr=self.opt.learning_rate,                        
            weight_decay=self.opt.l2reg
        )

        # 11. 启动训练：调用_train函数，返回最佳模型的参数字典（state_dict）
        best_model_path, best_epoch = self._train(optimizer, train_data_loader, val_data_loader, t_total)
        # 12. 加载最佳模型参数（基于验证集性能的最优参数，用于最终测试）
        self.model.load_state_dict(best_model_path)


        # 13. 测试集最终评估
        self.model.eval()
        #-------------------------- 测试集上评估并输出指标 ---------------------
        logger.info(f"🔥 Testing Best Epoch = {best_epoch}")
        y_true, y_pred, score = self._evaluate_acc_f1(test_data_loader)     #在测试集上评估模型，获取真实标签、预测标签和分数
        if self.opt.dataset=='semeval':
            aspect_P, aspect_R, aspect_F = semeval_PRF(y_true, y_pred)
            sentiment_Acc_4_classes = semeval_Acc(y_true, y_pred, score, 4)
            sentiment_Acc_3_classes = semeval_Acc(y_true, y_pred, score, 3)
            sentiment_Acc_2_classes = semeval_Acc(y_true, y_pred, score, 2)
            
            logger.info('>> P: {:.4f} , R: {:.4f} , F: {:.4f} '.format(aspect_P, aspect_R, aspect_F))
            logger.info('>> 4 classes acc: {:.4f} '.format(sentiment_Acc_4_classes))
            logger.info('>> 3 classes acc: {:.4f} '.format(sentiment_Acc_3_classes))
            logger.info('>> 2 classes acc: {:.4f} '.format(sentiment_Acc_2_classes))
        else:
            aspect_strict_Acc = sentihood_strict_acc(y_true, y_pred)
            aspect_Macro_F1 = sentihood_macro_F1(y_true, y_pred)
            aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC = sentihood_AUC_Acc(y_true, score)
            
            logger.info('>> aspect_strict_Acc: {:.4f} , aspect_Macro_F1: {:.4f} , aspect_Macro_AUC: {:.4f} '.format(
                    aspect_strict_Acc,    
                    aspect_Macro_F1,
                    aspect_Macro_AUC
            ))
            logger.info('>> sentiment_Acc: {:.4f} '.format(sentiment_Acc))
            logger.info('>> sentiment_Macro_AUC: {:.4f} '.format(sentiment_Macro_AUC))

            
        # ------------------------  按 seed 保存模型 ----------------------------------
        # 14. 保存最佳模型参数
        if self.opt.save_model:
            save_dir = f"state_dict/{self.opt.dataset}"
            os.makedirs(save_dir, exist_ok=True)
            #构建模型保存路径：state_dict/数据集名/seed.bm
            save_path = f"{save_dir}/seed{self.opt.seed}.bm"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.model.state_dict(), save_path)   #保存模型参数字典
            logger.info(f"💾 Model saved to {save_path}")



def main():
    """主函数：解析命令行参数、设置随机种子、配置数据集路径、初始化训练指导类并启动训练"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='semeval',  choices=['semeval','sentihood'], type=str, required=True)
    parser.add_argument('--learning_rate', default=3e-5, type=float, help='try 5e-5, 2e-5')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.001, type=float)
    parser.add_argument('--warmup_proportion', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=5, type=int, help='')
    parser.add_argument("--train_batch_size", default=32,type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for eval.")
    parser.add_argument('--log_step', default=50, type=int)
    parser.add_argument('--max_seq_len', default=120, type=int)
    parser.add_argument('--label_dim', default=5, type=int)
    parser.add_argument('--pretrained_model', default='bart-base', type=str) 
    parser.add_argument('--save_model', default=0, type=int)
    parser.add_argument('--device', default='cuda', type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=42, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float,
                        help='set ratio between 0 and 1 for validation support')
    opt = parser.parse_args()


    if opt.dataset=='sentihood':
        opt.label_dim =3

        
    #设置随机种子（保证结果可复现）
    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        
    #配置数据集文件路径（训练集、测试集、验证集）
    dataset_files = {
        'train': '../../../datasets/{}/bert_train.json'.format(opt.dataset),
        'test': '../../../datasets/{}/bert_test.json'.format(opt.dataset),
        'val': '../../../datasets/{}/bert_dev.json'.format(opt.dataset)
    }

    
    logger.info(opt.pretrained_model) 
    opt.optimizer = AdamW
    opt.model_class = ABSATokenizer
    opt.dataset_file = dataset_files
    opt.inputs_cols = ['text_bert_indices', 'input_mask', 'label']   #配置模型输入列名（与数据集张量对应）
    opt.initializer = torch.nn.init.xavier_uniform_
    opt.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()

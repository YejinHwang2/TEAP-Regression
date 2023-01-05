import sys
from pathlib import Path
import pandas as pd
import numpy as np
from copy import deepcopy
from collections import OrderedDict

from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from typing import Dict
from src.utils import RegressionMetricTaskRecorder, RegressionMetricRecorder
from src.dataset import StockRegressionDataset, PanelDataDict
from src.utils import ARGProcessor
from src.model import PanelRegressionModel
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class Maml_Regression_Trainer(nn.Module):
    def __init__(
        self, exp_name, log_dir, task_type,  model, batch_size,
        n_inner_step, total_steps, 
        n_valid_step, every_valid_step, print_step,
        inner_lr, outer_lr, device, clip_value, test_window_size):
        
        super(Maml_Regression_Trainer, self).__init__()
        self.exp_name = exp_name
        self.log_dir = Path(log_dir).resolve()
        self.device = device
        self.model = model.to(self.device)
        if task_type == "classification":
            self.loss_fn = nn.NLLLoss() 
        elif task_type == "regression":
            self.loss_fn = nn.MSELoss()
            
        self.n_inner_step = n_inner_step
        self.total_steps = total_steps
        self.n_valid_step = n_valid_step
        self.every_valid_step = every_valid_step
        self.print_step = print_step
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.batch_size = batch_size
        self.test_window_size = test_window_size[0]
        self.keep_weights = self.clone_weight(self.model)
        self.meta_optim = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.outer_lr
        )
        if self.device == 'cuda':
            self.cuda()
        
        # Recoder
        self.train_recorder = RegressionMetricRecorder().to(device)
        self.valid_recorder = RegressionMetricRecorder().to(device)
        self.test_recorder = RegressionMetricRecorder().to(device)
        
        self.task_recorder = RegressionMetricTaskRecorder().to(device)
 

        
    def init_experiments(self, exp_num=None, record_tensorboard: bool=True):
        # check if exp exists
        exp_dirs = sorted(list(self.log_dir.glob(f'{self.exp_name}_*')))
        if exp_num is None:
            exp_num = int(exp_dirs[-1].name[len(self.exp_name)+1:]) if exp_dirs else 0
            self.exp_num = exp_num + 1
        else:
            self.exp_num = exp_num
        self.exp_dir = self.log_dir / f'{self.exp_name}_{self.exp_num}'
        if record_tensorboard:
            self.writer = SummaryWriter(str(self.exp_dir))
        else:
            self.writer = None
        self.ckpt_path = self.exp_dir / 'checkpoints'
        self.ckpt_step_train_path =  self.ckpt_path / 'step' / 'train'
        self.ckpt_step_valid_path =  self.ckpt_path / 'step' / 'valid'
        for p in [self.ckpt_path, self.ckpt_step_train_path, self.ckpt_step_valid_path]:
            if not p.exists():
                p.mkdir(parents=True)    
    
    def get_acc(self,y_true, y_pred):
        correct = torch.eq(y_pred, y_true).sum().item()
        acc = correct/ len(y_true)
        return acc

    def clone_weight(self, model):
        return {k: v.clone() for k, v in model.state_dict().items()}

    def meta_update(self, dummy_loss, sum_grads):
        # Update theta_parameter by sum_gradients
        hooks = []
        for k,v in enumerate(self.model.parameters()):
            def closure():
                key = k
                return lambda grad: sum_grads[key]
            hooks.append(v.register_hook(closure()))

        self.meta_optim.zero_grad()
        dummy_loss.backward()
        self.meta_optim.step()

        for h in hooks:
            h.remove()
                
    # inner loop per 1 task
    def inner_loop(self, support_x, support_y, query_x, query_y, is_meta_train):
        updated_state_dict = {k: v.clone() for k, v in self.keep_weights.items()}
        for i in range(self.n_inner_step):
            if i > 0:
                self.model.load_state_dict(updated_state_dict)
            support_e = self.model(support_x)
            s_loss = self.loss_fn(support_e, support_y)
            grad = torch.autograd.grad(
                    s_loss, 
                    self.model.parameters(),
                    create_graph=True,
                )
            for i, (k,w) in enumerate(updated_state_dict.items()):
                updated_state_dict[k] = updated_state_dict[k] - self.inner_lr * grad[i].data
        
        s_mse = self.task_recorder.metrics['Support_MSE'](support_e, support_y)
        s_mae = self.task_recorder.metrics['Support_MAE'](support_e, support_y)
        s_mape = self.task_recorder.metrics['Support_MAPE'](support_e, support_y)
       
        self.model.load_state_dict(updated_state_dict)
        query_e = self.model(query_x)
        q_loss = self.loss_fn(query_e, query_y)
        
        q_mse = self.task_recorder.metrics['Query_MSE'](query_e, query_y)
        q_mae = self.task_recorder.metrics['Query_MAE'](query_e, query_y)
        q_mape = self.task_recorder.metrics['Query_MAPE'](query_e, query_y)
        
        
        if is_meta_train:
            q_grad = torch.autograd.grad(q_loss, self.model.parameters(), create_graph=True)
        else:
            q_grad = None
        
        
        return s_mse, s_mae, s_mape, q_mse, q_mae, q_mape, q_grad, query_e

    # outer loop per batch
    def outer_loop(self, meta_dataset):
        self.model.meta_train()
        batch_task, window_size = meta_dataset.generate_batch_task(all_tasks=meta_dataset.all_tasks) # PanelDataDict
        train_tasks  = PanelDataDict(batch_task,window_size = window_size)
        train_tasks.to(self.device)
        all_q_grads = []

        self.keep_weights = self.clone_weight(self.model)
        
        for i  in range(self.batch_size):
            x_spt = train_tasks['support'][i]
            y_spt = train_tasks['support_labels'][i]
            x_qry = train_tasks['query'][i]
            y_qry = train_tasks['query_labels'][i]
            s_mse, s_mae, s_mape, q_mse, q_mae, q_mape, q_grad, query_e = self.inner_loop(x_spt, y_spt, x_qry, y_qry, is_meta_train=True)
            self.train_recorder.update('Support_MSE', s_mse)
            self.train_recorder.update('Support_MAE', s_mae)
            self.train_recorder.update('Support_MAPE', s_mape)
            self.train_recorder.update('Query_MSE', q_mse)
            self.train_recorder.update('Query_MAE', q_mae)
            self.train_recorder.update('Query_MAPE', q_mape)
            
            
            all_q_grads.append(q_grad)
            self.model.load_state_dict(self.keep_weights)
            
        
        sum_q_grads = [torch.stack(grads).sum(dim=0) for grads in list(zip(*all_q_grads))]
        
        x_spt = train_tasks['support'][0]
        y_spt = train_tasks['support_labels'][0]
        
        dummy_e = self.model(x_spt)
        dummy_loss = self.loss_fn(dummy_e, y_spt)
        
        self.meta_update(dummy_loss, sum_q_grads)
        return 


    def meta_train(self, meta_trainset,
                meta_validset_time,
                meta_validset_entity,
                meta_validset_mix, 
                print_log: bool=True):
        
        best_eval_mse = 10000.0
        for step in range(self.total_steps):
            self.train_recorder.reset()
            # Meta-Train per epoch
            self.outer_loop(meta_trainset)
            if ( step % self.print_step == 0) or (step == self.total_steps-1):
                prefix = 'Train'
                train_logs = self.train_recorder.compute(prefix)
                cur_eval_mse = train_logs[f'{prefix}-Query_MSE']
                cur_eval_mae = train_logs[f'{prefix}-Query_MAE']
                cur_eval_mape = train_logs[f'{prefix}-Query_MAPE']
                
                self.log_results(train_logs, prefix, step=step, total_steps=self.total_steps, print_log=True)
                torch.save(self.model.state_dict(), str(self.ckpt_step_train_path / f'{step}-{cur_eval_mse:.4f}.ckpt'))

                
            # Meta-Valid
            if (self.every_valid_step != 0):
                if (step % self.every_valid_step == 0) or (step == self.total_steps-1):
                    ref_step = step
                    
                    prefix = 'Valid-time'
                    valid_time_logs, cur_eval_mse_time, cur_eval_mae_time, cur_eval_mape_time = self.meta_valid(self.model, meta_validset_time, prefix, ref_step, self.n_valid_step)
                    
                    prefix = 'Valid-entity'
                    valid_entity_logs, cur_eval_mse_entity, cur_eval_mae_entity, cur_eval_mape_entity = self.meta_valid(self.model, meta_validset_entity, prefix, ref_step, self.n_valid_step)
                    
                    prefix = 'Valid-mix'
                    valid_mix_logs, cur_eval_mse_mix, cur_eval_mae_mix, cur_eval_mape_mix = self.meta_valid(self.model, meta_validset_mix, prefix, ref_step, self.n_valid_step)
                    
                    prefix = 'Valid'
                    cur_eval_mse = (cur_eval_mse_time + cur_eval_mse_entity + cur_eval_mse_mix) / 3
                    cur_eval_mae = (cur_eval_mae_time + cur_eval_mae_entity + cur_eval_mae_mix) / 3
                    cur_eval_mape = (cur_eval_mape_time + cur_eval_mape_entity + cur_eval_mape_mix) / 3
                    valid_final_log = {f'{prefix}-AvgMSE': cur_eval_mse, f'{prefix}-AvgMAE': cur_eval_mae, f'{prefix}-AvgMAPE': cur_eval_mape}
                    self.log_results(valid_final_log, prefix, step=ref_step, total_steps=self.total_steps, print_log=print_log)
                    
                    # save best
                    if (cur_eval_mse < best_eval_mse):
                        best_eval_mse = cur_eval_mse 
                        torch.save(self.model.state_dict(), str(self.ckpt_step_valid_path / f'{ref_step:06d}-{cur_eval_mse:.4f}.ckpt'))
                    
    def meta_valid(self, model, meta_dataset, prefix, ref_step, n_valid, print_log=True):
        self.valid_recorder.reset()
        valid_logs = self.run_valid(model, meta_dataset, n_valid, prefix)
        self.log_results(valid_logs, prefix, step=ref_step, total_steps=self.total_steps, print_log=print_log)
        cur_eval_mse = valid_logs[f'{prefix}-Query_MSE']
        cur_eval_mae = valid_logs[f'{prefix}-Query_MAE']
        cur_eval_mape = valid_logs[f'{prefix}-Query_MAPE']
        return valid_logs, cur_eval_mse, cur_eval_mae, cur_eval_mape
        
    def meta_test(self, model, meta_dataset,  print_log: bool=True):
        self.test_recorder.reset()
        prefix = meta_dataset.meta_type.capitalize()
        test_logs = self.run_test(model, meta_dataset, prefix)
        self.log_results(test_logs, prefix, step=0, total_steps=0, print_log=print_log)
        eval_mse = test_logs[f'{prefix}-Query_MSE']
        eval_mae = test_logs[f'{prefix}-Query_MAE']
        eval_mape = test_logs[f'{prefix}-Query_MAPE']
        return prefix, eval_mse, eval_mae, eval_mape
    
    def run_valid(self, model, meta_dataset, n_valid, prefix):
        model = model.to(self.device)
        model.meta_eval()
        pregress = tqdm(range(n_valid), total= n_valid, desc=f'Running {prefix}')
     
        for val_idx in pregress:
            batch_task, window_size = meta_dataset.generate_batch_task(all_tasks=meta_dataset.all_tasks) # PanelDataDict
            valid_tasks  = PanelDataDict(batch_task,window_size = window_size)
            valid_tasks.to(self.device)
            for i  in range(self.batch_size):
                x_spt = valid_tasks['support'][i]
                y_spt = valid_tasks['support_labels'][i]
                x_qry = valid_tasks['query'][i]
                y_qry = valid_tasks['query_labels'][i]
                s_mse, s_mae, s_mape, q_mse, q_mae, q_mape, q_grad, query_e = self.inner_loop(x_spt, y_spt, x_qry, y_qry, is_meta_train=False)
                self.valid_recorder.update('Support_MSE', s_mse)
                self.valid_recorder.update('Support_MAE', s_mae)
                self.valid_recorder.update('Support_MAPE', s_mape)
                self.valid_recorder.update('Query_MSE', q_mse)
                self.valid_recorder.update('Query_MAE', q_mae)
                self.valid_recorder.update('Query_MAPE', q_mape)
        
        
        pregress.close()
        valid_logs = self.valid_recorder.compute(prefix)       
        return valid_logs
    
    def run_test(self, model, meta_dataset, prefix):
        model = model.to(self.device)
        model.meta_eval()
        test_all_tasks = meta_dataset.all_tasks[self.test_window_size]
        test_tasks = PanelDataDict(test_all_tasks, window_size = self.test_window_size)
        test_tasks.to(self.device)
        pregress = tqdm(range(len(test_tasks['query'])), total= len(test_tasks['query']), desc=f'Running {prefix}')
        for test_idx in pregress:
            x_spt = test_tasks['support'][test_idx]
            y_spt = test_tasks['support_labels'][test_idx]
            x_qry = test_tasks['query'][test_idx]
            y_qry = test_tasks['query_labels'][test_idx]
            s_mse, s_mae, s_mape, q_mse, q_mae, q_mape, q_grad, query_e = self.inner_loop(x_spt, y_spt, x_qry, y_qry, is_meta_train=False)
            self.test_recorder.update('Support_MSE', s_mse)
            self.test_recorder.update('Support_MAE', s_mae)
            self.test_recorder.update('Support_MAPE', s_mape)
            self.test_recorder.update('Query_MSE', q_mse)
            self.test_recorder.update('Query_MAE', q_mae)
            self.test_recorder.update('Query_MAPE', q_mape)
        
        pregress.close()
        test_logs = self.test_recorder.compute(prefix)       
        return test_logs
    
    def log_results(self, logs, prefix, step, total_steps, print_log=False):
        
        for log_string, value in logs.items():       
            if self.writer is not None:
                self.writer.add_scalar(log_string, value, step)
                
        def extract(prefix, key, logs):
            mean = logs[f'{prefix}-{key}']
            s = ''
            s += f'{mean:.4f}'
            return s

        if print_log:
            only_one_to_print = True if prefix in ['Valid', 'Test'] else False

            if only_one_to_print:
                avgmse = extract(prefix, 'AvgMSE', logs)
                avgmae = extract(prefix, 'AvgMAE', logs)
                avgmape = extract(prefix, 'AvgMAPE', logs)
         
                print(f'[Meta {prefix}] Result - AvgMSE: {avgmse}, AvgMAE: {avgmae}, AvgMAPE: {avgmape} ')
                print()

            else:
                s_mse = extract(prefix, 'Support_MSE', logs)
                s_mae = extract(prefix, 'Support_MAE', logs)
                s_mape = extract(prefix, 'Support_MAPE', logs)
                q_mse = extract(prefix, 'Query_MSE', logs)
                q_mae = extract(prefix, 'Query_MAE', logs)
                q_mape = extract(prefix, 'Query_MAPE', logs)

                print(f'[Meta {prefix}]({step+1}/{total_steps})')
                print(f'  - [Support] MSE: {s_mse}, MAE: {s_mae}, MAPE: {s_mape}')
                print(f'  - [Query] MSE: {q_mse}, MAE: {q_mae}, MAPE: {q_mape}')
                print()
                
    def get_best_results(self, exp_num, record_tensorboard: bool=True):
        self.init_experiments(exp_num=exp_num, record_tensorboard=record_tensorboard)
        best_ckpt = sorted(
            (self.ckpt_step_valid_path).glob('*.ckpt'),
            key=lambda x: x.name.split('-')[1], 
            reverse=True
        )[0]
        
        best_step, train_loss = best_ckpt.name.rstrip('.ckpt').split('-')
        state_dict = torch.load(best_ckpt)
        return int(best_step), float(train_loss), state_dict

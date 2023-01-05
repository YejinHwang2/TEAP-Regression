from collections import defaultdict
import yaml
import inspect
import torch
import torch.nn as nn
import torchmetrics as tm
from typing import Dict, Tuple, List

class ARGProcessor():
    def __init__(self, setting_file):
        self.setting_file = setting_file
        self.load()

    def load(self):
        with open(self.setting_file) as file:
            self.kwargs = yaml.load(file, Loader=yaml.FullLoader)

    def save(self, path):
        with open(path, 'w') as file:
            yaml.dump(self.kwargs, file)
    
    def get_args(self, cls):
        cls_kwargs = {
            k: self.kwargs.get(k) for k in inspect.signature(cls.__init__).parameters.keys() 
            if self.kwargs.get(k) is not None
        }
        return cls_kwargs

class RegressionMetricTaskRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        cs = tm.MetricCollection({
            'Support_MSE': tm.MeanSquaredError(), 
            'Support_MAE': tm.MeanAbsoluteError(),
            'Support_MAPE': tm.MeanAbsolutePercentageError(),
            'Query_MSE': tm.MeanSquaredError(), 
            'Query_MAE': tm.MeanAbsoluteError(),
            'Query_MAPE': tm.MeanAbsolutePercentageError(),
        })
        
        self.metrics = cs.clone()
        
    @property
    def keys(self):
        return list(self.metrics.keys())


class RegressionMetricRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        cs = tm.MetricCollection({
            'Support_MSE': tm.MeanMetric(), 
            'Support_MAE': tm.MeanMetric(),
            'Support_MAPE': tm.MeanMetric(),
            'Query_MSE': tm.MeanMetric(), 
            'Query_MAE': tm.MeanMetric(),
            'Query_MAPE': tm.MeanMetric(),
        })

        self.metrics = cs.clone()
    @property
    def keys(self):
        return list(self.metrics.keys())
    
    def update(self, key, scores):
        self.metrics[key].update(scores)
            
    def compute(self, prefix: str):
        results = {}
        for k in self.keys:
            m = self.metrics[k].compute()
            if isinstance(m, torch.Tensor):
                m = m.cpu().detach().numpy()
            results[f'{prefix}-{k}'] = m
        return results
    
    def reset(self):
        for k in self.keys:
            self.metrics[k].reset()
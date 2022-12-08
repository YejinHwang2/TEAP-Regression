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

class MetricRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        cs = tm.MetricCollection({
            'Support_Accuracy': tm.MeanMetric(), 
            'Support_Loss': tm.MeanMetric(),
            'Query_Accuracy': tm.Accuracy(),
            'Query_Loss': tm.MeanMetric()
        })
        # cs_query = tm.MetricCollection({
        #     'Accuracy': tm.Accuracy(), 
        #     'Precision': tm.Precision(num_classes=2, average=None), 
        #     'Recall': tm.Recall(num_classes=2, average=None), 
        #     'Loss': tm.SumMetric(),
        # })
        self.metrics = cs.clone()
    @property
    def keys(self):
        return list(self.metrics.keys())
    
    def update(self, key, scores=None | torch.FloatTensor, targets=None | torch.LongTensor):
        if key == 'Query_Accuracy':
            if targets is None:
                raise KeyError('Must insert `targets` to calculate accuracy.')
            self.metrics[key].update(scores, targets)
            
        else:
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
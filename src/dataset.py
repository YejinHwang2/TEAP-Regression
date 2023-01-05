import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Generator
from tqdm import tqdm
from collections import Counter
import random

class PanelDataDict(dict):
    def __init__(self, data, window_size):
        self.window_size = window_size
        self._set_state(f'numpy')
        for k, v in data.items():
            data[k] = np.array(v)
        
        self.n_stocks = len(v)
        super().__init__(data)
    
    def tensor_fn(self, value, key):
        return torch.FloatTensor(value)

    def _set_state(self, state: str):
        self.state = state

    def to(self, device: None | str=None):
        if device is None:
            device = torch.device('cpu')
        else:
            device = torch.device(device)
        self._set_state(f'tensor.{device}')
        for key in self.keys():
            value = self.__getitem__(key)
            tvalue = self.tensor_fn(value, key)
            self.__setitem__(key, tvalue.to(device)) 
        
    def numpy(self):
        self._set_state('numpy')
        for key in self.keys():
            tvalue = self.__getitem__(key)
            if not isinstance(tvalue, np.ndarray): 
                self.__setitem__(key, tvalue.detach().numpy())

    def __str__(self):
        s = f'PanelDataDict(T={self.window_size}, {self.state})\n'
        for i, key in enumerate(self.keys()):
            value = self.__getitem__(key)
            s += f'- {key}: {value.shape}'
            s += '' if i == len(self.keys())-1 else '\n'
        return s

class StockRegressionDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            meta_type: str ='train', 
            data_dir: Path | str ='', 
            dtype: str ='kdd17', 
            batch_size: int =64,
            n_support: int =5, 
            n_query: int = 3,
            test_window_size: List[int] =[15],
            window_sizes: List[int] =[15]
        ):    
        """dataset ref: https://arxiv.org/abs/1810.09936

        In this meta learning setting, we have 3 meta-test and 1 meta-train
        vertical = stocks, horizontal = time
                train      |    test
           A               |
           B   meta-train  |   meta-test
           C               |      (1)
           ----------------|-------------
           D   meta-test   |   meta-test
           E     (2)       |      (3)

        meta-test (1) same stock, different time
        meta-test (2) different stock, same time
        meta-test (3) different stock, different time
        use `valid_date` to split the train / test set

        the number of training stock was splitted with number of total stocks * 0.8.
        we have 5 stock universe

        Args:
            meta_type (str, optional): _description_. Defaults to 'train'.
            data_dir (Path | str, optional): _description_. Defaults to ''.
            dtype (str, optional): _description_. Defaults to 'kdd17'.
            stock_universe (int, optional): _description_. Defaults to 0.
            batch_size (int, optional): Batch size. Number of stock x Number of timestamp that is aviable for each window size. Defaults to 64.
            n_support (int, optional): Number of support. Defaults to 4.

        """
        super().__init__()

        # data config
        self.data_dir = Path(data_dir).resolve()
        ds_info = {
            # train: (Jan-01-2007 to Jan-01-2015)
            # val: (Jan-01-2015 to Jan-01-2016)
            # test: (Jan-01-2016 to Jan-01-2017)
            'kdd17': {
                'path': self.data_dir / 'kdd17/price_long_50',
                'date': self.data_dir / 'kdd17/trading_dates.csv',
                'universe': self.data_dir / 'kdd17/stock_universe.json', 
                'start_date': '2007-01-01',
                'train_date': '2015-01-01', 
                'valid_date': '2016-01-01', 
                'test_date': '2017-01-01',
            },
            # train: (Jan-01-2014 to Aug-01-2015)
            # val: (Aug-01-2015 to Oct-01-2015)
            # test: (Oct-01-2015 to Jan-01-2016)
            'acl18': {
                'path': self.data_dir / 'stocknet-dataset/price/raw',
                'date': self.data_dir / 'stocknet-dataset/price/trading_dates.csv',
                'universe': self.data_dir / 'stocknet-dataset/stock_universe.json',
                'start_date': '2014-01-01',
                'train_date': '2015-08-01', 
                'valid_date': '2015-10-01', 
                'test_date': '2016-01-01',
            }
        }
        ds_config = ds_info[dtype]
        
        self.meta_type = meta_type
        if meta_type in ['test-time', 'test-stock', 'test-mix']: 
            self.window_sizes = test_window_size
        else:
            self.window_sizes = window_sizes
        self.batch_size = batch_size
        self.n_support = n_support
        self.n_query = n_query

        # get data
        self.data = {}
        self.all_tasks = {}
        ps = list((ds_config['path']).glob('*.csv'))
        with ds_config['universe'].open('r') as file:
            universe_dict = json.load(file)
        
        # meta_type: train / valid1: valid-time, valid2: valid-stock, valid3: valid-mix / test1, test2, test3
        if meta_type in ['train', 'valid-time', 'test-time']:
            universe = universe_dict['train']
        elif meta_type in ['valid-stock', 'valid-mix']:
            universe = universe_dict['valid']
        elif meta_type in ['test-stock', 'test-mix']:
            universe = universe_dict['test']
        else:
            raise KeyError('Error argument `meta_type`, should be in (train, valid-time, valid-stock, valid-mix, test-time, test-stock, test-mix)')

        if meta_type in ['train', 'valid-stock', 'test-stock']:
            date1 = ds_config['start_date']
            date2 = ds_config['train_date']
        elif meta_type in ['valid-time', 'valid-mix']:
            date1 = ds_config['train_date']
            date2 = ds_config['valid_date']
        elif meta_type in ['test-time', 'test-mix']:
            date1 = ds_config['valid_date']
            date2 = ds_config['test_date']
        else:
            raise KeyError('Error argument `meta_type`, should be in (train, valid-time, valid-stock, valid-mix, test-time, test-stock, test-mix)')

        iterator = [p for p in ps if p.name.strip('.csv') in universe]
        for p in tqdm(iterator, total=len(iterator), desc=f'Processing data for {self.meta_type}'):    
            stock_symbol = p.name.rstrip('.csv')
            df_single = self.load_single_stock(p)
            cond = df_single['date'].between(date1, date2)
            df_single = df_single.loc[cond].reset_index(drop=True)
            
            self.data[stock_symbol] = df_single


        self.n_stocks = len(universe)


    def load_single_stock(self, p: Path | str):
        def longterm_trend(x: pd.Series, k:int):
            return (x.rolling(k).sum().div(k*x) - 1) * 100

        df = pd.read_csv(p)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        if 'Unnamed' in df.columns:
            df.drop(columns=df.columns[7], inplace=True)
        if 'Original_Open' in df.columns:
            df.rename(columns={'Original_Open': 'Open', 'Open': 'Adj Open'}, inplace=True)

        # Open, High, Low
        z1 = (df.loc[:, ['Open', 'High', 'Low']].div(df['Close'], axis=0) - 1).rename(
            columns={'Open': 'open', 'High': 'high', 'Low': 'low'}) * 100
        # Close
        z2 = df[['Close']].pct_change().rename(columns={'Close': 'close'}) * 100
        # Adj Close
        z3 = df[['Adj Close']].pct_change().rename(columns={'Adj Close': 'adj_close'}) * 100

        z4 = []
        for k in [5, 10, 15, 20, 25, 30]:
            z4.append(df[['Adj Close']].apply(longterm_trend, k=k).rename(columns={'Adj Close': f'zd{k}'}))

        df_pct = pd.concat([df['Date'], z1, z2, z3] + z4, axis=1).rename(columns={'Date': 'date'})
        cols_max = df_pct.columns[df_pct.isnull().sum() == df_pct.isnull().sum().max()]
        df_pct = df_pct.loc[~df_pct[cols_max].isnull().values, :]

        return df_pct

    def sliding_window_idx(self, df_single, window_size):
    
        if len(df_single) >= window_size:
            x_spt_task = []
            y_spt_task = []
            x_qry_task = []
            y_qry_task = []

            for i in range(len(df_single)-window_size-self.n_support-self.n_query+1):
                x_spt = []
                y_spt = []
                x_qry = []
                y_qry = []

                for j in range(self.n_support+self.n_query):
                    if j < self.n_support:
                        spt_idx = [idx for idx in range(i+j, i+j+window_size)]
                        x_spt.append(spt_idx)
                        y_spt.append(i+j+window_size)

                    else:
                        qry_idx = [idx for idx in range(i+j, i+j+window_size)]
                        x_qry.append(qry_idx)
                        y_qry.append(i+j+window_size)

                x_spt_task.append(x_spt)
                y_spt_task.append(y_spt)
                x_qry_task.append(x_qry)
                y_qry_task.append(y_qry)
            return x_spt_task, y_spt_task, x_qry_task, y_qry_task
    
    def generate_data(self,df_single, x_spt_task, y_spt_task, x_qry_task, y_qry_task):
        num_task = len(x_spt_task)
        support_task = []
        support_labels = []
        query_task = []
        query_labels = []
        for i in range(num_task):
            support_inputs = []
            query_inputs = []
            for j in range(self.n_support):
                support_inputs.append(df_single.iloc[x_spt_task[i][j]].to_numpy()[:, 1:].astype(np.float64))

            support_labels.append(df_single['close'].iloc[y_spt_task[i]].to_numpy().astype(np.float64))
            support_task.append(np.array(support_inputs))
            for k in range(self.n_query):
                query_inputs.append(df_single.iloc[x_qry_task[i][k]].to_numpy()[:, 1:].astype(np.float64))
            query_labels.append(df_single['close'].iloc[y_qry_task[i]].to_numpy().astype(np.float64))
            query_task.append(np.array(query_inputs))   

        return support_task, support_labels, query_task, query_labels
    
    @property
    def symbols(self):
        return list(self.data.keys())
    
    def generate_all_task(self):
        all_tasks = dict()
        for window in self.window_sizes:
            all_tasks[window] = self.generate_all_task_per_window(window)
        self.all_tasks = all_tasks

    def generate_all_task_per_window(self,window_size):
        
        all_window_tasks = dict(
                query = [],
                query_labels = [],
                support = [],
                support_labels = [],
            )
        for symbol in self.symbols:
            df = self.data[symbol]
            x_spt_task, y_spt_task, x_qry_task, y_qry_task = self.sliding_window_idx(df, window_size)
            support_inputs, support_labels, query_inputs, query_labels = self.generate_data(df, x_spt_task, y_spt_task, x_qry_task, y_qry_task)
            all_window_tasks['query'].extend(query_inputs)
            all_window_tasks['query_labels'].extend(query_labels)
            all_window_tasks['support'].extend(support_inputs)
            all_window_tasks['support_labels'].extend(support_labels)
        
        all_window_tasks['query'] = np.array(all_window_tasks['query'])
        all_window_tasks['query_labels'] = np.array(all_window_tasks['query_labels'])
        all_window_tasks['support'] = np.array(all_window_tasks['support'])
        all_window_tasks['support_labels'] = np.array(all_window_tasks['support_labels'])
        return all_window_tasks
    
    def generate_batch_task(self, all_tasks):
        batch_tasks = dict(
                query = [],
                query_labels = [],
                support = [],
                support_labels = [],
            )

        
        if len(self.window_sizes) > 1:
            window_size = random.choice(self.window_sizes)
        else:
            window_size = self.window_sizes[0]
               
        num_task = len(all_tasks[window_size]['query'])
        batch_idx = random.sample(list(range(num_task)), self.batch_size)
        batch_tasks['query'] = all_tasks[window_size]['query'][batch_idx]
        batch_tasks['query_labels'] = all_tasks[window_size]['query_labels'][batch_idx]
        batch_tasks['support'] = all_tasks[window_size]['support'][batch_idx]
        batch_tasks['support_labels'] = all_tasks[window_size]['support_labels'][batch_idx]

        return batch_tasks, window_size
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import requests
from sklearn import preprocessing

key = 'realKnownCause/cpu_utilization_asg_misconfiguration.csv'


class NabDataset(Dataset):
    def __init__(self, data_settings):
        self.train = data_settings.train
        self.ano_span_count = 0 # self.ano_span_count is updated when read_data() function is is called
        
        df_x, df_y = self.read_data(data_file = data_settings.data_file, 
                       label_file = data_settings.label_file,
                       key = data_settings.key)
        
        #select and standardize data
        df_x = df_x[['value']]
        df_x = self.normalize(df_x)
        df_x.columns = ['value']
        
        # important parameters
        #self.window_length = int(len(df_x)*0.1/self.ano_span_count)
        self.window_length = 60
        if data_settings.train:
            self.stride = 1
        else:
            self.stride = self.window_length
        
        self.n_feature = len(df_x.columns)
        
        # x, y data
        x = df_x
        y = df_y
        
        # adapt the datasets for the sequence data shape
        x, y = self.unroll(x, y)
        
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(np.array([1 if sum(y_i) > 0 else 0 for y_i in y])).float()
        
        
        self.data_len = x.shape[0]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        
        return x, y
        

    
    # create sequences 
    def unroll(self, data, labels):
        un_data = []
        un_labels = []
        seq_len = int(self.window_length)
        stride = int(self.stride)
        
        idx = 0
        while(idx < len(data) - seq_len):
            un_data.append(data.iloc[idx:idx+seq_len].values)
            un_labels.append(labels.iloc[idx:idx+seq_len].values)
            idx += stride
        return np.array(un_data), np.array(un_labels)
    
    def read_data(self, data_file=None, label_file=None, key=None):
        url = "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json"
        response = requests.get(url)
        data = json.loads(response.text)

        # Check if the provided key exists in the JSON data
        if key not in data:
            raise KeyError("Key '{}' not found in the JSON data.".format(key))
        ano_spans = data[key]
        self.ano_span_count = len(ano_spans)
        df_x = pd.read_csv(data_file)
        df_x, df_y = self.assign_ano(ano_spans, df_x)
            
        return df_x, df_y
    
    def assign_ano(self, ano_spans=None, df_x=None):
        df_x['timestamp'] = pd.to_datetime(df_x['timestamp'])
        y = np.zeros(len(df_x))
        for ano_span in ano_spans:
            ano_start = pd.to_datetime(ano_span[0])
            ano_end = pd.to_datetime(ano_span[1])
            for idx in df_x.index:
                if df_x.loc[idx, 'timestamp'] >= ano_start and df_x.loc[idx, 'timestamp'] <= ano_end:
                    y[idx] = 1.0
        return df_x, pd.DataFrame(y)
    
    def normalize(self, df_x=None):
        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(df_x)
        df_x = pd.DataFrame(np_scaled)
        return df_x
    
    
# settings for data loader
class DataSettings:
    def __init__(self):
        # dataset name
        end_name = 'cpu_utilization_asg_misconfiguration.csv' 

        # location of datasets
        self.data_file = end_name # Just the dataset name
        self.key = 'realKnownCause/'+end_name # This key is used for reading anomaly labels
        self.label_file = 'combined_windows.json'
        self.train = True
        self.window_length = 60

    


def main():
    data_settings = DataSettings()
    # define dataset object and data loader object for NAB dataset
    dataset = NabDataset(data_settings=data_settings)
    print(dataset.x.shape, dataset.y.shape) # check the dataset shape
    
if __name__=='__main__':
    main()
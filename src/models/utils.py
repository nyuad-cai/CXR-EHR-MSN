import os
import torch
import random

import numpy as np
import pandas as pd
import torch.nn as nn

from typing import  Dict
from sklearn.metrics import roc_auc_score, average_precision_score



# general utilities

def count_parameters(model:nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_weights(weights: Dict[str,torch.Tensor]) -> Dict[str,torch.Tensor]:
    
    for k in list(weights.keys()):

        if k.startswith('backbone.'):
            
            if k.startswith('backbone.') and not k.startswith('backbone.heads'):
                
                weights[k[len("backbone."):]] = weights[k]
                
        del weights[k] 
    return weights



def evaluate_new(df):
    yt =np.array([np.array(x) for x in df['y_truth'].values])
    yp =np.array([np.array(x) for x in df['y_pred'].values])
    auroc = roc_auc_score(yt, yp)
    auprc = average_precision_score(yt, yp)
    return auprc, auroc

def bootstraping_eval(df, num_iter):
    """This function samples from the testing dataset to generate a list of performance metrics using bootstraping method"""
    auroc_list = []
    auprc_list = []
    for _ in range(num_iter):
        sample = df.sample(frac=1, replace=True)
        auprc, auroc = evaluate_new(sample)
        auroc_list.append(auroc)
        auprc_list.append(auprc)
    return auprc_list, auroc_list

def computing_confidence_intervals(list_,true_value):

    """This function calcualts the 95% Confidence Intervals"""
    delta = (true_value - list_)
    list(np.sort(delta))
    delta_lower = np.percentile(delta, 97.5)
    delta_upper = np.percentile(delta, 2.5)

    upper = true_value - delta_upper
    lower = true_value - delta_lower
    return (upper,lower)

def get_model_performance(df,summary_path):
    test_auprc, test_auroc = evaluate_new(df)
    auprc_list, auroc_list = bootstraping_eval(df, num_iter=100)
    upper_auprc, lower_auprc = computing_confidence_intervals(auprc_list, test_auprc)
    upper_auroc, lower_auroc = computing_confidence_intervals(auroc_list, test_auroc)
    print("\n--------------")
    text_a=str(f" {round(test_auroc, 3)} ({round(lower_auroc, 3)}, {round(upper_auroc, 3)})")
    text_b=str(f" {round(test_auprc, 3)} ({round(lower_auprc, 3)}, {round(upper_auprc, 3)})")
    print(text_a)
    print(text_b)
    summary = {'test_auroc':np.round(test_auroc,3),
               'lower_auroc':np.round(lower_auroc,3),
               'upper_auroc':np.round(upper_auroc,3),
               'test_auprc':np.round(test_auprc,3),
               'lower_auprc':np.round(lower_auprc,3),
               'upper_auprc':np.round(upper_auprc,3),
               'auroc_text':text_a,
               'auprc_text':text_b}
    
    final = pd.DataFrame(summary,index=[0])
    final.to_csv(os.path.join(summary_path,'summary.csv'),index=False)

    return (test_auprc, upper_auprc, lower_auprc), (test_auroc, upper_auroc, lower_auroc), (text_a,text_b)


def mask_tensor(tensor: torch.tensor):
    rows,cols = tensor.size()[0], tensor.size()[1]
    for i in range(rows):
        rand_idx = random.randint(0,cols-1) 
        flag = True
        while flag:
            if tensor[i,rand_idx] == 0.0:
                rand_idx = random.randint(0,cols-1)
            else:
                flag = False
        tensor[i,rand_idx] = 0.0
    return tensor
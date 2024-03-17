import pandas as pd
import torch
import numpy as np
from sklearn import preprocessing
from scapy.all import *

def getlabel(path):
    # label 가져오기
    le = preprocessing.LabelEncoder()
    label = pd.read_csv(path).Label
    
    le.fit(label)
    
    y=le.transform(label)

    unique_labels = np.unique(y)
    
    i=0
    for name in le.inverse_transform(unique_labels):
       print(f'{i} is',name) 
       i+=1
    
    return y

def getlabeldict(path):
    le = preprocessing.LabelEncoder()
    label = pd.read_csv(path).Label
    
    le.fit(label)
    
    y=le.transform(label)
    
    unique_label = np.unique(y)
    
    dict = {}
    
    for i, name in enumerate(le.inverse_transform(unique_label)):
        dict[name]=i
    
    return dict


if __name__ == '__main__':
    y=getlabel('metadata.csv')
    
    print(y)
    f = pd.read_csv('metadata.csv')
    f['Label']=y
    print(f.head)
    f.to_csv('et_bert_as.tsv',sep='\t')

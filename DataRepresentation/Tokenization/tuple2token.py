import numpy as np
import pandas as pd
import torch
from scapy.all import *

MAX_LENGTH = 1200

def tuple2token(csv:string, dim:int=1,file_path:string=""):
    
    max_length = 0
    
    metadata = pd.read_csv(csv)
    
    train_data = None
    # 1. max length 찾아주기
    
    for i in metadata.index:
        
        file_name = metadata.loc[i].File
        label = metadata.loc[i].Label
        
        pcap = rdpcap(file_path+'/'+label+'/'+file_name)
        length = len(pcap)
        
        if max_length < length < MAX_LENGTH:
            max_length = length
        
        elif length >= MAX_LENGTH:
            max_length = MAX_LENGTH
            break

    print("Max length is",max_length)
    
    for i in metadata.index:
        
        data = torch.full(size=(max_length,),fill_value=0,dtype=torch.int)
        
        file_name = metadata.loc[i].File
        label = metadata.loc[i].Label
        
        pcap = rdpcap(file_path+'/'+label+'/'+file_name)
        
        result = pcap2token(pcap)
        
        if(result is None):
            continue
        data[:result.size(0)] = result
        if train_data is None:
            train_data = data
        else:
            train_data = torch.concat([train_data,data])
        if i%1000==0:
            print(f'{i}번째까지 완료')
            print(train_data.shape)
            
    if dim == 1:
        train_data = train_data.reshape([-1,max_length])
    elif dim==2:
        train_data = train_data.reshape([-1,1,max_length])
        
    print(train_data.shape)
    return train_data
            
def pcap2token(pcap):
    # 방향을 알아내기 위해 src, dst ip 가져오기
    # udp packet이라 뭐가 server고 client인지 확실히 알 수 없음
    # 그냥.. 가장 첫 패킷을 기준으로 src, dst를 정해서 그걸로 정하자
        
    if len(pcap) == 0 :
        return 
    
    result=[]
    start_ip = pcap[0][IP].src
    
    for i in range(0,min(MAX_LENGTH,len(pcap))):
        
        packet = pcap[i]
        src_ip = packet[IP].src
        if src_ip == start_ip: dir = 1
        else: dir = 2
            
        token = len(packet) * dir
        
        result.append(token)
    
    return torch.tensor(result)
        
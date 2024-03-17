'''
protocol, length, direction을 모두 사용한다
3차원 tuple을 딕셔너리에 저장해서 
그 딕셔너리를 가지고 각 패킷을 토큰화
'''

import numpy as np
import pandas as pd
import torch
from scapy.all import *

MAX_LENGTH = 100

class triple2token:
    
    dictionary = {}
    
    def __init__(self,csv:string, dim:int=1,file_path:string=""):
        self.metadata = pd.read_csv(csv)
        self.dim = dim
        self.file_path = file_path
        
        
    def triple2token(self):
        
        max_length = 0
        
        train_data = None
        
        y = []
        
        # 1. max length 찾아주기
        for i in self.metadata.index:
            
            file_name = self.metadata.loc[i].File
            label = self.metadata.loc[i].Label
            
            pcap = rdpcap(self.file_path+'/'+label+'/'+file_name)
            length = len(pcap)
            
            if max_length < length < MAX_LENGTH:
                max_length = length
            
            elif length >= MAX_LENGTH:
                max_length = MAX_LENGTH
                break
        print("Max length is",max_length)
        
        # 2. Dictionary 만들면서 토큰화 진행
        
        for i in self.metadata.index:
            
            data = torch.full(size=(max_length,),fill_value=0,dtype=torch.int)
            
            file_name = self.metadata.loc[i].File
            label = self.metadata.loc[i].Label
            
            pcap = rdpcap(self.file_path+'/'+label+'/'+file_name)
            
            # result = self.pcap2token(pcap)
            if len(pcap)==0:
                continue
            
            y.append(label)
        #     data[:result.size(0)] = result
            
        #     if train_data is None:
        #         train_data = data
        #     else:
        #         train_data = torch.concat([train_data,data])
        #     if i%1000==0:
        #         print(f'{i}번째까지 완료')
        #         print(train_data.shape)
            
        # if self.dim == 1:
        #     train_data = train_data.reshape([-1,max_length])
        # elif self.dim==2:
        #     train_data = train_data.reshape([-1,1,max_length])
            
        return train_data, y

          
    def pcap2token(self, pcap):
            
        result=[]
        if(len(pcap)==0):
            return 
        
        start_ip = pcap[0][IP].src
        
        for i in range(0,min(MAX_LENGTH,len(pcap))):
            
            packet = pcap[i]
            src_ip = packet[IP].src
            
            protocol = 1 if packet.haslayer(UDP) else( 2 if packet.haslayer(TCP) else 3)
            if src_ip == start_ip: dir = 1
            else: dir = 2
            
            token = (protocol, dir, len(packet))
            
            if token not in self.dictionary:
                value = len(self.dictionary)+1
                self.dictionary[token] = value
            else:
                value = self.dictionary[token]
                
            result.append(value)
            
        return torch.tensor(result)
            
if __name__ == '__main__':
    t = triple2token('sc_metadata.csv',1,'../datasets/sc')
    t.triple2token()
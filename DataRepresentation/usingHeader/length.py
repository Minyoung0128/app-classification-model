import numpy as np
import pandas as pd
from scapy.all import *
import torch 

SIZE = 9
MAX_LENGTH = 100

def length(csv:string,dim:int,file_path:string=""):
    csv=pd.read_csv(csv)
    
    max_length=0
    
    for i in csv.index:
        # file name에 맞는 pcap 파일 가져오기
        name = csv._get_value(i, 'File')
        label = csv._get_value(i, 'Label')
        pcap = rdpcap(file_path+'/'+label+'/'+name)
        num_packet = len(pcap)
        
        # max length 업데이트
        max_length=max(max_length,num_packet)
        
        # 100 넘어가면 그냥 max length 100으로 잡고 끝내기 
        if(max_length>MAX_LENGTH):
            max_length = MAX_LENGTH
            break
        
    print("max length is",max_length)
    # max length를 기준으로 tensor 잡아주기
    
    train_data = None
    for i in csv.index:
        
        data = torch.full(size=(max_length,),fill_value=-1,dtype=torch.int)
        
        name = csv._get_value(i, 'File')
        label = csv._get_value(i, 'Label')
        
        file_name = file_path+"/"+label+'/'+name
        result = pcap2vector(file_name,max_length=max_length) 
        
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
    
def pcap2vector(file_name,max_length):
    
    pcap = rdpcap(file_name)
    
    result = []
    for i in range(0,min(len(pcap),max_length)):
        
        packet = pcap[i]
        
        length = len(packet)
        
        result.append(length)

    return torch.tensor(result,dtype=torch.float)
        


if __name__ =='__main__':
    
    result=length('a.csv',1,"cstnet-tls")
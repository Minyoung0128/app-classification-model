import numpy as np
import torch
import pandas as pd
from scapy.all import *

def direction(csv:string, dim:int=1, file_path:string=""):
    
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
        if(max_length>100):
            max_length = 100
            break
        
    print("max length is",max_length)
    # max length를 기준으로 tensor 잡아주기
    
    data = []
    
    for i in csv.index:
        
        name = csv._get_value(i, 'File')
        label = csv._get_value(i, 'Label')
        file_name = file_path+"/"+label+'/'+name
        result = capture_packet(file_name,max_length=max_length)
        
        if result is not None :
            data.append(result)
        
    return np.stack(data)
            
            
def capture_packet(file_name, max_length):
    
    
    pcap = rdpcap(file_name)
    
    first_packet = pcap[0]    
    server_ip = first_packet[IP].src
    client_ip = first_packet[IP].dst
    
    data = np.full((max_length,),fill_value=-1)
    
    for i in range(0,min(100,len(pcap))):
        p = pcap[i]
        
        isuplink = True if p[IP].src == client_ip else False
        
        p_len = 0 if isuplink else len(p)
    
    data = np.full((max_length,),fill_value=0)
    
    for i in range(0,min(100,len(pcap))):
        p = pcap[i]
        
        isuplink = True if p[IP].src == client_ip else False
        
        # uplink면 1 downlink면 -1 -> 0으로.. padding을 줘야될듯?
        
        p_len = 1 if isuplink else -1
    
        data[i] = p_len

    return data


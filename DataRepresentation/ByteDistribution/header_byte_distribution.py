import numpy as np
import pandas as pd
import torch
import byte
from scapy.all import *

def header_byte_distribution(csv:string, dim=1,file_path:string=""):
    
    csv=pd.read_csv(csv)
    
    train_data = None
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
            print("Max Length is",max_length)
            break
        print("Max Length is",max_length)
    # max length를 기준으로 tensor 잡아주기
    
    for i in csv.index:
        
        data = torch.full(size=(max_length*8,),fill_value=-1,dtype=torch.float)
        
        name = csv._get_value(i, 'File')
        label = csv._get_value(i, 'Label')
        
        file_name = file_path+label+'/'+name
        result = packet_distribution(file_name,max_length=max_length) 
        
        data[:result.size(0)] = result
        
        if train_data is None:
            train_data = data
        else:
            train_data = torch.concat([train_data,data])

        if i%1000==0:
            print(f'{i}번째까지 완료')
            print(train_data.shape)
    if dim == 1:
        train_data = train_data.reshape([-1,max_length*8])
    elif dim==2:
        train_data = train_data.reshape([-1,8,max_length])

    return train_data

def packet_distribution(file_name:string,max_length:int):
    
    # file_name에서 packet을 하나씩 가져오고
    # header file만 가져오기
    previous_time = 0
    
    packets = rdpcap(file_name)
    
    result = []
    
    for i in range(0,min(max_length,len(packets))):
        packet = packets[i]
        if previous_time==0:
            interval = 0
            previous_time = packet.time
        else :
            interval = packet.time - previous_time
            previous_time = packet.time
            
        protocol, header_byte = get_header_byte(packet)
        
        bd = get_byte_distribution(header_byte)
        
        result = result + [protocol, interval] + bd

    return torch.tensor(result,dtype=torch.float)

def get_header_byte(packet):
    
    ip_byte = bytes(packet.getlayer(IP))[:(packet[IP].ihl*4)]
    
    if packet.haslayer(UDP):
        protocol = 0
        # UDP
        udp_byte = bytes(packet[UDP])[:8]
        result = ip_byte + udp_byte
    if packet.haslayer(TCP):
        protocol=1
        # TCP
        tcp_header_length = len(packet[IP])-len(packet[TCP])
        tcp_byte = bytes(packet[TCP])[:tcp_header_length]
        
        result = ip_byte + tcp_byte
    
    else:
        print("no layer")
    return protocol,result

def get_byte_distribution(bytes):
    arr = np.zeros(256)
    
    for b in bytes:
        arr[b]+=1
        
    
    std_byte_freq = arr.std()
    ent_byte_freq = byte.entropy(arr)
    
    visible = arr[0x20:0x7F]
    invisible = np.append(arr[0x00:0x20], arr[0x7F])
    extended= arr[0x80:]
    
    ent_visible = byte.entropy(visible)
    ent_invisible =byte.entropy(invisible)
    ent_extended = byte.entropy(extended)
    
    return [std_byte_freq,ent_byte_freq,ent_visible,ent_invisible,ent_extended]

if __name__ == '__main__':
    result=header_byte_distribution('a.csv',1,"cstnet-tls")
    
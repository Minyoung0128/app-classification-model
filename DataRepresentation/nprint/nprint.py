import pandas as pd
import byte2torch as t
import torch
import numpy as np
import time
from sklearn import preprocessing
from scapy.all import *

# 1. csv 파일에서 이름 가져오기 
# 2. 파일 이름 가지고 pcap 파일 읽어서 tensor로 변환 라벨 붙여주기

MAX_LENGTH = 900
def nprint(csv:string, dim:int=1, file_path:string=""):
    start = time.time()

    # CONSTANT

    SIZE = 1088

    metadata = pd.read_csv(csv)

    file_name =metadata.File
    label = metadata.Label
    max_length = 0

    train_data = []
    
    for i in metadata.index:
        # file name에 맞는 pcap 파일 가져오기
        name = metadata._get_value(i, 'File')
        label = metadata._get_value(i, 'Label')
        pcap = rdpcap(file_path+'/'+label+'/'+name)
        num_packet = len(pcap)
        max_length=max(max_length,num_packet)
        # if(num_packet>max_length):
        #     print(name, label,"에서 업데이트, size : ",num_packet)
        # max length 업데이트
        if(max_length>MAX_LENGTH):
            max_length = MAX_LENGTH
            break
        print("Max Length is",max_length)
        
    print("Packet Max Length is",max_length)

    # max length에 맞춰서 그만큼의 packet을 가지는 data 생성 
    for i in metadata.index:
        
        name = metadata._get_value(i, 'File')
        label = metadata._get_value(i, 'Label')
        pcap = rdpcap(file_path+'/'+label+'/'+name)
        num_packet = len(pcap)
            
        x = torch.tensor(np.full(shape=(max_length,SIZE),fill_value=-1),dtype=torch.int8)
        
        # pcap 파일 돌면서 packet data 정제해주기 
        
        for j in range(min(num_packet,max_length)):
            packet = pcap[j]
            
            if(packet.haslayer(UDP)):
                udp = t.UDP2torch(packet)
                
                if(x is None):
                    x = udp
                else:
                    x[j]=udp
            
            if(packet.haslayer(TCP)):
                tcptensor=t.TCP2torch(packet)
                
                if(x is None):
                    x = tcptensor
                else:
                    x[j]=tcptensor
        
            if(packet.haslayer(ICMP)):
                icmp=t.ICMP2torch(packet)
                
                if(x is None):
                    x = icmp
                else:
                    x[j]=icmp
        
        if (i%1000==0):
            print(f'{i}번재 까지 끝')        
        train_data.append(x.unsqueeze(0))

    train_data = torch.cat(train_data, dim=0)
    if(dim==1):
        train_data=train_data.reshape(train_data.size(0),-1)
    print("Final data shape is" ,train_data.shape)

    finish = time.time()

    print("총 실행 시간 ", finish-start)    
    
    return train_data


def byte2bit(input):
    # byte 파일을 받아서 bit 형 tensor로 변환해주는 함수
    return torch.tensor([int(bit) for byte in input for bit in f'{byte:08b}'])

def UDP2torch(packet):
        # UDP packet을 받고, 이를 파싱해서 데이터를 만들어주는 함수 
        
        if packet.haslayer(UDP):

            # tcp, ICMP header 만들어주기
            tcp_header = torch.tensor(np.full(480,-1))
            icmp_header = torch.tensor(np.full(64,-1))
            
            # ip header
            ipheader=torch.tensor(np.full(480,-1))
            
            if packet.haslayer(IP):
                # ipv4 header
                ippart=byte2bit(bytes(packet.getlayer(IP))[:(packet[IP].ihl*4)])
            
            if packet.haslayer(IPv6):
                ippart = byte2bit(bytes(packet.getlayer(IPv6))[:40])
            
            ipheader[:ippart.size(dim=0)] = ippart
            
            # UDP header
            
            udpheader=byte2bit(bytes(packet[UDP])[:8])
            
            final_tensor= torch.concat((ipheader, tcp_header,udpheader,icmp_header),dim=0)
           
            return final_tensor


def TCP2torch(packet):
        # UDP packet을 받고, 이를 파싱해서 데이터를 만들어주는 함수 
    
        # udp, ICMP header 만들어주기
        udp_header = torch.tensor(np.full(64,-1))
        icmp_header = torch.tensor(np.full(64,-1))
        
        # ip header
        ip_header=torch.tensor(np.full(480,-1))
        
        if packet.haslayer(IP):
            # ipv4 header
            ippart=byte2bit(bytes(packet.getlayer(IP))[:(packet[IP].ihl*4)])
        
        if packet.haslayer(IPv6):
            ippart = byte2bit(bytes(packet.getlayer(IPv6))[:40])
        
        ip_header[:ippart.size(dim=0)] = ippart
        
        # TCP header
        
        tcp_header = torch.tensor(np.full(480,-1))
        tcp_header_length = len(packet[IP])-len(packet[TCP])
        tcp_real=byte2bit(bytes(packet[TCP])[:tcp_header_length])
        
        
        tcp_header[:tcp_real.size(dim=0)] = tcp_real
        
        final_tensor= torch.concat((ip_header, tcp_header,udp_header,icmp_header),dim=0)
        
        return final_tensor
    
def ICMP2torch(packet):
    
    # udp, ICMP header 만들어주기
    udp_header = torch.tensor(np.full(64,-1))
    tcp_header = torch.tensor(np.full(480,-1))
    
    # ip header
    ip_header=torch.tensor(np.full(480,-1))
    
    if packet.haslayer(IP):
        # ipv4 header
        ippart=byte2bit(bytes(packet.getlayer(IP))[:(packet[IP].ihl*4)])
    
    if packet.haslayer(IPv6):
        ippart = byte2bit(bytes(packet.getlayer(IPv6))[:40])
    
    ip_header[:ippart.size(dim=0)] = ippart
    
    # ICMP header
    
    icmp_header = byte2bit(bytes(packet[ICMP])[:8])
    final_tensor= torch.concat((ip_header, tcp_header,udp_header,icmp_header),dim=0)
    
    return final_tensor

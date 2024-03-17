'''
하나의 패킷을 토큰화 해서 표현 

1. Header value만 뽑아오기
2. Header value의 bytes를 n-gram을 활용해 토큰화
3. Hedaer value의 크기를 통일
3-1. ip header = 20 byte
3-2. protocol header = 40 byte
'''

import numpy as np
import pandas as pd
import torch
from scapy.all import *


class tokenizer:
    
    token_dict = {'[CLS]':0,'[SEP]':1,'[PAD]' :2,'[MASK]':3}
    
    def __init__(self, csv, path, ip_length, protocol_length, packet_length):
        self.ip_length=ip_length
        self.protocol_length=protocol_length
        self.packet_length=packet_length
        self.metadata = pd.read_csv(csv)
        self.file_path = path
        
    def main(self):
        
        token_data=[]
        len_dir_data=[]
        position_data =[]  
            
        for i in self.metadata.index:
            
            file_name = self.metadata.loc[i].File
            label = self.metadata.loc[i].Label
            
            pcap = rdpcap(self.file_path+'/'+label+'/'+file_name)
            
            token, len_dir, position = self.Embedding(pcap)
            
            token_data.append(token)
            len_dir_data.append(len_dir)
            position_data.append(position)
            
        return token_data, len_dir_data, position_data

    def Embedding(self, pcap):
        '''
        토큰, 세그먼트, 포지션 임베딩 동시에 진행
        '''
        # src ip, dst ip 가져와서 direction 결정
        server_ip = pcap[0][IP].src
        
        token = []
        len_dir = []
        position = []
        
        for index in range(min(self.packet_length,len(pcap))):
            packet = pcap[index]
            
            token_embed = self.token_Embedding(packet)
            
            dir = 1 if packet[IP].src == server_ip else 2
            
            token.append(token_embed)
            len_dir.append(dir * len(packet))
            position.append(index)
        
        return np.array(token), np.array(len_dir), np.array(position)
        
    def token_Embedding(self, packet):
        # 1. header value만 뽑아오기
        ip_header, tcp_header = self.get_header_value(packet)
        
        header2token = self.bytes2token(ip_header,self.ip_length) + self.bytes2token(tcp_header,self.protocol_length)+[1]
        return header2token

    
    def get_header_value(self, packet):
        ip_header_len=min(self.ip_length,packet[IP].ihl*4)
        ip_byte = bytes(packet.getlayer(IP))[:ip_header_len]
        if packet.haslayer(UDP):
            # UDP
            udp_byte = bytes(packet[UDP])[:8]
            protocol_byte = udp_byte
        elif packet.haslayer(TCP):
            # TCP
            tcp_header_length = len(packet[TCP]) - len(packet[TCP].payload)
            tcp_byte = bytes(packet[TCP])[:tcp_header_length]
            protocol_byte = tcp_byte
        elif packet.haslayer(ICMP):
            icmp_byte = bytes(packet[ICMP])[:8]
            protocol_byte = icmp_byte
        else:
            print("no layer")
        
        return ip_byte, protocol_byte
    
    def bytes2token(self, b, length):
        # length를 정해주면 그만큼 자르거나, padding을 줘서 token화 해줌 
        min_length = min(len(b), length)
        token = []
        byte_string = ''
        
        for byte in b[:min_length]:
            byte_string +=('00'+hex(byte)[2:])[-2:]
            
        for index in range(0,min_length,2):
            sub_string = byte_string[index:index+4]
            if not (sub_string in self.token_dict.keys()):
                value = len(self.token_dict)+1
                self.token_dict[sub_string]=value
            else:
                value = self.token_dict[sub_string]
            token.append(value)
        token= token + [2] * max(0, length - len(token))
        
        return token
    
    

if __name__ == '__main__':
    
    pcap2token = ['[CLS]']
    pcap2len = []
    pcap2pos = []
    
    t = tokenizer('m.csv', 'as', 20, 60, 900)
    token, len_dir, position = t.main()
    
    print(token.shape)
    print(len_dir.shape)
    print(position.shape)
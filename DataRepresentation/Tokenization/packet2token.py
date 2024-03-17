'''
packet의 feature를 이용해 토큰화
> 즉 한 패킷을 여러 개의 토큰으로 표현 
> [sep] [CLS] 토큰을 이용해 구분하면 유의미하게 작동하지 않을까?

'''

import os
import numpy as np
from scapy.all import *

ip_dict = {}
port_dict = {}

class packet2token():
    
    def __init__(self, path, seq_len, port=False):
        self.path = path
        self.seq_len = seq_len
        self.port = port
        self.token_len = 5 if port else 3
        self.ip_dict = {}
        self.port_dict = {}
        self.protocol_dict = {"UDP":1,"TCP":2,"ICMP":3,"UNKNOWN":4}
        
    def train(self):
        
        def get_feature(packet):
            
            protocol = 1 if packet.haslayer(UDP) else( 2 if packet.haslayer(TCP) else 3)
            length = len(packet)
            
            src_ip = packet[IP].src
            dest_ip = packet[IP].dst
            
            if src_ip in self.ip_dict:
                src_token = self.ip_dict[src_ip]
            else : 
                src_token = len(self.ip_dict)+2
                self.ip_dict[src_ip] = src_token
            
            if dest_ip in self.ip_dict:
                dest_token = self.ip_dict[dest_ip]
            else : 
                dest_token = len(self.ip_dict)+2
                self.ip_dict[dest_ip] = dest_token
                
            if self.port : 
                src_port = packet[TCP].sport
                dest_port = packet[TCP].dport
                    
                if src_port in self.port_dict:
                    srcport_token = self.port_dict[src_port]
                else : 
                    srcport_token = len(self.port_dict)+2
                    self.port_dict[src_port] = srcport_token
                
                if dest_port in self.port_dict:
                    destport_token = port_dict[dest_ip]
                else : 
                    destport_token = len(self.port_dict)+2
                    port_dict[dest_port] = destport_token

                return np.array([length, src_token, dest_token, srcport_token, destport_token])
            
            return np.array([length, src_token, dest_token]) 
        
        result = []
        pad_id = 0
        sep_id = 1
        
        for dir in os.listdir(self.path):
            path = os.path.join(self.path, dir)
            print(path)     
            for pcap_path in os.listdir(path):
                 
                pcap = rdpcap(os.path.join(path, pcap_path))
                 
                p2t = []
                 
                for i in range(min(self.seq_len, len(pcap))):
                    packet = pcap[i]
                    feature = get_feature(packet)
                    p2t.extend(feature)
                    # p2t.extend([sep_id])
                    
                if len(p2t)<self.seq_len*self.token_len:
                    p2t.extend([pad_id] * (self.seq_len*self.token_len - len(p2t)))

                result.append(p2t)
                
        return result          
    
    
if __name__ == '__main__':
    p = packet2token('/home/myk/min0/as',500,False)
    data = p.train()
    np.save('data.npy',data)
    
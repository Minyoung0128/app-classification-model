import torch 
import numpy as np
from scapy.all import *


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
    print(icmp_header)
    final_tensor= torch.concat((ip_header, tcp_header,udp_header,icmp_header),dim=0)
    
    return final_tensor

if __name__ == '__main__':
    pcap = rdpcap('icmp.pcapng')
    
    for j in range(0,5):
            packet = pcap[j]
            
            
            if(packet.haslayer(ICMP)):
                tcptensor=ICMP2torch(packet)
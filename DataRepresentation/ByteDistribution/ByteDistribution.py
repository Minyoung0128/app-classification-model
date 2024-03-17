'''
pcap file을 읽어와 패킷을 읽는다.
payload가 없는 패킷은 drop한다.
각 패킷 간의 interval을 측정한다. 

그럼 패킷 1차원으로 쌓은거 + 시간 + 라벨이 한 pcap file의 한 데이터가 됨
'''

import torch 
import numpy as np
from scapy.all import *
import pandas as pd

from collections import Counter
from sklearn import preprocessing

SIZE = 14

# p.time으로 packet의 시간을 가져올 수 있다. 

def entropy(arr):
    total = arr.sum()
    
    if total == 0:
        return 0
    entropy = 0 
    
    for i in arr:
        if i==0:
            continue
        p = i/total
        entropy += p*np.log2(p)
    
    return -entropy 

def count_byte(packet):
    arr = np.zeros(256)
    
    for byte in bytes(packet):
        arr[byte]+=1
    
    return arr

def byte_distribution(csv:string,dim:int,file_path:string=""):
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
        
        data = torch.full(size=(max_length*SIZE,),fill_value=-1)
        
        name = csv._get_value(i, 'File')
        label = csv._get_value(i, 'Label')
        
        file_name = "cstnet-tls/"+label+'/'+name
        result = capture_packet(file_name,max_length=max_length) 
        
        data[:result.size(0)]=result
        
        if train_data is None:
            train_data = data
        else:
            train_data=torch.concat([train_data,data])
            
    if dim==2:
        train_data = train_data.reshape([-1,SIZE*max_length])
        
    return train_data

    
def capture_packet(file_name,max_length):
    
    pcap = rdpcap(file_name)
    
    data = torch.full(size=(max_length,14),fill_value=-1,dtype=float)
    
    for i in range(0,min(len(pcap),max_length)):
        
        packet = pcap[i]
        print(file_name+f'{i}번째')
        byte_dis = packet_distribution(packet)
        
    return byte_dis 
        
    
def packet_distribution(packet):
    previous_time = 0
     # UDP, TCP 프로토콜이 아니거나, payload 가 없는 패킷들 drop 
    if not (packet.haslayer(UDP) | packet.haslayer(TCP)):
        return 
    
    # payload 없는 건 그냥 사용해보장...~
    # if not packet.haslayer(Raw):
    #     return 

    # data[0] : protocol
    # packet[2] payload length
    if packet.haslayer(UDP):
        protocol = 0
    #     payload = len(packet[UDP].payload) 
    else:
        protocol = 1
    #     payload = len(packet[TCP].payload)
    
    payload = len(packet)
    
    # data[1] interval time
    if previous_time==0:
        interval = 0
        previous_time = packet.time
    else :
            interval = packet.time - previous_time
            previous_time = packet.time
    
    arr = count_byte(packet)
    total_num = np.sum(arr)
    # data[3] = avg. byte frequency data[4] = std. byte frequency
    avg_byte_freq = arr.mean()
    std_byte_freq = arr.std()
    ent_byte_freq = entropy(arr)
    
    # data [6] [7]= entropy, proportion of visible byte
    visible = arr[0x20:0x7F]
    prop_visible = np.sum(visible)/total_num
    ent_visible = entropy(visible)
    
    # data[8] data[9]
    invisible = np.append(arr[0x00:0x20], arr[0x7F])
    prop_invisible = np.sum(invisible)/total_num
    ent_invisible = entropy(invisible)
    
    # data[10] data [11]
    extended= arr[0x80:]
    prop_extended = np.sum(extended)/total_num
    ent_extended = entropy(extended)
    
    prop_00 = arr[0x00]/total_num
    prop_ff = arr[0xff]/total_num
    packet_representation=torch.tensor([protocol,interval,payload,avg_byte_freq,std_byte_freq,ent_byte_freq,ent_visible,prop_visible,ent_invisible,prop_invisible,ent_extended,prop_extended,prop_00,prop_ff])

    return packet_representation
    
    

if __name__ == '__main__':
    result=byte_distribution('a.csv',1,"cstnet-tls")
    
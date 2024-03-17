import torch
import torch.nn as nn 
import math

class Embedding(nn.Module):
    
    '''
    Token Embedding, Position Embedding을 동시에 진행할 수 있도록 합치기
    '''
    def __init__(self, token_embed, pos_embed):
        super(Embedding, self).__init__()
        self.token_embed = token_embed
        self.pos_embed = pos_embed
        
    def forward(self, x):
        out = self.token_embed(x)
        out = self.pos_embed(out)
        return out
    

class TokenEmbedding(nn.Module):
    
    def __init__(self, max_len, d_embed):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_len, d_embed)
        self.d_embed = d_embed
        
    def forward(self, x):
        out = self.embedding(x)
        out = out * math.sqrt(self.d_embed)
        
        return out
        
class PositionalEmbedding(nn.Module):
    
    def __init__(self, d_embedding, max_len = 900, device = torch.device("cuda")):
        '''
        max_len : train data에서 한 pcap data가 가지는 최대 길이
        d_embed : 한 packet data를 embedding 할 때의 dimension
        device : 계산할 때 올려야되니까... 굳이 필요한가..?
        
        output
        [batch, seq_len, d_embedding] 모양의 position embedding 결과 tensor
        '''
        super(PositionalEmbedding, self).__init__()
        
        self.max_len = max_len
        self.d_embedding = d_embedding
        
        # 홀수면 cos, 짝수면 sin에 넣어서.. 사이즈를 조절
        encoding = torch.zeros(max_len, d_embedding)
        encoding.requires_grad = False
        
        position = torch.arange(0,max_len,device=device).float().unsqueeze(1) # [max_len, 1] 형태의 tensor
        
        _2k = torch.arange(0, d_embedding, 2, dtype = torch.float, device=device)
        w_k = 1/10000**(_2k/d_embedding).to(device)
        
        encoding[:,0::2] = torch.sin(position/w_k)
        encoding[:,1::2] = torch.cos(position/w_k)
        
        self.encoding = encoding.unsqueeze(0)
        
    def forward(self, x):
        _, seq_len, _ = x.size()
        device = x.device
        self.encoding = self.encoding.to(device)
        return x + self.encoding[:,:seq_len,:]
        
        
        
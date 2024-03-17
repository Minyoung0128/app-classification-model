import torch
import torch.nn.functional as F
import torch.nn as nn 
import copy
import math

class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self, d_model, h ,qkv_fc, out_fc, dr_rate):
        '''
        d_model : 
        h : encoder block의 개수 
        qkv_fc: d_embed * d_model 모양의 fully connected layer
        > deep copy를 통해 각 layer별 fc layer가 다 다른 값을 가지고 있어야 한다.
        '''
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.query_fc = copy.deepcopy(qkv_fc)
        self.key_fc = copy.deepcopy(qkv_fc)
        self.value_fc = copy.deepcopy(qkv_fc)
        self.out_fc = out_fc
        self.dropout = nn.Dropout(dr_rate)
    
    def transform(self, x, fc):
        '''
        x = (batch, n, d_embed) tensor
        fc = (d_embed, d_model)
        multi-layer에서는 n*d_k 모양이 아니고,
        n * d_model (= d_k * h) 모양으로 계산을 해서 이걸 다시 나눠줘야 됨
        그 함수 
        '''
        out = fc(x) # out.shape = [batch, n, d_model]
        batch = out.shape[0]
        out = out.view(batch, -1, self.h, self.d_model//self.h) # out.shape = [batch, n, h, d_k]
        out = out.transpose(1,2) # out.shape = [batch, h, n, d_k]
        return out 
        
    def forward(self, query, key, value, mask = None):
        '''
        input 
        query, key, value = ( batch, n, d_embedding ) tensor
        '''
        query = self.transform(query, self.query_fc) 
        key = self.transform(key, self.key_fc)
        value = self.transform(value, self.value_fc)
        
        out = self.scaled_dot_product_attention(query, key, value, mask) # out shape = [batch, h, n, d_k]
        
        # 이제 out을 다시 [batch, n, d_embedding]으로 만들어 줘야 함 
        batch = out.shape[0]
        out = out.transpose(1,2) # out = [batch, n, h, d_k]
        out = out.contiguous().view(batch, -1, self.d_model) # [batch, n, d_model]
        out = self.out_fc(out) # [batch, n, d_embedding]
        
        return out 
        
        
    def scaled_dot_product_attention(self, query, key, value, mask = None):
    
        '''
        query, key, value = batch_size*n*d_embed 모양의 torch.tensor
        output = n * d_k 모양의 torch.tensor
        '''
        
        # key shape > 마지막 값이 곧 d_k
        d_k = key.shape[-1]
        
        # 1. query과 key를 곱해 attention score를 만들기
        attention_score = torch.matmul(query, key.transpose(-2,-1)) # transpose(-2,-1) : 마지막 두 개의 값을 위치를 바꿔줌 > batch size는 그대로 
        
        # 2. Scaling
        attention_score = attention_score / math.sqrt(d_k)
        
        # 2-1. Mask
        if mask is not None:
            # masked_fill은 pytorch에서 기본 제공하는 함수
            # mask 행렬이 true인 곳을 저 값으로 바꾸는데 ==0을 이용해서 자동으로 boolean 행렬을 mask로 사용
            attention_score = attention_score.masked_fill(mask==0, -1e11) # mask인 부분은 엄청 작은 값으로 바꿔서 의미없게 하기
            
        # 3. softmax
        attention_score = F.softmax(attention_score, dim = -1)
        attention_score = self.dropout(attention_score)
        
        # 4. V와 matrix multiply
        out = torch.matmul(attention_score, value)
        
        return out    

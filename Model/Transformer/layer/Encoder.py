import torch
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np
import copy
import layer.ResidualConnectionLayer as res

class Encoder(nn.Module):
    
    def __init__(self, encoder_block, n_layer):
        # layer 개수만큼 encoder block을 쌓아준다.
        super(Encoder, self).__init__()
        self.layers = []
        
        for _ in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_block))
            
    def forward(self, x, mask):
        out = x
        for layer in self.layers:
            out = layer(out, mask)
        return out
    
class Encoder_Block(nn.Module):
    
    def __init__(self, head_attention, ff, norm, dr_rate):
        super(Encoder_Block, self).__init__()
        
        # 각 layer를 residual connection layer로 감싸줘야함. 
        self.residual1 = res.ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.residual2 = res.ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.head_attention = head_attention
        self.ff = ff
        
    def forward(self,x,mask):
        out = x
        out = self.residual1(out, lambda out: self.head_attention(query=out, key=out, value=out, mask=mask))
        
        out = self.residual2(out, self.ff)
        return out
    


    
        
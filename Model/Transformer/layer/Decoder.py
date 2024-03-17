import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import layer
from layer import *
class Decoder(nn.Module):
    
    def __init__(self, decoder_block, n_layer):
        super(Decoder, self).__init__()
        
        self.n_layer = n_layer
        
        self.layers = nn.ModuleList()
        
        for _ in range(n_layer):
            self.layers.append(copy.deepcopy(decoder_block))
    
    def forward(self, target, encoder_context, target_mask, src_tgt_mask):
        out = target
        for layer in self.layers:
            out = layer(out, encoder_context, target_mask, src_tgt_mask)
            
        return out 
        

class DecoderBlock(nn.Module):
    
    def __init__(self, self_attention, cross_attention, postion_ff, norm, dr_rate):
        super(DecoderBlock, self).__init__()
        
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.position_ff = postion_ff
        
        self.residual1 = layer.ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.residual2 = layer.ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.residual3 = layer.ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        
    def forward(self, target, encoder_context, target_mask, src_tgt_mask):
        out = target
        out = self.residual1(out, lambda out: self.self_attention(query=out, key=out, value=out, mask=target_mask))
        out = self.residual2(out, lambda out: self.cross_attention(query=out, key=encoder_context, value=encoder_context, mask=src_tgt_mask))
        out = self.residual3(out, self.position_ff)
        
        return out  
    

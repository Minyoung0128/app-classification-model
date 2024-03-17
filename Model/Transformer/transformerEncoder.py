import layer
from layer import EmbeddingLayer
from layer import MultiHeadAttentionLayer
from layer import FeedForwardLayer
import layer.Encoder as Encoder
import torch
import torch.nn as nn
from copy import deepcopy

class TransformerEncoder(nn.Module):
    
    def __init__(self, num_index, d_embed, d_model, d_ff, n_head, n_layer, output_size, dr_rate, device):
        
        super(TransformerEncoder, self).__init__()
        
        # 0. Embedding Layer
        tokenEmbedding = EmbeddingLayer.TokenEmbedding(num_index, d_embed)
        posEmbedding = EmbeddingLayer.PositionalEmbedding(d_embed)
        
        self.Embedding = EmbeddingLayer.Embedding(tokenEmbedding, posEmbedding).to(device)
         
        # 1. MultiHeadAttention Layer 만들기
        qkv_fc = nn.Linear(d_embed, d_model)
        out_fc = nn.Linear(d_model, d_embed)
        
        head_attention = MultiHeadAttentionLayer.MultiHeadAttentionLayer(d_model, n_head, qkv_fc, out_fc, dr_rate).to(device)
        
        # 2. FeedForwardLayer
        fc1 = nn.Linear(d_embed, d_ff)
        fc2 = nn.Linear(d_ff, d_embed) 
        
        ff = FeedForwardLayer.FeedForwardLayer(fc1, fc2).to(device)
        
        # 3. Residual Connection의 norm Layer
        norm = nn.LayerNorm(d_embed)
        
        # 4. Encoder Block
        encoder_block = Encoder.Encoder_Block(deepcopy(head_attention),ff = deepcopy(ff), norm = norm, dr_rate = dr_rate).to(device)
        
        # 5. Encoder 
        self.encoder = Encoder.Encoder(deepcopy(encoder_block), n_layer = n_layer).to(device)
        
        # 6. 최종 classifier
        self.classifier = nn.Sequential(
            nn.Linear(57600, 14000),
            nn.ReLU(),
            nn.Linear(14000, output_size)
        )
        
        self.device = device
        
    
    def forward(self, x):
        out = x
        mask = self.src_pad_mask(x,x, 0)
        
        out = self.Embedding(out)
        out = self.encoder(out, mask)
        out = out.reshape(out.shape[0], -1)
        out = self.classifier(out)
        
        return out        
    
    def src_pad_mask(self, query, key, pad_idx=0):
        
        # query, key는 똑같은 matrix라고 가정
        
        query_len = query.shape[0]
        key_len = key.shape[0]
        
        if query_len != key_len :
            raise Exception(f'Query and Key should have same shape : query len = {query_len}, key_len = {key_len}')
        
        pad_mask = (key == pad_idx).unsqueeze(1).unsqueeze(3)
        
        pad_mask.requires_grad = False
        
        return pad_mask

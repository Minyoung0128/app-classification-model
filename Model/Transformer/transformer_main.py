import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Encoder
import Decoder
import layers
from copy import deepcopy

class Transformer(nn.Module):
    
    def __init__ (self, src_embed, target_embed, encoder, decoder, classifier):
        super(Transformer, self).__init__()
        
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        
    def encode(self, src, src_mask):
        
        x = self.src_embed(src)
        return self.encoder(x, src_mask)
    
    def decode(self, target, encoder_out, target_mask, src_target_mask):
        x = self.target_embed(target)
        return self.decoder(x, encoder_out, target_mask, src_target_mask) # decode ouput [batch, seq_len, d_embedding] > 이걸 다시 label로 분류..해줘야 한다...ㅠ
    
    def make_pad_mask(self, query, key, pad_idx = 0):
        
        # self attention 말고 cross attention은 query와 key의 모양이 다를 수 있음
        '''
        input : tokenization만 된 상태의 vector로 들어옴 
        > 여기서 token이 padding인지 아닌지 판단해서 masking을 진행 
        
        query = [batch, query_len]
        key = [batch, key_len]
        pad_idx = [PAD] index를 의미하는 토큰 값, 보통 1 > 나는 0으로 설정..
        
        '''
       
        query_len, key_len = query.shape[1], key.shape[1]
        
        # torch.ne > 같은지 다른지 비교해서 다르면 true, 같으면 false를 반환하는 함수 
        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3) # [batch, 1, query_len,1]
        query_mask = query_mask.repeat(1,1,1, key_len) # key_len만큼 앞에 3개 차원을 복사..? [batch, 1, query_len, key_len]
        
        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2) # [batch, 1, 1, key_len]
        key_mask = key_mask.repeat(1,1,query_len,1) # [batch, 1, query_len, key_len]
        
        mask = query_mask & key_mask # 둘 다 padding인 곳만 거르면 됨
        mask.requires_grad = False # 이러면 이걸 계산할 때 matrix는 상수 취급되서 forward를 거쳐도 gradient가 업데이트되지 않는다.
        
        return mask
    
    def make_tgt_mask(self, tgt):
        pad_mask = self.make_pad_mask(tgt, tgt)
        seq_mask = self.make_subsequent_mask(tgt, tgt)
        mask = pad_mask & seq_mask
        return mask

    def make_src_tgt_mask(self, src, tgt):
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask


    def make_subsequent_mask(self, query, key):
        query_seq_len, key_seq_len = query.size(1), key.size(1)
        
        tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8') # lower triangle without diagonal
        mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)
        return mask
    
    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask

    def forward(self, src, target):
        src_mask = self.make_src_mask(src)
        target_mask = self.make_tgt_mask(target)
        src_target_mask = self.make_src_tgt_mask(src, target)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(target, encoder_out, target_mask, src_target_mask)
        out = self.classifier(decoder_out)
        
        return out
    

def main(src_token_size, target_token_size, device = torch.device("cuda"),dr_rate=0, max_len = 900, d_embed = 100, n_layer = 6, d_model = 100, h=10, d_ff=500):   
    '''
    src_token_size, target_token_size : 각 srcm target을 토큰화한 후 unique한 token의 개수 
    max_len = pcap 내에서 적용할 패킷의 개수 
    d_embed = packet, label embedding dimenstion
    n_layer = block을 몇 개 쌓을 건지의 개수 
    d_model = h * d_embed.. 아닌가 이건 왜 따로 전달되는걸까
    h = encoder block의 개수 ?
    d_ff = feed-forward layer에서 중간에 거치는 그거의 dimension
    '''
    
    # 1. src, target embedding layer
    
    src_token_embed = layers.TokenEmbedding(packet_size= src_token_size, d_embed = d_embed)
    target_token_embed = layers.TokenEmbedding(packet_size= target_token_size, d_embed = d_embed)
    
    # 2. position embedding layer
    pos_embed = layers.PositionalEmbedding(d_embedding = d_embed, max_len=max_len, device = device)
    
    # 3. token embedding + position embedding
    src_embed = layers.TransformerEmbedding(src_token_embed, deepcopy(pos_embed))
    target_embed = layers.TransformerEmbedding(target_token_embed,deepcopy(pos_embed))
    
    # 4. Multi-head Attention Layer
    attention = layers.MultiHeadAttentionLayer(d_model=d_model, h=h, qkv_fc=nn.Linear(d_embed,d_model), out_fc=nn.Linear(d_model, d_embed),dr_rate=dr_rate).to(device)
    
    # 5. Feed-Forward Layer
    ff_layer = layers.FeedForwardBlock(fc1=nn.Linear(d_embed, d_ff), fc2=nn.Linear(d_ff, d_embed)).to(device)

    norm = nn.LayerNorm(d_embed).to(device)
    # 6. encoder block
    encoder_block = Encoder.Encoder_Block(head_attention = deepcopy(attention), ff=deepcopy(ff_layer),norm=deepcopy(norm),dr_rate=dr_rate)
    
    # 7. decoder block
    decoder_block = Decoder.DecoderBlock(self_attention=deepcopy(attention), cross_attention=deepcopy(attention),postion_ff=deepcopy(ff_layer), norm = deepcopy(norm),dr_rate=dr_rate )
    
    # 8. encoder, decoder
    encoder = Encoder.Encoder(encoder_block, n_layer)
    decoder = Decoder.Decoder(decoder_block,n_layer)
    
    # 9. classifier
    classifier = nn.Linear(d_model, target_token_size)
    
    # 9. 최종 transformer
    model = Transformer(src_embed=src_embed, target_embed=target_embed, encoder=encoder, decoder=decoder, classifier=classifier).to(device)
    
    model.device = device
    
    return model


class TransformerEncoderClassifier(nn.Module):
    def __init__(self, src, num_index, embedding_dim, n_head, n_layer, output_size, device="cuda"):
        
        tokenEmedding = layers.TokenEmbedding(num_index, embedding_dim)
        posEmbedding = layers.PositionalEmbedding(num_index, embedding_dim)
        
        self.embedding = layers.TransformerEmbedding(tokenEmedding, posEmbedding)
        
        self.src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        
    
    def encode(self, src, src_mask):
        
        x = self.src_embed(src)
        return self.encoder(x, src_mask)
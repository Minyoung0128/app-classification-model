import torch
import numpy as npy
import torch.nn as nn
    
class GRU(nn.Module):
    
    def __init__ (self, num_embeddings,input_size, hidden_size, output_size,num_layers):
        super().__init__()
        
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout = 0.22)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        _, out1 = self.gru(x)
        out1 = self.fc(out1[-1,:,:])
        return out1

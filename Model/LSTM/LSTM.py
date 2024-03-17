import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from glove import Glove

class LSTM(nn.Module):
    
    def __init__ (self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=4, batch_first=True,bidirectional=True,dropout = 0.22)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _,(out,_) = self.lstm(x)
        # print("out1",out)
        # print("뽑아낸거 ",out[-1,:,:])
        out = self.fc(out[-1,:,:])
        return out
    
class LSTM_withEmbed(nn.Module):
    
    def __init__ (self, num_embeddings,input_size, hidden_size, output_size,num_layers):
        super().__init__()
        
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout = 0.22)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out0,(out1,out2) = self.lstm(x)
        
        # print("out1",out)
        # print("뽑아낸거 ",out[-1,:,:])
        out1 = self.fc(out1[-1,:,:])
        return out1

class LSTM_withGlove(nn.Module):
    
    def __init__ (self, num_embeddings, input_size, hidden_size, output_size, num_layers, Glove_model_path ):
        super().__init__()
        
        self.glove = Glove.load(Glove_model_path)
        pretrained_matrix = torch.tensor(self.glove.word_vectors,dtype=torch.float)
        self.embedding = nn.Embedding.from_pretrained(pretrained_matrix,freeze=False)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout = 0.22)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        _,(out1,_) = self.lstm(x)
        out1 = self.fc(out1[-1,:,:])
        return out1

class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers = 4, bidirectional = True,batch_first=True)
        self.layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        _,out = self.rnn(x)
        out = self.layer(out[-1,:,:])
        return out
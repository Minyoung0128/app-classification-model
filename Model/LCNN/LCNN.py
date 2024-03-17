import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from glove import Glove

class LCNN(nn.Module):
    final_size = 0
    
    def __init__(self, num_embeddings, hidden_dim, num_layers, input_size, output_size, batch_size) :
        super(LCNN,self).__init__()
        self.embedding = nn.Embedding(num_embeddings = num_embeddings,embedding_dim = input_size)
        
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout = 0.22)
        
        self.layer1= nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=1, stride=1),
            nn.ReLU()
        )
        
        self.layer2= nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(3200,output_size),
            # nn.Linear(4000,output_size)
        )
        
        self.batch_size = batch_size
        
    def forward(self,x):
        out = self.embedding(x)
        # print(out1.shape)
        _,(out,_) = self.lstm(out)
        # print(out2[-1,:,:].unsqueeze(2).shape)
        out = self.layer1(out[-1,:,:].unsqueeze(1))
        # print(out3.shape)
        out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # print("out4",out4.view(self.batch_size, -1).shape)
        out = self.fc(out.view(self.batch_size, -1))
        # print(out5.shape)

        return out
    
class LCNN_Glove(nn.Module):
    final_size = 0
    
    def __init__(self, num_embeddings, hidden_dim, num_layers, input_size, output_size, batch_size, glove_model) :
        super(LCNN_Glove,self).__init__()
        
        self.glove = Glove.load(glove_model)
        pretrained_matrix = torch.tensor(self.glove.word_vectors,dtype=torch.float)
        
        self.embedding = nn.Embedding.from_pretrained(pretrained_matrix,freeze=False)
        
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout = 0.22)
        
        self.layer1= nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1, stride=1),
            nn.ReLU()
        )
        
        self.layer2= nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
    
        self.fc = nn.Sequential(
            nn.Linear(6400,3000),
            nn.ReLU(),
            nn.Linear(3000,output_size)
        )
        
        self.batch_size = batch_size
        
    def forward(self,x):
        out1 = self.embedding(x)
        # print(out1.shape)
        _,(out2,_) = self.lstm(out1)
        # print(out2[-1,:,:].unsqueeze(2).shape)
        out3 = self.layer1(out2[-1,:,:].unsqueeze(1))
        # print(out3.shape)
        out4 = self.layer2(out3)
        # print("out4",out4.view(self.batch_size, -1).shape)
        out5 = self.fc(out4.view(self.batch_size, -1))
        # print(out5.shape)

        return out5


class CNN(nn.Module):
    final_size = 0
   
    def __init__(self, num_embeddings, input_size, output_size, batch_size) :
        super(CNN,self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings,embedding_dim=input_size)
       
        self.layer1= nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=200, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )
       
        self.layer2= nn.Sequential(
            nn.Conv1d(in_channels=200, out_channels=400, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )
       
        self.layer3= nn.Sequential(
            nn.Conv1d(in_channels=400, out_channels=800, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )
       
        self.fc = nn.Sequential(
            nn.Linear(4800,2000),
            nn.Linear(2000,output_size)
        )
       
        self.batch_size = batch_size
       
    def forward(self,x):
        out1 = self.embedding(x).permute(0,2,1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3).reshape(self.batch_size,-1)
        out5 = self.fc(out4)
        return out5
import torch.nn as nn 

class FeedForwardLayer(nn.Module):
    
    def __init__(self, fc1, fc2):
        super(FeedForwardLayer, self).__init__()
        self.fc1 = fc1 # [batch, d_embedding, k]
        self.relu = nn.ReLU() 
        self.fc2 = fc2 # [batch, k, d_embedding]
        
    def forward(self, x):
        '''
        input 
        x : [batch, n, d_embedding] tensor
        '''
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out
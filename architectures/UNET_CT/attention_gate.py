
from torch import nn 

class Attention_gate(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_gate,self).__init__()
        self.W_g = nn.Sequential(
            nn.GroupNorm(32,F_g),
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            )
        
        self.W_x = nn.Sequential(
            nn.GroupNorm(32,F_l) ,
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
        )

        self.psi = nn.Sequential(
            nn.GroupNorm(32,F_int), 
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.Sigmoid()
        )
        
        
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x): 
        g1 = self.W_g(g) 
        x1 = self.W_x(x) 
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

data = torch.rand(200,3)
label = torch.rand(200,5)

class net_test(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        super().__init__()
        self.net1 = nn.Linear(in_ch,32).to(torch.device('cuda:0'))
        self.net2 = nn.Linear(32,32).to(torch.device('cuda:0'))
        self.net3 = nn.Linear(32,32).to(torch.device('cuda:0'))
        self.net4 = nn.Linear(32,32).to(torch.device('cuda:0'))
        self.net5 = nn.Linear(32,32).to(torch.device('cuda:0'))
        self.net6 = nn.Linear(32,32).to(torch.device('cuda:0'))
        self.net7 = nn.Linear(32,out_ch).to(torch.device('cuda:1'))

    def forward(self, x):
        x = self.net1(x.to(torch.device('cuda:0')))
        x = self.net2(x)
        x = self.net3(x)
        x = self.net4(x)
        x = self.net5(x)
        x = self.net6(x)

        x = self.net7(x.to(torch.device('cuda:1')))
        # x = x.cpu()
        return x

model = net_test(3,5)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
crit = nn.MSELoss()

for epoch in tqdm(range(100000)):
    label = label.to((torch.device('cuda:1')))
    output = model(data)
    optimizer.zero_grad()
    loss = crit(output,label)
    loss.backward()
    optimizer.step()
    print(loss.item())
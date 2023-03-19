import torch
import torch.nn as nn
import torch.nn.functional as F

class DivisionNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DivisionNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x.shape = (batch_size, 2)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x.squeeze(dim=1)


net = DivisionNet(2, 10)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(10000):
    x = torch.randn(100, 2)
    y = x[:, 0] / x[:, 1]
    y_pred = net(x)

    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch %d, loss = %.4f' % (epoch, loss.item()))

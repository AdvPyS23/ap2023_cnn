import torch.nn as nn
import torch.optim as optim

#  create a class that inherits from the nn.Module class

batch_size = 32

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, x):
        try:
            if x.size() != (batch_size, 1, 28, 28):
                raise ValueError("Input tensor size is not (batch_size, 1, 28, 28)")
        except ValueError as ve:
            print(ve)
            return None

        try:
            x = nn.functional.relu(self.conv1(x))
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            x = nn.functional.relu(self.conv2(x))
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            x = x.view(-1, 64 * 5 * 5)
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
            return x
        except RuntimeError as re:
            print(re)
            return None


# train function
# def train(net, trainloader, epochs, device):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#     net.to(device)
#
#     for epoch in range(epochs):  # loop over the dataset multiple times
#         running_loss = 0.0
#         for i, data in enumerate(trainloader, 0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             # zero the parameter gradients
#             optimizer.zero_grad()
#
#             # forward + backward + optimize
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             # print statistics
#             running_loss += loss.item()
#             if i % 2000 == 1999:    # print every 2000 mini-batches
#                 print('[%d, %5d] loss: %.3f' %
#                       (epoch + 1, i + 1, running_loss / 2000))
#                 running_loss = 0.0
#
#     print('Finished Training')


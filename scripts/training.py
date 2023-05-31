import torch
import torch.nn as nn
import torch.optim as optim
import traceback

#  create a class that inherits from the nn.Module class

batch_size = 32 # Problematic, find a workaround

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
                raise ValueError(f"Input tensor size is {x.size()}, expected size is (batch_size, 1, 28, 28)")
        except ValueError as ve:
            print(ve)
            return None

        try:
            x = nn.functional.relu(self.conv1(x))
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            x = nn.functional.relu(self.conv2(x))
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            x = x.view(-1, 12 * 4 * 4)
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
            return x
        except RuntimeError as re:
            print("Runtime Error:", re)
            print("Traceback:", traceback.format_exc())
            return None


# train function

def train(net, trainloader, testloader, epochs, device):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.to(device)
    losses_train = []
    losses_test = []
    acc_train = []
    acc_test = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f'Epoch {epoch+1}:')
        running_loss = 0.0
        running_test_loss = 0.0
        running_acc = 0.0
        correct_batches = 0.0
        # Training section of the epoch
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_acc += (outputs.argmax(1) == labels).type(torch.float).sum().item()
            if i % 100 == 99:    # print and save every 100 batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                losses_train.append(running_loss / 100)
                acc_train.append(running_acc)
                running_loss = 0.0
                running_acc = 0.0

        # Testing section of the epoch
        size = len(testloader.dataset)
        num_batches = len(testloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for i, (X, y) in enumerate(testloader, 0):
                pred = net(X)
                test_loss += criterion(pred, y).item()
                running_test_loss += criterion(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                correct_batches += (pred.argmax(1) == y).type(torch.float).sum().item()
                if i % 100 == 99:    # save every 100 batches
                    losses_test.append(running_test_loss / 100)
                    acc_test.append(correct_batches)
                    running_test_loss = 0.0
                    correct_batches = 0.0

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


        # Save model at each epoch
        torch.save(net.state_dict(), f'./models/cnn_model_epoch_{epoch+1}.pth')

    print('Finished Training')
    return losses_train, losses_test, acc_train, acc_test


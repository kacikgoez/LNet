import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

if __name__ == "__main__":
    from WebLayoutData import WebData
else:
    from .WebLayoutData import WebData
'''
    Open tasks: 
        - add easy grid scaling 
'''

# Number of channels for size separation
sep = 3
# Grid resolution for webpage histrogram
grid = 7
# Learning rate
lr = 0.001
# Batch size
batch_size = 16
# Epochs
epochs = 50
# Percentage of training data reserved for testing
test_percentage = 0.1

'''
            nn.Conv2d(sep * 2, 6, 3, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 8, 2, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 12, 2, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
'''

'''
    Network for checking how likely a logo is to be the identity logo
    Def
'''
class WebLayoutNet5(nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.cnn_relu_stack = nn.Sequential(
            nn.Conv3d(1, 16, (6, 1, 1), stride=(1, 1, 1), padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, (3, 2, 2), stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 8, (1, 6, 6), stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

        ''' 
        self.cnn_relu_stack = nn.Sequential(
            nn.Conv3d(1, 16, (6, 1, 1), stride=1, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 8, (1, 7, 7), stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )'''

    def forward(self, x):
        logits = self.cnn_relu_stack(x)
        return logits

    def soft_forward(self, x):
        return self.softmax(self.forward(x))

class WebLayoutNet7(nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.cnn_relu_stack = nn.Sequential(
            nn.Conv3d(1, 16, (6, 1, 1), stride=(1, 1, 1), padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 8, (1, 7, 7), stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

        ''' 
        self.cnn_relu_stack = nn.Sequential(
            nn.Conv3d(1, 16, (6, 1, 1), stride=1, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 8, (1, 7, 7), stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )'''

    def forward(self, x):
        logits = self.cnn_relu_stack(x)
        return logits

    def soft_forward(self, x):
        return self.softmax(self.forward(x))


if __name__ == "__main__":
    # Number of grids
    # Original dataset (no random alterations if False, random alteration if True)
    dataset = WebData(grid, "./data5.json", sep, False)
    # Original dataset
    odataset = WebData(grid, "./data5.json", sep, False)
    # Expanded set only
    expset = WebData(grid, "./data5.json", sep, True, True)
    # Validation dataset size
    val_size = round(len(dataset) * test_percentage)

    # Split dataset
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])
    # Training data
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    # Validiation data
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)
    # Original dataset
    orig_loader = DataLoader(odataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # Expanded set only
    exp_loader = DataLoader(expset, batch_size=batch_size, shuffle=True, drop_last=True)
    # List of test sets
    sets = [val_loader, orig_loader, exp_loader]

    # Define network
    net = WebLayoutNet7()

    # Loss function and optimization algorithm
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    #optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=0)
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')

    #torch.save(net.state_dict(), "model-2-5x5.pth")

    # Softmax to get an estimate for the confidence (maps between 0 and 1)
    soft = nn.Softmax(dim=1)
    for testset in sets:
        correct = 0
        avg_conf = 0
        total = 0
        posr = 0
        negr = 0
        with torch.no_grad():
            for x, data in enumerate(testset):
                images, labels, uri = data
                outputs = net(images.float())
                conf = soft(outputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                l = (predicted == labels)
                for i, j in enumerate(l):
                    if predicted[i] != labels[i]:
                        posr += 1
                        avg_conf += torch.max(conf)
                        print(images[i])
                        print(uri[i])
                        print(conf[i])
                        print("P:" + str(predicted[i]), "L:" + str(labels[i]))
                correct += (predicted == labels).sum().item()
        print("-" * 50)
        if testset == val_loader:
            print("PERFORMANCE ON TEST SET (RANDOM SUBSET OF TRAINING SET)")
        elif testset == orig_loader:
            print("PERFORMANCE ON ORIGINAL")
        print("Correct predictions: ", posr, correct - posr)
        print("Total size of validation: ", total)
        print('Accuracy of the network on the ' + str(total) + ': %d %%' % (
                100 * correct / total))
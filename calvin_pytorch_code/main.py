import cancerdataset
import glob
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torchvision.transforms import Lambda
from torchvision.transforms.v2 import ToDtype
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

image_paths = np.array(glob.glob('/Users/calvindejong/Downloads/cancer_images/IDC_regular_ps50_idx5/**/*.png', recursive = True))
labels = np.zeros(len(image_paths),dtype=int)
for i in range(len(image_paths)):
    label = image_paths[i][-5]
    labels[i] = int(label)
    #labels[i] = image[i][-5]

img_transforms = transforms.Compose(
    [transforms.Resize((50,50)),
    transforms.v2.ToDtype(torch.float),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



dataset = cancerdataset.CancerDataset(
    img_labels=labels,
    img_paths=image_paths,
    transform=img_transforms,
    target_transform=Lambda(lambda y: torch.zeros(
    2, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
)

train_dataset, test_dataset = torch.utils.data.random_split(
    dataset=dataset,
    lengths=[0.8,0.2],
    generator=torch.Generator().manual_seed(22)
)
    

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

data, _  = next(iter(train_dataloader))
print(data.shape)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(50*50*3, 1000),
            nn.ReLU(), # f(x) = max(0,x)
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32,kernel_size=3,stride=1,padding=1)
        self.fc1 = nn.Linear(1152,500)
        self.fc2 = nn.Linear(500,80)
        self.fc3 = nn.Linear(80,2)
    def forward(self,x):
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.fc1(x)            # Apply fully connected layer
        x = self.fc2(x)
        x = self.fc3(x)
        return x

conv = ConvNet().to(device)

learning_rate = 1e-3
batch_size = 64
epochs = 5

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(conv.parameters(), lr=learning_rate)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, conv, loss_fn, optimizer)
    test_loop(test_dataloader, conv, loss_fn)
print("Done training")

# Save model

torch.save(conv.state_dict(), 'model_weights.pth')










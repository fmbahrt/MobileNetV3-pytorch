"""
Just a minimal training script for trying out the mobilenet implementation
on CIFAR
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from layers import Config, MobileNetV3

# Load the model
model = MobileNetV3(Config.large)

# some data fun
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=32,
    shuffle=True
)


testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=32,
    shuffle=False
)

criterion = nn.CrossEntropyLoss()
opt       = optim.Adam(model.parameters())

# Time to train
EPOCHS = 1
for epoch in range(EPOCHS):

    for i, data in enumerate(trainloader):
        inputs, labels = data

        opt.zero_grad()

        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        opt.step()

        print("loss: {}".format(loss.item()))

print("BOOM")

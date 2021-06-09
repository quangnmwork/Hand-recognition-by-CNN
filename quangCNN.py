import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import statistics
from modelCNNLarge import CNNLarge

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# inital
num_epochs = 5
learning_rate = 0.001
batch_size = 32
classes = ('hello', 'like', 'love')

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((26, 26))])

directory = r'data\\train_data'
train_dataset = []
test_dataset = []

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        # print(filename + " Labels:" +filename[-5])
        img = cv2.imread(os.path.join(directory, filename), 0)
        # print(img)
        obj = (transform(img), int(filename[-5]) - 1)
        # print(transform(img).shape)
        # print(transform(img).shape)
        train_dataset.append(obj)
directory = r'data\\test_data'
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(directory, filename), 0)
        obj = (transform(img), int(filename[-5]) - 1)
        test_dataset.append(obj)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False)
# print(train_dataset[0])
# i1,l1 = next(iter(train_loader))
# cnt=0
# while i1 is not None:
#     cnt+=1
#     i1, l1 = next(iter(train_loader))
# print(cnt)

classes = ('hello', 'ok', 'love')

model = CNNLarge().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
print(n_total_steps)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # print("Outputs",outputs)
        # print("Label",labels)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 64 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(3)]
    n_class_samples = [0 for i in range(3)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # print(labels)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

PATH = 'modelLarge.pth'
# print(model.state_dict())
# print(optimizer.state_dict())
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, PATH)

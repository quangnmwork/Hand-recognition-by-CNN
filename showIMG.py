import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
import torchvision.transforms as transforms
from processData import Dataset
from sklearn.model_selection import train_test_split
from modelCNN import CNN
import numpy as np
from PIL import Image
import cv2

dataset = Dataset()
classes = dataset.data['mapping']

X_train = []
Y_train = []
for i in range(math.floor(len(dataset.data['image'])*0.8)):
    X_train.append(torch.tensor(dataset.data['image'][i]))
    Y_train.append(dataset.data['labels'][i])
def imshow(data,x):
    data = np.array(data, dtype=np.uint8)
    img = Image.fromarray(data)
    print(Y_train[x])
    img.show()

labels = ['Ok', 'Silent', 'Dislike', 'Like', 'Hi', 'Hello', 'Stop', ' ']

PATH = 'model.pth'
model = CNN()
checkpoint = torch.load(PATH)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()

imshow(X_train[10000],10000)



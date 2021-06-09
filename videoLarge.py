import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import cv2
from cv2 import INTER_AREA
from modelCNNLarge import CNNLarge
from BEN_detectFinger import handLandmarks
import time
import numpy as np
import torchvision.transforms as transforms

labels = ['hello', 'like', 'love']

# load model
PATH = 'modelLarge.pth'
model = CNNLarge()
checkpoint = torch.load(PATH)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



model.load_state_dict(checkpoint['model_state_dict'],strict=False)
#print(checkpoint['optimizer_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()

ben = handLandmarks()
pTime = 0
cTime = 0
video = 0

def processImg(img):
    kernel = np.ones((3, 3), np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range  skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    # extract skin colur imagw
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    #fill dark spots
    mask = cv2.dilate(mask, kernel, iterations=4)
    # blur img
    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    return mask
cap = cv2.VideoCapture(video)
while True:
    success, img = cap.read()
    gesture = ' '
    ret, frame = cap.read()
    # print(ret)
    frame = cv2.flip(frame, 1)
    newFrame = cv2.resize(frame, (1500, 800))

    cv2.rectangle(newFrame, (900, 100), (1400, 600), (0, 255, 0), 0)
    roi = newFrame[100:600, 900:1400]

    kernel = np.ones((3, 3), np.uint8)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    #cv2.imshow("hsv", hsv)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    newMask = mask
    mask = cv2.resize(mask,(26,26))
    mask = mask.reshape([-1, 1, 26, 26])
    mask = mask.astype('float32')
    mask = torch.tensor(mask)
    with torch.no_grad():
        output = model.forward(mask)
        predict = torch.max(output, 1)[1]
        gesture = labels[predict.item()]
        print(labels[predict.item()])
    cv2.putText(newFrame, f"{gesture}", (50, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0), 6)
    cv2.imshow('new', newFrame)
    cv2.imshow('mask', newMask)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
cap.release()

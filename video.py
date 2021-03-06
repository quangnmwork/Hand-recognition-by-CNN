import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import cv2
from cv2 import INTER_AREA

from modelCNN import CNN
from BEN_detectFinger import handLandmarks
import time
import numpy as np
import torchvision.transforms as transforms

labels = ['Ok', 'Silent', 'Dislike', 'Like', 'Hi', 'Hello', 'Stop', ' ']

# load model
PATH = 'model.pth'
model = CNN()
checkpoint = torch.load(PATH)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print(optimizer.state_dict())
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()

# process video
ben = handLandmarks()
pTime = 0
cTime = 0
video = 0
# video = 'rtsp://ftech:ad1235min@192.168.130.27/live0'
cap = cv2.VideoCapture(video)
while True:
    success, img = cap.read()
    ben.showFinger(img)
    pointList, box = ben.storePoint(img)
    check, img1, img2 = ben.drawAndResize(img, box)
    img1, img2 = ben.getFeature()
    gesture = ' '
    imgShow = img2
    if type(img2) == 'NoneType':
        pass
    else:

        img2 = cv2.resize(img2,(26,26))
        img3=img2.astype('float32')
        img3 = img3.T
        img3 = torch.tensor(img3)
        img3 = img3.reshape([-1,1,26,26])
        print(img3.shape)
        with torch.no_grad():
            output = model.forward(img3)
            predict = torch.max(output, 1)[1]
            gesture = labels[predict.item()]
            print(labels[predict.item()])
    if len(box) != 0:
        cv2.rectangle(img, (box[0] - 20, box[1] - 20), (box[2] + 20, box[3] + 20), (255,0, 0), 2)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps))+f" {gesture}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("ben", img)
    if check == True:
        cv2.imshow("image", imgShow)
        #cv2.imshow("img3",img3)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
cap.release()

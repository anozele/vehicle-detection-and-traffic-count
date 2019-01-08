import cv2
import pandas
import os
import numpy as np
from keras.models import model_from_json


json_file=open('model.json','r')
lastmodel=json_file.read()
json_file.close()
load_model=model_from_json(lastmodel)

load_model.load_weights("model.h5")

img=cv2.imread("31.jpg",0)
cv2.imshow("image",img)
img=cv2.resize(img,(100,100))
img=np.array(img).reshape(-1,100,100,1)

k=load_model.predict(img)


import cv2
import numpy as np
import pickle
import os
import random
#getting all the paths i have 

path=os.getcwd()
path1=path
labels=["cars","bus","bike"]

#create my traing sample data 
train_data=[]       
def training_data():
    for label in labels:
        path2=os.path.join(path1,label)
        images=os.listdir(path2)
        y=labels.index(label)
        for im in images:
            img=cv2.imread(os.path.join(path2,im),cv2.IMREAD_GRAYSCALE)
            img=cv2.resize(img,(100,100))
            train_data.append([img,y])

training_data()
random.shuffle(train_data)
X=[]
y=[]

for feat,labls in train_data:
    X.append(feat)
    y.append(labls)
    
X=np.array(X).reshape(-1,100,100,1)
y=np.array(y,"float32")



#saving train features data           
p_out=open("train.pickle","wb")
pickle.dump(X,p_out)
p_out.close()

# saving my labels data according to my features

p_out=open("test.pickle","wb")
pickle.dump(y,p_out)
p_out.close()

print("evrythinh finees")


    



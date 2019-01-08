import numpy as np
from keras import utils
from keras.models import Sequential
from keras.layers import Dropout,Flatten,Dense,Activation,Conv2D,MaxPooling2D
import pickle
X=pickle.load(open("Xn.pickle","rb"))
y=pickle.load(open("yn.pickle","rb"))

X=X/255
X=utils.to_categorical(X,3)
y=utils.to_categorical(y,3)

model=Sequential()

model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))


model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
          
model.add(Flatten())


model.add(Dense(70,activation='relu'))
model.add(Dense(1,activation="softmax"))

model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

model.fit(X,y,epochs=25,validation_split=0.3)



new_model_json=model.to_json()
with open("model.json","w")  as json_file:
    json_file.write(new_model_json)
   
model.save_weights("model.h5")

import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten
from random import shuffle
import cv2
import os

Base_dir = os.path.dirname(os.path.abspath('__file__'))
Train_dir=os.path.join(Base_dir,"CAX_Superhero_Train")
Test_dir = os.path.join(Base_dir,"CAX_Superhero_Test")
Image_size=32
n_classes=12

# creating training data
def create_training_data():
    current_id=0
    training_data  = []
    label_ids = {}
    for root,dirs,files in os.walk(Train_dir):
        for file in files:
            if file.endswith("jpg") or file.endswith("png") or file.endswith("jpeg"):
                path  = os.path.join(root,file)
                label = os.path.basename(root).replace(" ","_").lower()
                if not label in label_ids:
                    label_ids[label]=current_id
                    current_id+=1
                id_= label_ids[label]
                
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
                img = cv2.resize(img, (Image_size,Image_size))
                training_data.append([np.array(img),np.array(id_)])
    shuffle(training_data)     
    np.save('train_data.npy', training_data)
    with open("label.pickel","wb") as f:
        pickle.dump(label_ids,f)
        
    return training_data

# creating testing data
def create_testing_data():
    testing_data=[]
    f=[]
    for root,dirs,files in os.walk(Test_dir):
        for file in files:
            if file.endswith("jpg") or file.endswith("png") or file.endswith("jpeg"):
                path=os.path.join(root,file)
                filename=file.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
                img = cv2.resize(img, (Image_size,Image_size))
                testing_data.append(np.array(img))
                f.append(filename)
    np.save('test_data.npy',testing_data)
    with open("filename.pickel","wb") as fi:
        pickle.dump(f,fi) 
    return testing_data


test_data =create_testing_data()
data = create_training_data()

label_dict=pickle.load(open( "label.pickel", "rb" ))
inv_label_dict={v:k for k,v in label_dict.items()}

data=np.load("train_data.npy")
train_data = data[:-500]
val_data  = data[-500:]

# train data
x_train=np.array([i[0]  for i in train_data])
label_train=np.array([i[1] for i in train_data])

# validation data
x_val=np.array([i[0] for i in val_data])
label_val =np.array([i[1] for i in val_data])

#normalized
x_train=x_train/255.0
x_val=x_val/255.0



#convnet 5 layer model
def create_model():
    model = Sequential()
    #layer 1
    model.add(Conv1D(5,kernel_size=5,activation='relu', strides = 1 ,input_shape=(Image_size,Image_size)))
    model.add(MaxPooling1D(pool_size=2, strides = 2)) 

    
    #layer2
    model.add(Conv1D(16,kernel_size=5,activation='relu', strides = 1))
    model.add(MaxPooling1D(pool_size=2, strides = 1))
    model.add(Dropout(0.25)) 
    
    # layer 3
    model.add(Flatten())
    model.add(Dense(units = 120, activation = 'relu'))
 
    
    # layer 4
    model.add(Dense(units = 240, activation = 'relu'))
    model.add(Dropout(0.5)) 
    #output layer
    model.add(Dense(units = n_classes, activation = 'softmax'))
    
    return model

def train_model():
    model=create_model()
    model.summary()
    batch_size = 512
    epochs = 350
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, label_train, batch_size=batch_size, epochs=epochs, verbose=1, 
                  validation_data=(x_val, label_val))

    test_loss,test_accuracy = model.evaluate(x_val,label_val)

    model.save_weights('train_model.h')
    print("Test Accuracy :",test_accuracy)
    return model

def prediction(model):
    x_test =np.load("test_data.npy")
    img_name=pickle.load(open( "filename.pickel", "rb" ))

    output=[]
    for x in x_test:
        img=np.expand_dims(x,axis=0)
        img=img/255.0
        pred=model.predict(img)
        out=inv_label_dict[np.argmax(pred)]
        output.append(out)
    
    img_name=np.array(img_name)
    output=np.array(output)
    arr=np.array([img_name,output])
    arr=arr.transpose()
    df=pd.DataFrame(arr)
    df.to_csv("q4_python.csv",index=False,header=True,sep='\t', encoding='utf-8')


t_model =train_model()
prediction(t_model)

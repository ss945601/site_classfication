import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout,Activation
from keras.layers.normalization import BatchNormalization
from tensorflow.keras import layers
from keras.utils import to_categorical
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import data, io
from PIL import Image
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mimage
from tensorflow.keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import csv
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
seed = 14
np.random.seed(seed)

## -- config -- ##
classification = ['bedroom', 'CALsuburb', 'coast', 'forest', 'highway', 'industrial', 'insidecity', 'kitchen', 'livingroom', 'mountain', 'opencountry', 'PARoffice', 'store', 'street', 'tallbuilding']

mapping_label_file = './mid_term_mapping.txt'
trainPath = './train/'
testPath = './testset/'
img_w = 120
img_h = 120
channel = 1
isReadModel = False
modelName = "best_model.h5"#'eps150.h5'
csv_file = modelName+"_ans.csv"
set_batch_size = 64
eps = 100
numOfClassifications = len(classification)
plt.style.use('fivethirtyeight')
testID = []
imgTest = []

## -- function -- ##
def load_train_data():
    x_train = []
    y_train = []
    ## Open file
    fp = open(mapping_label_file, "r")
    # 變數 lines 會儲存 filename.txt 的內容
    lines = fp.readlines()
    # close file
    fp.close()
    # print content
    for i in range(len(lines)):
        name = lines[i].split(", ")[0]
        idx = int(lines[i].split(", ")[1])
        classification[idx] = name
    
    for className in classification:
        print("Load:"+str(classification.index(className)+1) + "-" +className)
        for filename in glob.glob(trainPath+className+"/*.jpg"): #assuming gif
            im=mimage.imread(filename)
            im = resize(im, (img_w,img_h,channel))
            img_array = np.array(im)
            x_train.append(img_array)
            y_train.append( classification.index(className) )
        print("Done")
     
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    #Print the data type of x_train
    print((x_train.shape))
    #Print the data type of y_train
    print((y_train.shape))
    x_train = x_train.astype('float32')
    x_train = x_train / 255 # rescale rbg value

    return x_train, y_train

def load_test_data():
    x_test = []
    ## Open file
    fp = open(mapping_label_file, "r")
    # 變數 lines 會儲存 filename.txt 的內容
    lines = fp.readlines()
    # close file
    fp.close()
    # print content
    for i in range(len(lines)):
        name = lines[i].split(", ")[0]
        idx = int(lines[i].split(", ")[1])
        classification[idx] = name
    print("Load test images")
    for filename in glob.glob(testPath+"/*.jpg"):
        testID.append(filename)
        im=mimage.imread(filename)
        im = resize(im, (img_w,img_h,channel))
        imgTest.append(im)
        img_array = np.array(im)
        x_test.append(img_array)
    print("Done")
    #Print the data type of x_test
    x_test =  np.array(x_test)
    x_test = x_test.astype('float32')
    x_test = x_test / 255 # rescale rbg value
    print(x_test)
    return x_test
    
def getModel():
    model = Sequential()
    model.add(Conv2D(128, (3, 3), input_shape=(img_w,img_h,channel)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(numOfClassifications, activation='softmax'))
    print(model.summary())
    return model
    
    
    


if isReadModel:
    waitPredictData = load_test_data()
    model = tf.keras.models.load_model(
        modelName,
        custom_objects=None,
        compile=False
    )
else:
    x_train, y_train = load_train_data()
    y_train_one_hot = to_categorical(y_train) # ex: label_5 set to 0,0,0,0,1,0....(num of classifications)
    print(y_train_one_hot)
    waitPredictData = load_test_data()
    model = getModel()
    
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss='categorical_crossentropy',
    optimizer= sgd,
    metrics=['accuracy'])

    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train_one_hot, test_size=0.2, random_state=seed,stratify=y_train_one_hot)


    checkpoint = ModelCheckpoint("best_model.h5", monitor='loss', verbose=1,
        save_best_only=True, mode='auto', period=10)
        
    hist = model.fit(X_train, Y_train, validation_data=(X_test,Y_test),
    batch_size=set_batch_size, epochs=eps,
    callbacks=[checkpoint],
    shuffle=True,verbose=1 )

    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    #Visualize the models loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()
    model.save(modelName)

predictions = model.predict(waitPredictData)
csv_columns = ['id','class']
dict_data = []
for i in range(0,len(predictions)):
    id = testID[i].split('/')[-1].split('.')[0]
    probs = np.argmax(predictions[i])
    dict_data.append({'id':id,'class':probs})
#    if i <= 8:
#        print(id + ":" + classification[probs])
#        imgArr = np.array(imgTest[i])
#        img = plt.imshow(imgArr.squeeze())
#        plt.show()

#    dict_data['id'].append(id)
#    dict_data['class'].append(probs)


try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in dict_data:
            writer.writerow(data)
except IOError:
    print("I/O error")

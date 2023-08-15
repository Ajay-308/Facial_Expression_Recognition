#!/usr/bin/env python
# coding: utf-8

# In[63]:


from keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
import pandas as pd
import numpy as np


# In[64]:


TRAIN_DIR = 'images/train'
TEST_DIR = 'images/validation'


# In[65]:


def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir,label)):
            image_paths.append(os.path.join(dir,label,imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths,labels


# In[66]:


train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)


# In[67]:


print(train)


# In[68]:


test = pd.DataFrame()
test['image'], test['label'] = createdataframe(TEST_DIR)


# In[69]:


print(test)
print(test['image'])


# In[70]:


from tqdm.notebook import tqdm


# In[71]:


def extract_features(images):
    features = []
    for image_path in tqdm(images):
        img = load_img(image_path, grayscale=True, target_size=(48, 48))  # Load image and resize
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    return features

train_features = extract_features(train['image'])


# In[72]:


test_features = extract_features(test['image'])


# In[73]:


x_train = train_features/255.0
x_test = test_features/255.0


# In[74]:


from sklearn.preprocessing import LabelEncoder


# In[75]:


le = LabelEncoder()
le.fit(train['label'])


# In[76]:


y_train = le.transform(train['label'])
y_test = le.transform(test['label'])


# In[77]:


y_train = to_categorical(y_train,num_classes = 7)
y_test = to_categorical(y_test,num_classes = 7)


# In[78]:


model = Sequential()
# convolutional layers
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
# fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(7, activation='softmax'))


# In[79]:


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy' )


# In[ ]:


model.fit(x= x_train,y = y_train, batch_size = 128, epochs = 10, validation_data = (x_test,y_test))


# In[57]:


model_json = model.to_json()
with open("emotiondetector.json",'w') as json_file:
    json_file.write(model_json)
model.save("emotiondetector.h5")


# In[27]:


from keras.models import model_from_json


# In[29]:


json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")


# In[30]:


label = ['angry','disgust','fear','happy','neutral','sad','surprise']


# In[31]:


def ef(image):
    img = load_img(image,grayscale =  True )
    feature = np.array(img)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0
    


# In[34]:


import matplotlib.pyplot as plt
image = 'images/train/sad/42.jpg'
print("original image is of sad")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)
plt.imshow(img.reshape(48,48),cmap='gray')


# In[55]:


image = 'images/train/fear/5.jpg'
print("original image is of fear")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)
plt.imshow(img.reshape(48,48),cmap='gray')


# In[49]:


from PIL import Image
import numpy as np

image_path = r"C:\Users\mraja\OneDrive\Documents\python\Face_Emotion_Recognition_Machine_Learning-main\images\train\neutral\0.jpeg"

# Load and preprocess the image using PIL
img = Image.open(image_path).convert('L')  # Convert to grayscale
img = img.resize((48, 48))  # Resize to model input size
img = np.array(img) / 255.0  # Convert to NumPy array and normalize pixel values to [0, 1]

# Reshape image for model prediction
img = img.reshape(1, 48, 48, 1)

# Perform prediction and display
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("Original image is of neutral")
print("Model prediction is", pred_label)

import matplotlib.pyplot as plt
plt.imshow(img.reshape(48, 48), cmap='gray')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





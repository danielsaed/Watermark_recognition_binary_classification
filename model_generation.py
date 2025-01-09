
# ## Machine Learning Watermark Detection Model

# Using NN on Tensorflow
#
# Current dataset is 600 images


import os
import PIL
import PIL.Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from keras.preprocessing.image import ImageDataGenerator

path = ""

#-----------------------------------------------------------start of model creation
# %%
batch_size = 32
img_size = (320,400)

# %%
train_ds = tf.keras.utils.image_dataset_from_directory(
  r"C:\Users\d84316956\Desktop\Dataset",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=img_size,
  batch_size=batch_size)

# %% [markdown]
# Validation Data

# %%
#for the img quatity it was imposible to train de dataset with less img, thats why the model is validated with the same ones unfortunetly.
val_ds = tf.keras.utils.image_dataset_from_directory(
  r"C:\Users\d84316956\Desktop\Dataset",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=img_size,
  batch_size=batch_size)

# %% [markdown]
# Dataset Preprocessing

# %%
normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in [0,1].
print(np.min(first_image), np.max(first_image))

# %% [markdown]
# Layers Neural Network

# %%
num_classes = 2

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(320, 400,3)),
  tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32,3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(num_classes,'sigmoid')
])

# %% [markdown]
# Model Compile

# %%
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
  #loss='categorical_crossentropy',
  metrics=['accuracy'])

#model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# %% [markdown]
# Fitting Model

# %%
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=11
  
  )

# %%

model.save('deeplearning.h5') #saves the model

#-------------------------------------------------------------end of model creation

'''# %%
train_datagen = ImageDataGenerator(rescale=1./255,                   
                                   validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

# %%
train_ds = train_datagen.flow_from_directory(
    r"C:\Users\d84316956\Desktop\Dataset\Training\good", 
    target_size=img_size, 
    color_mode='rgb', 
    batch_size=32, 
    class_mode='binary', 
    subset='training',
    shuffle=True,
    seed=42
    )

val_ds = train_datagen.flow_from_directory(
    r"C:\Users\d84316956\Desktop\Dataset\Training\good",
    target_size=img_size, 
    color_mode='rgb', 
    batch_size=32, 
    class_mode='binary', 
    subset='validation', 
    shuffle=False
    )

test_generator = test_datagen.flow_from_directory(
    r"C:\Users\d84316956\Desktop\testTresh",
    target_size=img_size, 
    color_mode='rgb',
    batch_size=32, 
    class_mode='binary', 
    shuffle=False
    )

image_batch, labels_batch = next(iter(train_ds))
first_image = image_batch[0]
# Notice the pixel values are now in [0,1].
print(np.min(first_image), np.max(first_image))

# %%
for _ in range(5):
    img, label = train_ds.next()
    print(img.shape)   #  (1,256,256,3)
    plt.imshow(img[0])
    plt.show()

# %% [markdown]
# Train Data



# %%
#loaded_model = load_model('modelo.h5',custom_objects=None, compile=True)
new_model = keras.models.load_model(r'C:\Users\d84316956\Desktop\deeplearning.h5')

# %%
pd.DataFrame(history.history).plot(figsize = (16, 10))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# %%
model.evaluate(test_generator)

# %%

it = test_datagen.flow_from_directory(r'C:\Users\d84316956\Desktop\testTresh',
                              # don't generate labels
                              #class_mode='binary',
                              color_mode='rgb',
                              classes=['0'],
                              # don't shuffle
                              shuffle=False,
                              # use same size as in training
                              target_size=img_size)

batch_size = 10

# %%
#path_images_for_predict = r"C:\Users\d84316956\Desktop\Dataset\Predict\Anexo 1. INFORME DE AMPLIACION - COLONIAL"
path_images_for_predict = r"C:\Users\d84316956\Desktop\testTresh"
it = test_datagen.flow_from_directory(r"C:\Users\d84316956\Desktop\testTresh",
            # don't generate labels
            #class_mode='binary'
            color_mode='rgb',
            classes=["1"],
            # don't shuffle
            shuffle=False,
            # use same size as in training
            target_size=img_size)
        
preds = new_model.predict_generator(it)

dic = {}
lst = []
for num,i in enumerate(preds):
    print(np.argmax(i))
    if np.argmax(i) == 1:
        #dic[file_name][img_name] = np.argmax(i)
        break

# %%
it.filenames[0]

# %%
preds = new_model.predict_generator(it)

dic = {}
lst = []
for num,i in enumerate(preds):
    dic[it.filenames[num]] = np.argmax(i)
    if np.argmax(i) == 1:
        print(it.filenames[num])
        print('marca agua')

# %%
print(dic)

# %%
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
axs = axs.flatten()

axs[0].imshow(img)
axs[1].imshow(treshImage1,cmap='gray')
axs[2].imshow(invert,cmap='gray')
axs[3].imshow(treshImage2,cmap='gray')
axs[4].imshow(im_gray)
axs[0].set(title="Normal")
axs[1].set(title="Thresh1")
axs[2].set(title="Opening")
axs[3].set(title="Thresh2")
axs[4].set(title="Gray")

# %%
for _ in range(8):
    img, label = it.next()
    print(img.shape)   #  (1,256,256,3)
    plt.imshow(img[_])
    plt.title(np.argmax(preds[_]),color="white")
    plt.tick_params(color="white", labelcolor="white")
    plt.show()

# %%'''
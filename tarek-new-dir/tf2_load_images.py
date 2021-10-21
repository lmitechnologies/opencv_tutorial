#https://www.tensorflow.org/tutorials/load_data/images

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

print(tf.__version__)
import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, fname='flower_photos', untar=True )
data_dir= pathlib.Path(data_dir)
print(data_dir)
#loading path of files

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

roses = list(data_dir.glob('roses/*'))
im=PIL.Image.open(str(roses[0]))
#im.show()

#roses = list(data_dir.glob('roses/*'))
#PIL.Image.open(str(roses[1]))

#Load data using Keras utility
batch_size = 32
img_height = 180
img_width = 180

#80% images for training and 20% for validation
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset='training', seed=123,
    image_size = (img_height, img_width), batch_size = batch_size )
#returns number of files for training

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset = 'validation', seed = 123,
    image_size = (img_height,img_width), batch_size = batch_size)
#returns number for files for validation

class_names = train_ds.class_names
print(class_names)
#class_names attributes on training data set

plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3 ,3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis('off')
    plt.show() #indent of plt.show() does not affect output.

for image_batch,labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break       
#The image_batch is a tensor of the shape (32, 180, 180, 3). 
#This is a batch of 32 images of shape 180x180x3 (the last dimension refers to color channels RGB). 
# The label_batch is a tensor of the shape (32,), these are corresponding labels to the 32 images.

#It is possible to call .numpy() on either of these tensors to convert them to a numpy.ndarray

#Normalizing data

normalization_layer = tf.keras.layers.Rescaling(1./255)
#Two ways of using this layer. First is to apply it to the dataset by calling Dataset.map

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
#next and iter are built in functions of python. Very useful
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))
#Pixel values are now in [0,1]

# If you would like to scale pixel values to [-1,1] you can instead write tf.keras.layers.Rescaling(1./127.5, offset=-1)
# You previously resized images using the image_size argument of tf.keras.utils.image_dataset_from_directory. 
# If you want to include the resizing logic in your model as well, you can use the tf.keras.layers.Resizing layer.

#CONFIGURE DATASET FOR PERFORMANCE - Data prefetching to avoid data loading being a bottleneck

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#TRAIN A MODEL

num_classes = 5

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)

model.summary()

#INSTEAD OF MODEL.FIT() -> https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch






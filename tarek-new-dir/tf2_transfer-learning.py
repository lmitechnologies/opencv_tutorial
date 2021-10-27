#https://www.tensorflow.org/tutorials/images/transfer_learning

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory

#DATA PREPROCESSING
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160,160)

train_dataset = image_dataset_from_directory(
    train_dir,shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
print(type(train_dataset))

validation_dataset = image_dataset_from_directory(
    validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
print(type(validation_dataset))
#type tf.data.dataset

class_names = train_dataset.class_names
plt.figure(figsize=(10,10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis('off')
    plt.show()

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)
#As the original dataset doesn't contain a test set, you will create one. 
#To do so, determine how many batches of data are available in the validation set using 
# tf.data.experimental.cardinality, then move 20% of them to a test set.

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

#CONFIGURE DATASET FOR PERFORMANCE
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

#USE DATA AUGMENTATIONS
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10,10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0]/255) #to normalize
        plt.axis('off')
plt.show()

#RESCALE PIXEL VALUES
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)

#You will create the base model from the MobileNet V2 model developed at Google. 
# This is pre-trained on the ImageNet dataset, a large dataset consisting of 1.4M images and 1000 classes. 
# ImageNet is a research training dataset with a wide variety of categories like jackfruit and syringe. 
# This base of knowledge will help us classify cats and dogs from our specific dataset.

#### CREATE BASE MODEL FROM PRETRAINED MODEL MobileNet V2 ####
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
#First, you need to pick which layer of MobileNet V2 you will use for feature extraction. 
# The very last classification layer (on "top", as most diagrams of machine learning models go from bottom to top) 
# is not very useful. 
# Instead, you will follow the common practice to depend on the very last layer before the flatten operation. 
# This layer is called the "bottleneck layer". 
# The bottleneck layer features retain more generality as compared to the final/top layer.

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch) #creating convolutional base later to use as feature extractor
print(feature_batch.shape)

#### FEATURE EXTRACTION ####
base_model.trainable = False #prevents weights in a entire model's layer from being updated during training.
base_model.summary()

#### ADDING A CLASSIFICATION HEAD AND BUILDING BASE MODEL 'X' ####

#**** To generate predictions from the block of features, average over the spatial 5x5 spatial locations, 
# using a tf.keras.layers.GlobalAveragePooling2D layer to convert the features to a 
# single 1280-element vector per image
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

#Dense layer added to convert these features into a single prediction per image
prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

#Build a model by chaining together the data augmentation, rescaling, base_model and feature extractor layers 
# using the Keras Functional API.
#  As previously mentioned, use training=False as our model contains a BatchNormalization layer.
inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
#Note: Binary crossentropy special case of Categorical cross entropy where m=2

model.summary()
#It shows that The 2.5M parameters in MobileNet are frozen, 
# but there are 1.2K trainable parameters in the Dense layer. 
# These are divided between two tf.Variable objects, the weights and biases. shown below

len(model.trainable_variables)

#### TRAIN THE MODEL ####
initial_epochs = 10
loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

#### LOOKING AT LEARNING CURVES WHEN USING MobileNet V2 base model as a fixed feature extractor ####
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

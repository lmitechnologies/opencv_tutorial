import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import layers

#Load flowers datasets from tensorflow datasets and specify train, validation, and test splits
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:90%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True
)

#Grab and print the number of classes the data is associated with
num_classes = metadata.features['label'].num_classes
print(num_classes)

#Grab the label names
get_label_name = metadata.features['label'].int2str

#Associate each image with their label and plot the first image with its label
image, label = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))
plt.show()

#Define a sequential model with input layers for resizing and rescaling the images
IMG_SIZE = 180
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1./255)
    ])

#Put the image through the model and plot the resulting image
result = resize_and_rescale(image)
_ = plt.imshow(result)
plt.show()
print("Min and max pizel values:", result.numpy().min(), result.numpy().max())

#Define a sequential model that applies random flips and rotation to images (data augmentation)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])

#Add the images to a batch
image = tf.expand_dims(image, 0)
plt.figure(figsize=(10, 10))
for i in range(9):
    augmented_image = data_augmentation(image)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image[0])
    plt.axis("off")
plt.show()

#Define the model with input layers defined above
model = tf.keras.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

#Compile model
model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

train_ds = image.batch(32)
print(image.shape)

'''
epochs = 5
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
loss, acc = model.evaluate(test_ds)
print("Accuracy", acc)
'''

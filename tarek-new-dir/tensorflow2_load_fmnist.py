import tensorflow as tf 
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

np.set_printoptions(linewidth=200)
plt.imshow(train_images[0])
print(train_labels[0])
print(train_images[0])

train_images = train_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(102, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])

predictions = model(train_images[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(train_labels[:1], predictions).numpy()


model.compile(optimizer = tf.optimizers.Adam(), 
    loss=loss_fn, 
    metrics=['accuracy'])

model.fit(train_images,train_labels, epochs = 10)

model.evaluate(test_images, test_labels)


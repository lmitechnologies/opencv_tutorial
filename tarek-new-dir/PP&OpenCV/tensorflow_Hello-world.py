import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
#Model has 1 layer and the layer has 1 neuron and input shape to is just 1 value

model.compile(optimizer='sgd',loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys=np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))

#https://www.coursera.org/learn/introduction-tensorflow/lecture/tVvjQ/writing-code-to-load-training-data
import tensorflow as tf
import numpy as np

from keras_models import build_model 
import argparse 

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def main():

  gpus = get_available_gpus()
  print('available devices',gpus)
  model = build_model(name = 'shallow',
                      num_classes = 10,
                      frozen_layers = (0, -4),
                      input_shape = (784,),
                      base_learning_rate = 0.0001)
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

  # Preprocess the data (these are Numpy arrays)
  x_train = x_train.reshape(60000, 784).astype('float32') / 255
  x_test = x_test.reshape(10000, 784).astype('float32') / 255

  y_train = y_train.astype('float32')
  y_test = y_test.astype('float32')

  # Reserve 10,000 samples for validation
  x_val = x_train[-10000:]  
  y_val = y_train[-10000:]
  x_train = x_train[:-10000]
  y_train = y_train[:-10000]

  history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=3,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=(x_val, y_val))

if __name__ == "__main__":
  main()
    


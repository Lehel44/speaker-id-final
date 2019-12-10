import tensorflow as tf
from keras.models import load_model
import os
import numpy as np

# Mute excessively verbose Tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

conv_encoder_path = '/home/ubuntu/speaker-id/model/conv_encoder.hdf5'
input_1_path = '/home/ubuntu/speaker-id/data/examples/input_1.npy'

# Load model.
model = load_model(conv_encoder_path, compile=False)

# Load inputs
input_1 = np.load(input_1_path)

result = model.predict(input_1)

print(result)


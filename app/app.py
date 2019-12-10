# -*- coding: utf-8 -*-
import httplib
import math
import os
import uuid

import numpy as np
import tensorflow as tf
from flask import Flask, request, Response, json

import paths
from app.compute import compare_vectors
from app.util import check_audio_file, average_vectors, update_dict, get_name_by_id, \
    get_password
from model_loader import ModelLoader
from preprocess import preprocess

# Flask app.
app = Flask(__name__)

# Load configuration.
app.config.from_object('config.Config')

# Mute excessively verbose Tensorflow output.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Load models.
model_loader = ModelLoader()


@app.route('/enroll', methods=['POST'])
def enroll():
    if check_audio_file(app, request, 'audio_file'):
        user_id = str(uuid.uuid4())
        user_name = request.form.get('username')
        password = request.form.get('password')
        update_dict(paths.PASSWORDS, user_id, password)
        audio_file = request.files['audio_file']
        preprocessed_audio_file = preprocess(audio_file, 3, 2)
        voice_vector = model_loader.predict(preprocessed_audio_file, 'conv_encoder')
        np.save(paths.VECTORS + user_id, voice_vector)
        # Update user dict.
        update_dict(paths.USER_DICT, user_id, user_name)
    return Response(str(200), mimetype='text/plain')


@app.route('/identify', methods=['POST'])
def identify():
    if check_audio_file(app, request, 'audio_file'):
        audio_file = request.files['audio_file']
        preprocessed_audio_file = preprocess(audio_file, 3, 2)
        voice_vector = model_loader.predict(preprocessed_audio_file, 'conv_encoder')
        user_name, user_id, min_value = compare_vectors(app, voice_vector)
        app.logger.info('User name: ' + user_name + ' | min value: ' + str(min_value))
        if math.isnan(min_value) or min_value > 20:
            # Store voice vector for later average, if authentication succeed.
            np.save(paths.VECTORS_TO_AVERAGE + user_id, voice_vector)
            response = Response(status=httplib.FORBIDDEN, mimetype='text/plain')
            response.headers['user_id'] = user_id
            return response
        # If similar enough, then average vectors.
        average_vectors(user_id)
        response = Response(status=httplib.ACCEPTED, mimetype='text/plain')
        response.headers['user_id'] = user_id
        return response
    return Response(status=httplib.BAD_REQUEST, mimetype='text/plain')


@app.route('/authenticate', methods=['POST'])
def authenticate():
    password_from_req = request.form.get('password')
    user_id = request.form.get('user_id')
    real_password = get_password(user_id)
    if real_password == password_from_req:
        average_vectors(user_id, np.load(paths.VECTORS_TO_AVERAGE + user_id + '.npy'))
        response_body = {'user_id': user_id, 'user_name': get_name_by_id(user_id)}
        return Response(response=json.dumps(response_body), status=httplib.ACCEPTED, mimetype='text/plain')
    # Delete stored voice vector.
    os.remove(paths.VECTORS_TO_AVERAGE + user_id + '.npy')
    return Response(status=httplib.FORBIDDEN, mimetype='text/plain')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

# -*- coding: utf-8 -*-
import os
import pickle
import uuid

import numpy as np
import logging
from numpy import dot
from numpy.linalg import norm

import paths


def allowed_file(app, filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def count_distance(v1, v2, distance='euclidean'):
    if distance == 'euclidean':
        return np.linalg.norm(v1 - v2)
    elif distance == 'cosine':
        return dot(v1, v2) / (norm(v1) * norm(v2))
    else:
        logging.info('Unrecognized distance.')
        raise ValueError('Unrecognized distance.')


def check_audio_file(app, request, file_id):
    if file_id not in request.files:
        return False
    file = request.files[file_id]
    if file and allowed_file(app, file.filename):
        return True
    return False


def save_audio_file(request, file_id, path):
    username = request.form.get('username')
    user_id = str(uuid.uuid4())
    file = request.files[file_id]
    file.save(os.path.join(path, user_id + '.wav'))
    return user_id, username, file


def average_vectors(user_id, new_vector):
    orig_vector = np.load(paths.VECTORS + user_id + '.npy')
    average_vector = (orig_vector + new_vector) / 2
    np.save(paths.VECTORS + user_id, average_vector)
    os.remove(paths.VECTORS_TO_AVERAGE + user_id + '.npy')


def update_dict(dict_path, key, value):
    user_dict = dict()
    if os.path.isfile(dict_path):
        with open(dict_path, 'rb') as handle:
            user_dict = pickle.load(handle)
    user_dict[key] = value
    with open(dict_path, 'wb') as handle:
        pickle.dump(user_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_name_by_id(key):
    with open(paths.USER_DICT, 'rb') as handle:
        user_dict = pickle.load(handle)
        return user_dict[key]


def write_dict_to_file(path, user_dict):
    with open(path, 'wb') as handle:
        pickle.dump(user_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_password(key):
    with open(paths.PASSWORDS, 'rb') as handle:
        password_dict = pickle.load(handle)
        return password_dict[key]

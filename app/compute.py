import os
from os import walk

import numpy as np

from app import paths
from app.util import count_distance, update_dict, write_dict_to_file, get_name_by_id

def compare_vectors(app, current_vector, distance='euclidean'):
    distance_map = dict()
    file_names = []
    for (_, _, names) in walk(paths.VECTORS):
        file_names.extend(names)
        break
    if file_names:
        for file_name in file_names:
            other_vector = np.load(paths.VECTORS + file_name)
            distance_map[os.path.splitext(file_name)[0]] = count_distance(current_vector, other_vector, distance)
    app.logger.info(distance_map)
    min_user_id = min(distance_map, key=distance_map.get)
    min_value = distance_map[min_user_id]
    user_name = get_name_by_id(min_user_id)
    return user_name, min_user_id, min_value

def update_distances(current_user_id, current_vector, distance='euclidean'):
    current_user_dict = dict()
    file_names = []
    for (_, _, names) in walk(paths.VECTORS):
        file_names.extend(names)
        break
    if file_names:
        for file_name in file_names:
            other_vector = np.load(paths.VECTORS + file_name)
            # Update current user dict.
            dist = count_distance(current_vector, other_vector, distance)
            current_user_dict[os.path.splitext(file_name)[0]] = dist
            # Update other dict too and save to file.
            update_dict(paths.DISTANCES + os.path.splitext(file_name)[0], current_user_id, dist)
    # Write current user dict to file.
    write_dict_to_file(paths.DISTANCES + current_user_id, current_user_dict)
    # Save current vector to file.
    np.save(paths.VECTORS + current_user_id, current_vector)
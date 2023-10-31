from multiprocessing import Pool
from functools import partial
import inspect
import mediapipe as mp
from PIL import Image
import numpy as np
import pandas as pd
import os
import cv2
from rembg import remove


# For finding pose cords
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.0,
    min_tracking_confidence=0.0)

def path_to_img(img_path, img_size, flip_image, remove_background=False):
    img = Image.open(img_path)
    img = img.resize(img_size)
    img = np.array(img)

    if img.shape != (img_size[0], img_size[1], 3):
        img = np.array([img, img, img])
        img = img.reshape(img_size[0], img_size[1], 3)
    if remove_background:
        img = remove(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if flip_image:
        img = np.fliplr(img)
    return img


def image_to_pose_cords(img):
    results = pose_detector.process(np.array(img))
    if results.pose_landmarks is None:
        array = np.zeros((33, 4))
    else:
        a = results.pose_landmarks.ListFields()[0][1]
        array = []
        for val in a:
            array.append([val.x, val.y, val.z, val.visibility])
    
    np.expand_dims(array, -1)
    return np.array(array)


def path_to_pose(path: str, img_size, flip_image=False, remove_background=False):
    img = path_to_img(path, img_size, flip_image, remove_background)
    pose = image_to_pose_cords(img)
    return img, pose


def process_namepose(dataset_item, data_dir, img_size, remove_background):
    name, pose = dataset_item
    img_path = os.path.join(data_dir, name)
    img1, cords1 = path_to_pose(img_path, img_size, remove_background=remove_background)
    img1 = Image.fromarray(img1)
    img1.save('unsorted/' + str(pose) + '_' + name)
    img1 = None
    return pose, cords1


def process_namepose_val(dataset_item, data_dir, img_size):
    name, pose = dataset_item
    img_path = os.path.join(data_dir, name)
    img1, cords1 = path_to_pose(img_path, img_size)
    return img1, cords1


from concurrent.futures import ProcessPoolExecutor
def get_train_ds(dataset, data_dir, img_size, remove_background):
    num_threads = 12
    tasks = []
    pool = ProcessPoolExecutor(num_threads)

    for i in range(len(dataset)):
        tasks.append(pool.submit(process_namepose, dataset[i], data_dir, img_size, remove_background))

    outputs = []
    for i in range(len(tasks)):
        if i%10 == 0:
            print(i)
        outputs.append(tasks[i].result())

    return outputs

def load_ds_from_folder(dataset, data_dir, img_size, remove_background):
    num_threads = 12
    tasks = []
    pool = ProcessPoolExecutor(num_threads)

    for i in range(len(dataset)):
        tasks.append(pool.submit(process_namepose, dataset[i], data_dir, img_size, remove_background))

    outputs = []
    for i in range(len(tasks)):
        if i%10 == 0:
            print(i)
        outputs.append(tasks[i].result())

    return outputs


def get_val_ds(dataset, data_dir, img_size):
    num_threads = 12
    tasks = []
    pool = ProcessPoolExecutor(num_threads)

    for i in range(len(dataset)):
        tasks.append(pool.submit(process_namepose_val, dataset[i], data_dir, img_size))

    outputs = []
    for task in tasks:
        outputs.append(task.result())

    return outputs

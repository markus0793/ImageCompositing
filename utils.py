import functools
import os
import random
from multiprocessing import current_process

import numpy as np
import cv2
import pickle as pkl

from color_matcher import ColorMatcher
from color_matcher.normalizer import Normalizer


def save_objects_pickel(path, name, objects):
    full_path = f"{path}{name}.pkl"
    with open(full_path, 'wb') as file:
        pkl.dump(objects, file, pkl.HIGHEST_PROTOCOL)


def load_objects_pickel(path, name):
    full_path = f"{path}{name}.pkl"
    if os.path.exists(full_path):
        with open(full_path, 'rb') as file:
            objects = pkl.load(file)
            return objects
    return []


def save_data_pickel(path, name, objects):
    full_path = f"{path}{name}.pkl"
    with open(full_path, 'wb') as file:
        pkl.dump(objects, file, pkl.HIGHEST_PROTOCOL)


def load_data_pickel(path, name):
    full_path = f"{path}{name}.pkl"
    if os.path.exists(full_path):
        with open(full_path, 'rb') as file:
            objects = pkl.load(file)
            return objects
    return []


def extract_classes_LBL(img_lbl: np.ndarray, n_classes=10):
    """
    Extract n_classes example(1,2,3,4) from an segmentation image
    :param img_lbl: segmentation Image with n_classes
    :param n_classes: number of classes the image has.
    :return: n
    """
    classes = []
    for i in range(1, n_classes):
        low = np.array([i, i, i])
        high = np.array([i, i, i])
        img_gray = cv2.inRange(img_lbl, low, high)
        # make to binary image
        # (thresh, mask) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        # prepare the 5x5 shaped filter
        mask = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
        if np.sum(mask) < 4000:
            continue
        classes.append(mask)
    return classes


def extract_classes_REAL(img_real: np.ndarray, LBL_classes):
    img_real_c = []
    for mask in LBL_classes:
        img = cv2.bitwise_and(img_real, img_real, mask=mask)
        img_real_c.append(img)
    return img_real_c


def extract_contours(img, type="tight"):
    """
    extract edges form pictures
    :param img: RGB picture
    :return: binary picture with ages
    """
    sharpen_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, sharpen_filter)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    thresh, edges = cv2.threshold(gray, np.mean(gray) + 20, 255, cv2.THRESH_TOZERO | cv2.THRESH_BINARY)
    edges = cv2.Canny(edges, thresh, 220, apertureSize=3)

    edges = cv2.dilate(edges, np.ones((7, 7), np.uint8))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, (7, 7))

    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, (5, 5))
    # cv2.imwrite(f"./data/objects/{type}mid_test.png", edges)
    return edges


def crop_to_box(mask, real, box):
    max_x = mask.shape[1]
    max_y = mask.shape[0]
    x, y, w, h = box
    w4 = int(np.floor(w / 3))
    h4 = int(np.floor(h / 3))
    w = np.minimum(max_x, x + w + 2 * w4)
    h = np.minimum(max_y, y + h + 2 * h4)
    if w != max_x: w = w - x
    if h != max_y: h = h - y
    x = np.maximum(0, x - w4)
    y = np.maximum(0, y - h4)
    mask = mask[y:y + h, x:x + w]
    real = real[y:y + h, x:x + w]
    box = (x, y, w, h)
    return mask, real, box


def extract_contours_masks(img_real: np.ndarray, real_classes, max_masks=40):
    """
    extract masks from selected image parts
    :param img_real: The origin image
    :param real_classes: slected segments of origin image
    :param max_masks: max extracted masks
    :return: max extraced mask and contours ( mask, boundingbox(x,y,w,h)
    """
    cnts = []
    for i, real in enumerate(real_classes):
        edges = extract_contours(real, str(i))
        cnt = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        #print(i, " ", len(cnt))
        cnts.extend(cnt)

    print(len(cnts))

    cnts = sorted(cnts, key=cv2.contourArea)[::-1]

    masks_lbl = []
    masks_rgb = []
    boxs = []
    max_masks = np.minimum(max_masks, len(cnts))
    cnts = cnts[:max_masks]
    low = np.array([1, 1, 1])
    high = np.array([255, 255, 255])
    i = 0
    for c in cnts:
        # box
        box = cv2.boundingRect(c)
        # mask
        mask = np.zeros_like(img_real, np.uint8)
        mask = cv2.fillPoly(mask, pts=[c], color=(255, 255, 255))
        mask = cv2.drawContours(mask, [c], -1, (255, 255, 255))
        cv2.floodFill(mask, None, c[0][0], (255, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (5, 5))
        mask = cv2.inRange(mask, low, high)
        # print(mask)
        # rgb
        real = cv2.bitwise_and(img_real, img_real, mask=mask)
        # crop
        mask, real, box = crop_to_box(mask, real, box)
        # check if it usefull size
        if (np.sum(mask) / 255) < box[2] * box[3] / 8:
            continue
        if box[2] < 150 or box[3] < 150:
            continue
        # append
        boxs.append(box)
        masks_lbl.append(mask)
        masks_rgb.append(real)
        # cv2.imwrite(f"./data/objects/{i:00d}_test.png", real)
        i += 1
    return (masks_lbl, masks_rgb, boxs)


def extract_masks_mp(img_dir: str, filename):
    """
    used to give to a worker or directly.
    Can also be used normally gives filename with (lbl, real) file and extraxt mask
    :param img_dir: Directory of the images
    :param filename: file name of the image (lbl, real)
    :return: masks, reals, box
    """
    LBL, REAL = filename
    print(f"Start: {filename}")
    img_lbl = cv2.imread(f"{img_dir}{LBL}")
    img_real = cv2.imread(f"{img_dir}{REAL}")
    lbl_classes = extract_classes_LBL(img_lbl, np.max(img_lbl))
    real_classes = extract_classes_REAL(img_real, lbl_classes)
    masks, reals, box = extract_contours_masks(img_real, real_classes)
    print(f"Finished: {filename}")
    return (masks, reals, box)


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    # print(result.shape)
    return result


def insert_stuff_randomly(img_path, stuff, img_size):
    img = cv2.imread(img_path)
    print("Insert Random:")
    resized = cv2.resize(img[700:img.shape[1]-400, 100:img.shape[0]-100],(224,224))
    img_mask = np.zeros_like(img[:, :, 0], np.uint8)
    img_real = np.copy(img)
    cm = ColorMatcher()
    place_area_x_left, place_area_x_right = 0, int(img.shape[1] / 3)
    isleft = random.choice([True, False])
    random_x_min = place_area_x_left if isleft else place_area_x_right
    random_x_max = int(place_area_x_right) if isleft else img.shape[1] - 1
    for mask, real, box in stuff:
        x, y, w, h = box
        rotate_degree = random.randint(0, 360)
        mask = rotate_image(mask, rotate_degree)
        real = rotate_image(real, rotate_degree)
        random_x = random.randint(random_x_min, random_x_max)
        random_y = random.randint(400, img.shape[0] - 800)
        x = np.maximum(0, random_x if random_x + w < img.shape[1] else img.shape[1] - (w + 1))
        y = np.maximum(0, random_y if random_y + h < img.shape[0] else img.shape[0] - (h + 1))
        if w >= img.shape[1] or h >= img.shape[0]: continue
        real = cm.transfer(real, resized, method='mvgd')
        real = Normalizer(real).uint8_norm()
        img_mask[y:y + h, x:x + w] = np.where(mask > 0, 1, img_mask[y:y + h, x:x + w])
        mask_3 = np.expand_dims(mask, 2)
        mask_3 = np.where(mask_3 > 0, False, True)
        img_real[y:y + h, x:x + w, :] = np.where(np.repeat(mask_3, 3, axis=2), img_real[y:y + h, x:x + w, :], real)
    print(" FINISHED")
    #cv2.imwrite(f"./data/objects/{img_path[-10:]}_real.png", img_real)
    #img_mask = cv2.inRange(img_mask, 1, 255)
    #cv2.imwrite(f"./data/objects/{img_path[-10:]}_mask.png", img_mask)
    img = cv2.GaussianBlur(img, (7, 7), 0)
    img_real = cv2.GaussianBlur(img_real, (7, 7), 0)
    img = cv2.resize(img,img_size)
    img_real = cv2.resize(img_real, img_size)
    img_mask = cv2.resize(img_mask, img_size)

    return {"change_img": (img, img_real), "mask": img_mask}


# workers util
def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = calculate(func, args)
        output.put(result)


def calculate(func, args):
    result = func(*args)
    return result

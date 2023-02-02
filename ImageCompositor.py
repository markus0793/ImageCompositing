import math
import os
import random

import cv2
import numpy as np
import pickle as pkl

def segment_image_edges(img, img_real=None, name=""):
    sharpen_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, sharpen_filter)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
    edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)

    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[::-1]
    masked = None
    if len(cnt) > 1:
        if len(cnt) > 50:
            cnt = cnt[0:50]
        print(f"Number of contours:{str(len(cnt))}")
        masks = []
        for i, c in enumerate(cnt):
            mask = np.zeros_like(img, np.uint8)
            masked = cv2.drawContours(mask, [c], -1, 255, -1)
            cv2.floodFill(masked, None, c[0][0], (255, 255, 255))
            low = np.array([1, 1, 1])
            high = np.array([255, 255, 255])
            masked = cv2.inRange(masked, low, high)
            masks.append(masked)
        return masks, cnt
    return [], []

def segment_image_color( img, color="green"):
    """
    Segment the image to extract the object using computer vision techniques
    """
    gray_img = img
    img = pre_precessing_img(img)

    if color == "green":
        low = np.array([60, 80, 60])
        high = np.array([90, 255, 255])
    elif color == "orange":
        low = np.array([100, 110, 60])
        high = np.array([110, 255, 255])
    elif color == "red":
        low = np.array([120, 100, 60])
        high = np.array([135, 255, 255])
    elif color == "gray":
        low = np.array([0, 0, 0])
        high = np.array([255, 255, 255])
    mask = []
    if color == "gray":
        for i in range(1, 30):
            low = np.array([i, i, i])
            high = np.array([i, i, i])
            masked = cv2.inRange(gray_img, low, high)
            masked = post_processing_img(masked)
            # cv2.imwrite(f"color_{color}_{i}.jpg", masked)
            mask.append(masked)
    else:
        masked = cv2.inRange(img, low, high)
        masked = post_processing_img(masked)
        cv2.imwrite(f"color_{color}.jpg", masked)
        mask.append(masked)
    return mask

def pre_precessing_img( img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    img = cv2.GaussianBlur(img, (25, 25), 0)
    cv2.imwrite('Blurred Image.jpg', img)
    return img

def extract_object( img, mask, name):
    """
    Multiply the mask with the image to extract the object
    """
    img = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imwrite(f"object{name}.jpg", img)
    return img

def post_processing_img( img):
    kernel = np.ones((5, 5), np.uint8)
    # prepare the 5x5 shaped filter
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # Applying bilateral filter to the image
    return img

def opening_img( img):
    kernel = np.ones((5, 5), np.uint8)
    # prepare the 5x5 shaped filter
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # Applying bilateral filter to the image
    return img

def closing_img( img):
    kernel = np.ones((3, 3), np.uint8)
    # prepare the 5x5 shaped filter
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # Applying bilateral filter to the image
    return img


def separet_gray_scale(img, real):
    segment_image_edges(img, "edge_grey")
    masks_grey = segment_image_color(img, "gray")
    all_good_objets_mask_real = {
                                 "crop_objects": []}
    for i, mask_grey in enumerate(masks_grey):
        mask_grey = closing_img(mask_grey,)
        #mask_grey = cv2.GaussianBlur(mask_grey, (11, 11), 0)
        mask_grey = cv2.bilateralFilter(mask_grey, 9, 75, 75)
        mask_all_obj = extract_object(real, mask_grey, f"objects_{i}")
        objectes_mask, cnt = segment_image_edges(mask_all_obj, f"object_edges_{i}")
        for i, mask_single_object in enumerate(objectes_mask):
            x, y, w, h = cv2.boundingRect(cnt[i])
            if w < 155 and h < 155:
                continue
            elif w < 75:
                continue
            elif h < 75:
                continue
            object_real = extract_object(real, mask_single_object, "")
            cv2.imshow("preview", object_real)
            k = cv2.waitKey(0)
            if k == ord("y"):
                mid_point = (x + math.floor((w / 2)), y + math.floor((h / 2)))
                crop_mask_object = objectes_mask[i][y:y + h, x:x + w]
                crop_real_object = object_real[y:y + h, x:x + w]
                all_good_objets_mask_real["crop_objects"].append((mid_point, crop_mask_object, crop_real_object))
    return all_good_objets_mask_real

def choose_trash_objects(image_dir, image_n):
    image_n = math.floor(image_n * 2)
    files = sorted(os.listdir(image_dir))
    image_n = np.maximum(np.minimum(math.floor(len(files)/ 2), image_n), 2)
    files = files[:image_n]
    cv2.startWindowThread()
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('preview', 1500, 1300)
    all_objects = {"crop_objects": [],}
    for i in range(1, len(files), 2):
        print(files[i], files[i - 1])
        img = cv2.imread(f"{image_dir}{files[i - 1]}")
        img_real = cv2.imread(f"{image_dir}{files[i]}")
        objects = separet_gray_scale(img, img_real)
        #save_objects_with_masks(i, objects)
        all_objects["crop_objects"].extend(objects["crop_objects"])
    cv2.destroyAllWindows()
    return all_objects
def save_objects_pickel(path,name, objects):#
    full_path = f"{path}{name}.pkl"
    with open(full_path, 'wb') as file:
        pkl.dump(objects, file, pkl.HIGHEST_PROTOCOL)

def load_objects_pickel(path,name):
    full_path = f"{path}{name}.pkl"
    if os.path.exists(full_path):
        with open(full_path, 'rb') as file:
            objects = pkl.load(file)
            return objects
    return []

def insert_object_randomly(img, objects):
    pass

def create_random_data(image_dir, objects, quantity):
    rgb_images_names = sorted(filter(lambda x: x.endswith("RGB.png"), os.listdir(image_dir)))
    #images = [cv2.imread(f"{image_dir}{file}") for file in files]
    crop_objects = objects["crop_objects"]


    for i in range(quantity):
        random_rgb_name = rgb_images_names[random.randint(0,len(rgb_images_names))]
        random_image = cv2.imread(f"{image_dir}{random_rgb_name}")
        random_objects_n = random.randint(0,np.minimum(23,len(crop_objects)))
        random_objects_indexs = random.choices(range(0, len(crop_objects)),k=20)
        random_objects = [crop_objects[index] for index in random_objects_indexs]
        image_data = insert_object_randomly(random_image,random_objects)



if __name__ == "__main__":
    image_dir_path = "./data/pack200/"
    pickle_path = "./data/objects/"
    pkl_name = "test"
    object_dict = load_objects_pickel(pickle_path,pkl_name)
    if not object_dict:
        object_dict = choose_trash_objects(image_dir_path, 1)
        save_objects_pickel(pickle_path, pkl_name, object_dict)
    create_random_data(image_dir_path,object_dict,2)
    #print(object_dict)











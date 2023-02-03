import math
import os
import random

import cv2
import numpy as np
import pickle as pkl
from color_matcher import ColorMatcher
from color_matcher.normalizer import Normalizer


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
            #cv2.imshow("preview", object_real)
            k = cv2.waitKey(0)
            if k == ord("y"):
                bound_rect_corners = (y,x,h,w)
                crop_mask_object = objectes_mask[i][y:y + h, x:x + w]
                crop_real_object = object_real[y:y + h, x:x + w,:]
                all_good_objets_mask_real["crop_objects"].append((bound_rect_corners, crop_mask_object, crop_real_object))
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
def save_objects_pickel(path,name, objects):
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

def save_data_pickel(path,name, objects):
    full_path = f"{path}{name}.pkl"
    with open(full_path, 'wb') as file:
        pkl.dump(objects, file, pkl.HIGHEST_PROTOCOL)

def load_data_pickel(path,name):
    full_path = f"{path}{name}.pkl"
    if os.path.exists(full_path):
        with open(full_path, 'rb') as file:
            objects = pkl.load(file)
            return objects
    return []

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  #print(result.shape)
  return result

def insert_object_randomly(img, objects):
    img_mask = np.zeros_like(img[:,:,0], np.uint8)
    img_real = img
    cm = ColorMatcher()
    for cord, mask, real in objects:
        x, y, w, h = cord
        rotate_degree = random.choice([ random.randint(0,15), random.randint(165,195), random.randint(345,360)])
        mask = rotate_image(mask,rotate_degree)
        real = rotate_image(real, rotate_degree)
        place_area_x, place_area_y  = 200, 700
        if w > 1800 or h > 1300: continue
        place_w_x, place_w_y = np.maximum(img.shape[0]-w,place_area_x), np.maximum(img.shape[1]-h,place_area_y)
        x = random.randint(place_area_x,place_w_x)
        y = random.randint(place_area_y,place_w_y)
        #real = cm.transfer(real,img[750:1600, 260:2000,:], method='mvgd')
        real = cm.transfer(real,img[x:x+w, y:y+h,:], method='mvgd')
        real = Normalizer(real).uint8_norm()
        #show_image(mask)
        img_mask[x:x+w, y:y+h] = np.where(mask > 0, 1, img_mask[x:x+w, y:y+h])
        mask_3 = np.expand_dims(mask, 2)
        mask_3 = np.where(mask_3 > 0, False,True)
        img_real[x:x+w, y:y+h,:] = np.where(np.repeat(mask_3,3,axis=2), img_real[x:x+w, y:y+h,:],real)
    #show_image(img_real)
    #show_image(img_mask)
    global image_shape
    img_mask = cv2.resize(img_mask, image_shape)
    img_real = cv2.resize(img_real, image_shape)
    #show_image(img_real)
    return img_mask, img_real

def show_image(img):
    cv2.startWindowThread()
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('preview', 1500, 1300)
    cv2.imshow("preview",img)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_random_data(image_dir, objects, quantity):
    rgb_images_names = sorted(filter(lambda x: x.endswith("RGB.png"), os.listdir(image_dir)))
    #images = [cv2.imread(f"{image_dir}{file}") for file in files]
    crop_objects = objects["crop_objects"]
    image_datas = {"change_img": [],
                   "mask": [] }
    rand_images = random.choices(rgb_images_names, k=quantity)
    for i, name in enumerate(rand_images):
        random_image = cv2.imread(f"{image_dir}{name}")
        random_objects_n = random.randint(0,np.minimum(10,len(crop_objects)))
        random_objects = random.choices(crop_objects,k=random_objects_n)
        real_first = cv2.resize(random_image,image_shape)
        mask, real_second = insert_object_randomly(random_image,random_objects)
        mask_3 = np.repeat(np.expand_dims(mask, 2), 3, axis=2)
        im_h = cv2.hconcat([real_first, real_second, mask_3])
        real_first = cv2.GaussianBlur(real_first, (7,7),0)
        real_second = cv2.GaussianBlur(real_second, (7,7),0)
        # show the output image
        #show_image(im_h)
        image_datas["change_img"].append((real_first, real_second))
        image_datas["mask"].append(mask)
    return image_datas

image_shape = (224, 224)

if __name__ == "__main__":
    image_dir_path = "./data/pack200/"
    pickle_path = "./data/objects/"
    pkl_name = "all"
    dataset = "train_set"
    object_dict = load_objects_pickel(pickle_path,pkl_name)
    if not object_dict:
        object_dict = choose_trash_objects(image_dir_path, 399)
        save_objects_pickel(pickle_path, pkl_name, object_dict)
    data = load_data_pickel(pickle_path, dataset)
    if not data:
        data = create_random_data(image_dir_path,object_dict,8)
        save_objects_pickel(pickle_path,dataset,data)
    mask = data["mask"]
    ha = data["change_img"]
    for i, (left, right) in enumerate(ha):
        cv2.imwrite(f"./data/Imag_data/{i:02}_0_right.png",right)
        cv2.imwrite(f"./data/Imag_data/{i:02}_0_left.png", left)
        cv2.imwrite(f"./data/Imag_data/{i:02}_1_mask.png", mask[i])

    #print(object_dict)











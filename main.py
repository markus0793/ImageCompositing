import cv2
import os

from DatasetCreator import DatasetCreator
from utils import extract_contours

image_shape = (224, 224)

if __name__ == "__main__":
    image_dir_path = "./data/pack200/"
    pickle_path = "./data/objects/"
    pkl_name = "all"
    dataset = "train_set"
    img = cv2.imread(f"{image_dir_path}001_001_RGB.png")
    files = sorted(os.listdir(image_dir_path))
    rgb_files = sorted(filter(lambda x: x.endswith("RGB.png"), os.listdir(image_dir_path)))
    lbl_files = sorted(filter(lambda x: x.endswith("LBL.png"), os.listdir(image_dir_path)))
    filenames = list(zip(*[lbl_files, rgb_files]))
    dc = DatasetCreator(image_dir_path, filenames, "./data/saves/")
    dc.img_number()
    dc.load_stuff("./data/objects/", "test.pkl")
    if not dc.stuff:
        dc.extract_stuff(num_workers=4)
        dc.save_stuff("./data/objects/", "test.pkl")
    dc.create_Dataset(num_workers=4, data_set_size=40)

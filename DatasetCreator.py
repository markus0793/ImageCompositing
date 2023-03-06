import datetime
import os
import pickle as pkl
from multiprocessing import Queue, Process

import cv2
import numpy as np
import multiprocessing.dummy as mp

from utils import *


class DatasetCreator:
    """
    Create a random Dataset from Segmentation mask image and Real image
    """
    human = False

    img_dir = "./"
    img_filenames = []
    save_dir = "./"
    n_img = 1

    def __init__(self, img_dir: str, img_filenames, save_dir: str, human=False, filter=None):
        """
        Init Dataset Images and chooses filter and human interaction
        :param img_dir: Location of the img_files with respect to workDir
        :param img_filenames: List of tuples (LBL_file_name, RGB_file_name)
        :param save_dir: Directory to save object and Datasets
        :param human: Enable interaction to choose potential dataset objects
        :param filter: Special modes to try to get better results
        """
        self.stuff = None
        self.img_filenames = img_filenames
        self.img_dir = img_dir
        self.save_dir = save_dir
        self.human = human

    def img_number(self, n=-1):
        """
        Selects how many images from the img_filenames should be used.
        if not set all will be used
        :param n: Numbers of images pair to be loaded (-1 for all)
        """
        if n == -1:
            self.n_img = len(self.img_filenames)
        else:
            self.n_img = np.maximum(np.minimum(n, len(self.img_filenames)), 1)

    def extract_stuff_mp(self, num_workers, files):
        """
        Extract stuff automatically optimised with multithreading

        :param num_workers: number of workers running
        :param files: Segmentation and real file dict
        :return: list of tuples with (masks, reals, box)
        """
        task_queue = Queue()
        done_queue = Queue()
        tasks = [(extract_masks_mp, (self.img_dir, filename)) for filename in files]
        # Submit tasks
        for task in tasks:
            task_queue.put(task)
        # Start worker processes
        pros = [Process(target=worker, args=(task_queue, done_queue)) for _ in range(num_workers)]
        for pro in pros:
            pro.start()
        # Get and print results
        done = []
        for i in range(len(tasks)):
            done.extend(list(zip(*done_queue.get())))
        for i in range(num_workers):
            task_queue.put("STOP")
        for pro in pros:
            pro.join()
        return done

    def create_random_images_mp(self, num_workers, files, stuff, img_size):
        """
        Create random images Multi processing
        :param num_workers: number of workers
        :param files: list of tuples with filenames
        :param stuff: list of stuff
        :param img_size: image size to compress
        :return: list of {"change_img": (img, img_real), "mask": img_mask}
        """
        task_queue = Queue()
        done_queue = Queue()
        tasks = []
        for filename in files:
            random_number = np.random.randint(0, np.minimum(len(stuff), 30))
            random_stuff = random.choices(stuff, k=random_number)
            tasks.append((insert_stuff_randomly, (f"{self.img_dir}{filename}", random_stuff, img_size)))
        # Submit tasks
        for task in tasks:
            task_queue.put(task)
        # Start worker processes
        pros = [Process(target=worker, args=(task_queue, done_queue)) for _ in range(num_workers)]
        for pro in pros:
            pro.start()
        # Get and print results
        done = []
        for i in range(len(tasks)):
            done.append(done_queue.get())
        for i in range(num_workers):
            task_queue.put("STOP")
        for pro in pros:
            pro.join()
        return done

    def extract_stuff(self, num_workers=1):
        """
        Extract stuff and saves it in the class for future use
        :param num_workers: Count of workers used
        """
        files = self.img_filenames[:self.n_img]

        all_objects = {"crop_objects": [], }
        stuff = []
        if num_workers == 1:
            for filename in files:
                stuff.extend(list(zip(*extract_masks_mp(self.img_dir, filename))))
        else:
            stuff = self.extract_stuff_mp(num_workers, files)
        if self.human:
            cv2.startWindowThread()
            cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('preview', 1500, 1300)
            cv2.destroyAllWindows()
        self.stuff = stuff

    def create_Dataset(self, train_set_p=0.60, verify_set_p=0.20, test_set_p=0.20, num_workers=1, data_set_size=10,
                       r_seed=10, img_size=(224, 224)):
        """
        Takes the already loaded stuff and create a train, verify and test set with it
        No background image and stuff is twice in this sets
        (it could be because there a no background images without stuff in it (images missing))
        :param train_set_p: % of train dataset form data_set_size (train_set_p + verify_set_p + test_set_p = 1)
        :param verify_set_p: % of verify dataset form data_set_size (train_set_p + verify_set_p + test_set_p = 1)
        :param test_set_p: % of test dataset form data_set_size (train_set_p + verify_set_p + test_set_p = 1)
        :param num_workers: numbers of workes
        :param data_set_size: How many images all 3 set schould have overall
        :param r_seed: random seed
        :param img_size: Dataset image size to compress data
        :return: saves data so safe_dir
        """
        # backgrounds
        np.random.seed(r_seed)
        img_rgb = list(list(zip(*self.img_filenames))[1])
        np.random.shuffle(img_rgb)
        b_size = len(img_rgb)
        sb_train, sb_verify, sb_test = int(b_size * train_set_p), int(b_size * verify_set_p), int(b_size * test_set_p)
        background = {"train": img_rgb[:sb_train],
                      "test": img_rgb[sb_train:sb_train + sb_verify],
                      "verify": img_rgb[sb_train + sb_verify:sb_train + sb_verify + sb_test]}
        s_size = len(self.stuff)
        ss_train, ss_verify, ss_test = int(s_size * train_set_p), int(s_size * verify_set_p), int(s_size * test_set_p)
        np.random.shuffle(self.stuff)

        stuff = {"train": self.stuff[:ss_train],
                 "test": self.stuff[ss_train:ss_train + ss_verify],
                 "verify": self.stuff[ss_train + ss_verify:ss_train + ss_verify + ss_test]}
        sizes = {"train": int(np.round(data_set_size * train_set_p)),
                 "test": int(np.round(data_set_size * test_set_p)),
                 "verify": int(np.round(data_set_size * verify_set_p))}
        data = {"train": [], "test": [], "verify": []}
        for key in data.keys():
            backgrounds = np.random.choice(background[key], sizes[key])
            if num_workers == 1:
                for filename in backgrounds:
                    random_number = np.random.randint(0, np.minimum(len(stuff[key]), 30))
                    random_stuff = random.choices(stuff[key], k=random_number)
                    data[key].append(insert_stuff_randomly(f"{self.img_dir}{filename}", random_stuff, img_size))
            else:
                data[key] = self.create_random_images_mp(num_workers, backgrounds, stuff[key], img_size)
        self.save_dataset(data, sizes)

    def save_dataset(self, data, sizes):
        """
        Saves datasets as pkl
        :param data: Data to be saved
        :param sizes: Sizes of the Dataset as Dict {train,verify,test}
        :return: None
        """
        now = datetime.datetime.now().strftime('%m-%d_%H_%M_%S')
        full_path = f"{self.save_dir}Dataset_{now}_" \
                    f"train({sizes['train']})_" \
                    f"verify({sizes['verify']})_" \
                    f"test({sizes['test']}).pkl"
        with open(full_path, 'wb') as file:
            pkl.dump(data, file, pkl.HIGHEST_PROTOCOL)

    def save_stuff(self, dir_path: str, name: str):
        full_path = f"{dir_path}{name}.pkl"
        if self.stuff:
            with open(full_path, 'wb') as file:
                pkl.dump(self.stuff, file, pkl.HIGHEST_PROTOCOL)
        else:
            print("No stuff to save")

    def load_stuff(self, dir_path: str, name):
        """
        load and replace all previously loaded stuff(extracted objects from given data after the use of extract_stuff

        :param dir_path: Directory of the file
        :param name: Name of the file
        """
        full_path = f"{dir_path}{name}.pkl"
        if os.path.exists(full_path):
            with open(full_path, 'rb') as file:
                self.stuff = pkl.load(file)
        else:
            print(f"Could not find file at: {full_path}")

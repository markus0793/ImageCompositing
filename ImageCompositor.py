import math
import os
from copy import copy

import cv2
import numpy as np
from skimage.filters import threshold_otsu


class ImageCompositor:
    def __init__(self):
        pass

    def pre_precessing_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        img = cv2.GaussianBlur(img, (25, 25), 0)
        cv2.imwrite('Blurred Image.jpg', img)
        return img

    def post_processing_img(self, img):

        kernel = np.ones((5, 5), np.uint8)
        # prepare the 5x5 shaped filter
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        # Applying bilateral filter to the image
        return img

    def opening_img(self, img):
        kernel = np.ones((5, 5), np.uint8)
        # prepare the 5x5 shaped filter
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # Applying bilateral filter to the image
        return img

    def closing_img(self, img):
        kernel = np.ones((3, 3), np.uint8)
        # prepare the 5x5 shaped filter
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        # Applying bilateral filter to the image
        return img

    def segment_image_kmeans(self, img):
        """
        Segment the image to extract the object using computer vision techniques
        """
        img = self.pre_precessing_img(img)
        twoDimage = img.reshape((-1, 3))
        twoDimage = np.float32(twoDimage)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 4
        attempts = 10

        ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        img = res.reshape((img.shape))
        img = self.post_processing_img(img)
        cv2.imwrite("kmeans.jpg", img)

    def segment_image_edges(self, img, img_real=None, name=""):
        """
        Segment the image to extract the object using computer vision techniques
        """
        # img = self.pre_precessing_img(img)
        # sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
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

    def segment_image_otsu(self, img):
        """
        Segment the image to extract the object using computer vision techniques
        """
        img = self.pre_precessing_img(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        thresh = threshold_otsu(img_gray)
        img_otsu = img_gray < thresh

        def filter_image(image, mask):
            r = image[:, :, 0] * mask
            g = image[:, :, 1] * mask
            b = image[:, :, 2] * mask

            return np.dstack([r, g, b])

        filtered = filter_image(img, img_otsu)

        cv2.imwrite("otsu.jpg", filtered)

    def segment_image_color(self, img, color="green"):
        """
        Segment the image to extract the object using computer vision techniques
        """
        gray_img = img
        img = self.pre_precessing_img(img)

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
                masked = self.post_processing_img(masked)
                # cv2.imwrite(f"color_{color}_{i}.jpg", masked)
                mask.append(masked)
        else:
            masked = cv2.inRange(img, low, high)
            masked = self.post_processing_img(masked)
            cv2.imwrite(f"color_{color}.jpg", masked)
            mask.append(masked)
        return mask

    def create_mask(self):
        """
        Create a binary mask that corresponds to the object
        """
        pass

    def extract_object(self, img, mask, name):
        """
        Multiply the mask with the image to extract the object
        """
        img = cv2.bitwise_and(img, img, mask=mask)
        # cv2.imwrite(f"object{name}.jpg", img)
        return img

    def paste_object(self, dest_img_path):
        """
        Paste the extracted object onto the destination image
        """
        dest_img = cv2.imread(dest_img_path, cv2.IMREAD_COLOR)
        pass

    def save_image(self, output_path):
        """
        Save the output image
        """
        pass


def separate_colors(img):
    # img = cv2.imread("./data/IMG_20200623_195145.jpg")
    # img = cv2.imread("./data/youngentrepr.jpg")
    # img = cv2.resize(img, (720,720))
    imgComp = ImageCompositor()
    # imgComp = ImageCompositor("./data/IMG_20200623_195145.jpg", 0)
    # imgComp = ImageCompositor("./data/Food-Waste.jpg", 0)
    imgComp.segment_image_edges(img)
    # imgComp.segment_image_kmeans(img)
    # imgComp.segment_image_otsu(img)
    mask_green = imgComp.segment_image_color(img, "green")[0]
    mask_green = imgComp.opening_img(mask_green)

    mask_orange = imgComp.segment_image_color(img, "orange")[0]
    mask_orange = imgComp.opening_img(mask_orange)
    mask_red = imgComp.segment_image_color(img, "red")[0]

    img_green = imgComp.extract_object(img, mask_green, "green")
    img_orange = imgComp.extract_object(img, mask_orange, "orange")
    img_red = imgComp.extract_object(img, mask_red, "red")

    mask_green = imgComp.segment_image_edges(img_green, "green")
    mask_red = imgComp.segment_image_edges(img_red, "red")
    mask_orange = imgComp.segment_image_edges(img_orange, "orange")


def separet_gray_scale(img, real):
    imgComp = ImageCompositor()
    imgComp.segment_image_edges(img, "edge_grey")
    masks_grey = imgComp.segment_image_color(img, "gray")
    all_good_objets_mask_real = []
    for i, mask_grey in enumerate(masks_grey):
        mask_grey = imgComp.closing_img(mask_grey)
        #mask_grey = cv2.GaussianBlur(mask_grey, (11, 11), 0)
        mask_grey = cv2.bilateralFilter(mask_grey, 9, 75, 75)
        mask_all_obj = imgComp.extract_object(real, mask_grey, f"objects_{i}")
        objectes_mask, cnt = imgComp.segment_image_edges(mask_all_obj, f"object_edges_{i}")
        good_objekts_id = []
        for i, mask_single_object in enumerate(objectes_mask):
            x, y, w, h = cv2.boundingRect(cnt[i])
            if w < 155 and h < 155:
                continue
            elif w < 75:
                continue
            elif h < 75:
                continue
            object_real = imgComp.extract_object(real, mask_single_object, "")
            cv2.imshow("preview", object_real)
            k = cv2.waitKey(0)
            if k == ord("y"):
                mid_point = (x + math.floor((w / 2)), y + math.floor((h / 2)))
                crop_mask_object = objectes_mask[i][y:y + h, x:x + w]
                crop_real_object = object_real[y:y + h, x:x + w]
                good_objekts_id.append((mid_point, crop_mask_object, crop_real_object))
        all_good_objets_mask_real.extend(good_objekts_id)
    return all_good_objets_mask_real


def save_objects_with_masks(index, objects):
    path = "./data/objects/"
    for i, (midpoint, mask, obj) in enumerate(objects):
        cv2.imwrite(f"{path}{index:03d}_{i:03d}_mask.png", mask)
        cv2.imwrite(f"{path}{index:03d}_{i:03d}_real.png", obj)


if __name__ == "__main__":
    path = "./data/pack200/"
    cv2.startWindowThread()
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('preview', 1500, 1300)
    files = sorted(os.listdir(path))
    for i in range(1, len(files), 2):
        print(files[i], files[i - 1])
        img = cv2.imread(f"{path}{files[i - 1]}")
        img_real = cv2.imread(f"{path}{files[i]}")
        # separate_colors(img)
        objects = separet_gray_scale(img, img_real)
        save_objects_with_masks(i, objects)


    cv2.destroyAllWindows()

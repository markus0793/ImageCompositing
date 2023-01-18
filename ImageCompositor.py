import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu


class ImageCompositor:
    def __init__(self):
        pass

    def pre_precessing_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        img = cv2.GaussianBlur(img, (21, 21), 0)
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

    def closing_img(self,img):
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

    def segment_image_edges(self, img, name=""):
        """
        Segment the image to extract the object using computer vision techniques
        """
        #img = self.pre_precessing_img(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
        edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)

        cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)[-2], key=cv2.contourArea)
        masked = None
        if len(cnt) > 1:
            print(f"Number of contours:{str(len(cnt))}")
            cnt = cnt[::-1]
            masks = []
            for i, c in enumerate(cnt):
                mask = np.zeros_like(img, np.uint8)
                masked = cv2.drawContours(mask, [c], -1, 255, -1)
                height, width = masked.shape[:-1]
                cv2.floodFill(masked, None, c[0][0], 255)
                low = np.array([1, 1, 1])
                high = np.array([255, 255, 255])
                masked = cv2.inRange(masked, low, high)
                masks.append(masked)
            return masks
        return []

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

    def segment_image_color(self,img, color="green"):
        """
        Segment the image to extract the object using computer vision techniques
        """
        gray_img = img
        img = self.pre_precessing_img(img)

        if color == "green":
            low = np.array([60,80,60])
            high = np.array([90,255,255])
        elif color == "orange":
            low = np.array([100,110,60])
            high = np.array([110,255,255])
        elif color == "red":
            low = np.array([120,100,60])
            high = np.array([135,255,255])
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
                #cv2.imwrite(f"color_{color}_{i}.jpg", masked)
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

    def extract_object(self,img, mask, name):
        """
        Multiply the mask with the image to extract the object
        """
        img = cv2.bitwise_and(img,img, mask=mask)
        #cv2.imwrite(f"object{name}.jpg", img)
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
    masks = imgComp.segment_image_color(img,"gray")
    for i, mask in enumerate(masks):
        open_mask = imgComp.opening_img(mask)
        masked = imgComp.extract_object(real,open_mask, f"objects_{i}")
        masks = imgComp.segment_image_edges(masked,f"object_edges_{i}")
        good_objekts_id = []

        for i, mask in enumerate(masks):
            object = imgComp.extract_object(real,mask,"")
            cv2.putText(object, "Press 'y' to save, 'n' to discard or 'q' to quit", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)
            cv2.imshow("preview", object)
            # waits for user to press any key
            # (this is necessary to avoid Python kernel form crashing)
            k = cv2.waitKey(1000)
            if k == ord("y"):
                good_objekts_id.append(i)
                print()
            # closing all open windows




if __name__ == "__main__":
    cv2.startWindowThread()
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('preview', 800, 600)

    img = cv2.imread("./data/Waste1-LBL.png")
    img_real = cv2.imread("./data/Waste1.png")
    #separate_colors(img)
    cv2.imshow("preview", img)
    separet_gray_scale(img, img_real)
    cv2.destroyAllWindows()




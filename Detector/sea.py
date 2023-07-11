import numpy as np


class DetectorSea:
    def __init__(self):
        self.merged_mask = None
        self.first_height = None
        self.lower_blue = np.array([90, 50, 50])
        self.upper_blue = np.array([130, 255, 255])
        self.lower_gray = np.array([100, 0, 100])
        self.upper_gray = np.array([200, 50, 200])
        self.lower_green = np.array([40, 50, 50])
        self.upper_green = np.array([80, 255, 255])
        self.max_height_in_px = 50
        self.max_size_of_little_items = 50
        self.color_line = (211, 130, 49)

    def border_sea(self, cv2, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_image, self.lower_blue, self.upper_blue)
        gray_mask = cv2.inRange(hsv_image, self.lower_gray, self.upper_gray)
        green_mask = cv2.inRange(hsv_image, self.lower_green, self.upper_green)
        self.merged_mask = cv2.bitwise_or(blue_mask, cv2.bitwise_or(gray_mask, green_mask))
        self.percentage_of_sea()
        return cv2.bitwise_and(image, image, mask=self.merged_mask)

    def percentage_of_sea(self):
        nb_total_pixel_filter_black, nb_total_pixel = 0, 0
        for i in range(len(self.merged_mask)):
            for j in range(len(self.merged_mask[i])):
                nb_total_pixel += 1
                if self.merged_mask[i][j] == 0:
                    nb_total_pixel_filter_black += 1
        print('The sea occupies {0:.2f}% of the visibility'.format(100 - nb_total_pixel_filter_black / nb_total_pixel * 100))

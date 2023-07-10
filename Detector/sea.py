import numpy as np


class DetectorSea:
    def __init__(self):
        self.first_height = None
        self.lower_blue = np.array([70, 50, 50])
        self.upper_blue = np.array([160, 255, 255])
        self.max_height_in_px = 50
        self.max_size_of_little_items = 50
        self.color_line = (211, 130, 49)

    def cut_line(self, contours):
        if self.first_height is None:
            self.first_height = contours[1][0][1]
        elif self.first_height + self.max_height_in_px < contours[1][0][1]:
            for i in range(1, len(contours)):
                for j in range(len(contours[i])):
                    contours[i][j][0], contours[i][j][1] = 0, 0

    def border_sea(self, cv2, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_image, self.lower_blue, self.upper_blue)
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_contours = []
        ##### REMOVE LITTLE ITEM BLUE #####
        for i in range(len(contours)):
            if len(contours[i]) > self.max_size_of_little_items:
                self.cut_line(contours[i])
                final_contours.append(contours[i])
        final_contours = tuple(final_contours)
        cv2.drawContours(image, final_contours, -1, self.color_line, 2)

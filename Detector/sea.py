import numpy as np


class DetectorSea:
    def __init__(self):
        self.lower_blue = np.array([70, 50, 50])
        self.upper_blue = np.array([160, 255, 255])

    def border_sea(self, cv2, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_image, self.lower_blue, self.upper_blue)
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_contours = []
        ##### REMOVE LITTLE ITEM BLUE #####
        for i in range(len(contours)):
            if len(contours[i]) > 500:
                final_contours.append(contours[i])
        final_contours = tuple(final_contours)
        cv2.drawContours(image, final_contours, -1, (0, 255, 0), 2)

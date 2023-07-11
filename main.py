import cv2
from Detector.sea import DetectorSea


class main:
    def __init__(self):
        self.FILE = "../1.png"
        self.image = cv2.imread(self.FILE)
        self.height, self.width = self.image.shape[:2]
        self.detector_sea = DetectorSea()
        self.run()

    def adjustment_picture(self):
        self.image = self.image[150:self.height - 100, 50:self.width - 50]
        new_width = int(self.image.shape[1] * 0.5)
        new_height = int(self.image.shape[0] * 0.5)
        self.image = cv2.resize(self.image, (new_width, new_height))

    def display_picture(self):
        cv2.imshow('Contours', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run(self):
        self.adjustment_picture()
        self.image = self.detector_sea.border_sea(cv2, self.image)
        self.display_picture()


main()

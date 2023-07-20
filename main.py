import cv2


def main():
    video = cv2.VideoCapture(0)
    while True:
        cv2.imshow('Contours', video.read()[1])
        if cv2.waitKey(1) == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


main()

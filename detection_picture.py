import tensorflow as tf
import cv2
import numpy as np


def load_model(model_path):
    return tf.saved_model.load(model_path)


def load_labelmap(labelmap_path):
    with open(labelmap_path, 'r') as f:
        labelmap_ = [line.strip() for line in f]
    return labelmap_


def detect_objects(model, image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(image_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections_ = model(input_tensor)
    num_detections = int(detections_.pop('num_detections'))
    detections__ = {key: value[0, :num_detections].numpy() for key, value in detections_.items()}
    detections__['num_detections'] = num_detections

    detections__['detection_classes'] = detections__['detection_classes'].astype(np.int64)

    return detections__


def draw_boxes(image, detections_, labelmap_, min_score_thresh=0.5):
    height, width, _ = image.shape

    for i in range(detections_['num_detections']):
        class_id = detections_['detection_classes'][i]
        score = detections_['detection_scores'][i]

        if score < min_score_thresh:
            continue

        ymin, xmin, ymax, xmax = detections_['detection_boxes'][i]
        (left, right, top, bottom) = (int(xmin * width), int(xmax * width),
                                      int(ymin * height), int(ymax * height))

        class_name = labelmap_[class_id]
        label = f"{class_name}: {score:.2f}"

        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


if __name__ == "__main__":
    model_path = 'saved_model'
    labelmap_path = 'training/labelmap.pbtxt'
    image_path = 'test1.jpg'

    model = load_model(model_path)
    labelmap = load_labelmap(labelmap_path)
    detections = detect_objects(model, image_path)

    image_with_boxes = draw_boxes(cv2.imread(image_path), detections, labelmap)
    cv2.imshow('Object Detection', image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

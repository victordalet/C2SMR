import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from train.cnn import CNN


def main():
    image_size = (224, 224)
    batch_size = 32
    nb_class_cloud = 12
    nb_epochs = 100
    path_train = "../../data/cloud"
    path_test = "../../data/cloud/class_altocumulus/Altocumulus.jpg"
    path_save_weights = "../../data/weights/cloud/w.h5"
    class_cloud_name = [
        "altocumulus",
        "cirrocumulus",
        "cirrostratus",
        "cirrus",
        "cumulonimbus",
        "cumulus",
        "cumulus congest",
        "cumulus humilis",
        "cumulus medioc",
        "nimbostratus",
        "stratocumulus",
        "stratus"
    ]
    cnn_cloud = CNN(image_size, batch_size, nb_class_cloud, nb_epochs, path_train, path_test, path_save_weights,
                    class_cloud_name)
    cnn_cloud.run()


main()

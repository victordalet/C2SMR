import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TrainCloud:
    def __init__(self):
        self.predicted_class = None
        self.test_image = None
        self.accuracy = None
        self.loss = None
        self.model = None
        self.image_size = (224, 224)
        self.batch_size = 32
        self.nb_class_cloud = 12
        self.nb_epochs = 100
        self.path_train = "../../data/cloud"
        self.path_test = "../../data/cloud/class_altocumulus/Altocumulus.jpg"
        self.path_save_weights = "../../data/weights/cloud/w.h5"
        self.class_cloud_name = [
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
        # ---- Processing for the randomness of future images ---- #
        self.train_data = keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        self.test_data = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
        # ---- Loading data ---- #
        self.train_dataset = self.train_data.flow_from_directory(
            self.path_train,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        print(self.train_dataset)
        self.run()

    def run(self):
        self.train()
        self.eval()
        self.test()
        self.save()

    def train(self):
        # ---- convolution model ---- #
        self.model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.GlobalMaxPooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.nb_class_cloud, activation='softmax')
        ])
        # ---- compiling and training the model ---- #
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.train_dataset, epochs=self.nb_epochs)

    def eval(self):
        self.loss, self.accuracy = self.model.evaluate(self.train_dataset)
        print(f"Loss: {self.loss}")
        print(f"Accuracy: {self.accuracy}")

    def test(self):
        self.test_image = keras.preprocessing.image.load_img(
            self.path_test,
            target_size=self.image_size
        )
        self.test_image = keras.preprocessing.image.img_to_array(self.test_image)
        self.test_image = tf.expand_dims(self.test_image, 0)

        self.predicted_class = self.model.predict(self.test_image)[0]
        print(self.predicted_class)
        class_max_temp = 0
        for i in range(len(self.predicted_class)):
            if self.predicted_class[class_max_temp] < self.predicted_class[i]:
                class_max_temp = i
        print(self.class_cloud_name[class_max_temp])

    def save(self):
        self.model.save_weights(self.path_save_weights)


TrainCloud()

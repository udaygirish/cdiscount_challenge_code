import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from configs import config


class ModelMaker:
    def __init__(self):
        self.description = "Model Maker Class. Currently supported models: Exception, ResNet 101, Inception V3, Efficient Net-B7"
        self.input_shape = config.model_parameter_dict["target_size"] + (3,)
        print("==" * 10)
        print(
            "The default input shape of the model will be {}".format(
                self.input_shape)
        )
        self.num_classes = config.model_parameter_dict["num_classes"]
        print(
            "The default number of output classes of the model will be {}".format(
                self.num_classes
            )
        )
        print("==" * 10)
        print(
            "The default configuration of all model makers use 1*1 conv as the head initialiser after the Pretrained model"
        )
        print("To change please write a custom model following the same functionality")
        print("==" * 10)
        self.train_from_scratch = config.model_parameter_dict["train_from_scratch"]
        self.layer_count_trainable = config.model_parameter_dict[
            "layer_count_trainable"
        ]

    def make_final_model(self, model_name=config.model_parameter_dict["model_name"]):
        print("The selected model is:{}".format(model_name))
        if model_name == "exception_min":
            model = self.make_model_exception()
        elif model_name == "resnet-101":
            model = self.make_model_resnet101()
        elif model_name == "inception-v3":
            model = self.make_model_inceptionv3()
        elif model_name == "efficientnet-b7":
            model = self.make_model_effnet_b7()
        else:
            raise Exception(
                "Model name is not matching with current implemented models.Please ensure proper model name is given.Check Config"
            )
        print("==" * 10)
        print(model.summary())
        print("==" * 10)
        # Implement Model graph as a Picture to outputs with versioning

        return model

    # Xception Model Maker
    def make_model_exception(self):
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        for size in [128, 256, 512, 728]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(size, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling2D()(x)

        activation = "softmax"
        units = self.num_classes

        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(units, activation=activation)(x)
        return keras.Model(inputs, outputs)

    # ResNet 101 Model Maker - 1*1 Conv Type
    def make_model_resnet101(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        # base_model = tf.keras.applications.resnet.ResNet101(include_top=False, weights = "imagenet", input_shape = input_shape)
        base_model = tf.keras.applications.ResNet101V2(
            include_top=False, weights=None, input_shape=self.input_shape
        )
        x = inputs
        x = base_model(x)
        # print(x.shape)
        x = layers.BatchNormalization()(x)
        global_avg_layer = layers.GlobalAveragePooling2D()
        # x = global_avg_layer(x)
        #     x = layers.Dropout(0.3)(x)
        #     x = layers.Conv2D(1024, 1, activation='relu')(x)
        #     x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(256, 1, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Flatten()(x)
        # x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        if self.train_from_scratch == 0:
            base_model.trainable = True
            layer_count_trainable = -(self.layer_count_trainable)
            print(layer_count_trainable)
            for layer in base_model.layers[:layer_count_trainable]:
                layer.trainable = False
        elif self.train_from_scratch == 1:
            base_model.trainable = True
        else:
            raise Exception(
                "The model expects a pretrained or training from scratch please ensure everything is set properly"
            )

        return tf.keras.Model(inputs, outputs)

    # Inception V3 Model Maker - 1*1 Conv Type
    def make_model_inceptionv3(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        base_model = tf.keras.applications.InceptionV3(
            include_top=False, weights=None, input_shape=self.input_shape
        )
        x = inputs
        x = base_model(x)
        # print(x.shape)
        x = layers.BatchNormalization()(x)
        global_avg_layer = layers.GlobalAveragePooling2D()
        # x = global_avg_layer(x)
        # x = layers.Flatten()(x)
        x = layers.Conv2D(512, 1, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        if self.train_from_scratch == 0:
            base_model.trainable = True
            layer_count_trainable = -(self.layer_count_trainable)
            print(layer_count_trainable)
            for layer in base_model.layers[:layer_count_trainable]:
                layer.trainable = False
        elif self.train_from_scratch == 1:
            base_model.trainable = True
        else:
            raise Exception(
                "The model expects a pretrained or training from scratch please ensure everything is set properly"
            )

        return tf.keras.Model(inputs, outputs)

    # Efficient Net B7 Model maker - 1*1 Conv Type
    def make_model_effnet_b7(self):
        inputs = keras.Input(shape=self.input_shape)
        base_model = tf.keras.applications.EfficientNetB7(
            include_top=False, weights="imagenet", input_shape=self.input_shape
        )

        x = base_model(inputs)
        global_avg_layer = layers.GlobalAveragePooling2D()
        #     x = global_avg_layer(x)
        #     x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(256, 1, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        if self.train_from_scratch == 0:
            base_model.trainable = True
            layer_count_trainable = -(self.layer_count_trainable)
            print(layer_count_trainable)
            for layer in base_model.layers[:layer_count_trainable]:
                layer.trainable = False
        elif self.train_from_scratch == 1:
            base_model.trainable = True
        else:
            raise Exception(
                "The model expects a pretrained or training from scratch please ensure everything is set properly"
            )
        return keras.Model(inputs, outputs)

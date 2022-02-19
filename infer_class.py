import tensorflow as tf
from tensorflow import keras
from src.model import ModelMaker
from src.metrics import precision_m, recall_m, f1_m

# Class method implementation for Multiple Deployment strategies
class InferModel:
    def __init__(self, model_path, model_name):
        self.description = "Class method implementation for Multiple Deployment Strategies  \
                            Current Supported - Direct Model Load, Tensorflow Serving"
        self.model_path = model_path
        self.model_name = model_name
        self.model_maker = ModelMaker()
        self.custom_objects = {
            "precision_m": precision_m,
            "recall_m": recall_m,
            "f1_m": f1_m,
        }

    def direct_model_load(self):
        if self.model_path.split(".")[-1] in ["h5", "hdf5"]:
            model = keras.models.load_model(
                self.model_path, custom_objects=self.custom_objects
            )
            # Loaded Model Summary
            print(model.summary())
            return model
        elif os.path.isdir(self.model_path):
            model = tf.saved_model.load(
                self.model_path, custom_objects=self.custom_objects
            )
            print(model.summary())
            return model
        else:
            raise Exception(
                "Could not load the model- No model present in the given path in acceptable format \
            Please check and rerun the code"
            )

    # Tensorflow Serving

    # Triton Serving

    # Web App - TF JS

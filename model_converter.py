import tensorflow as tf
from tensorflow import keras

from src.metrics import f1_m, precision_m, recall_m
from src.model import ModelMaker

"""

Current Support:
-> h5 or hdf5 to pb
-> h5 or hdf5 to tflite
-> pb to tflite

This code is written in order to support this implementation on multiple platforms
and deployment strategies such as Tensorflow Lite, Tensorflow JS, Tensorflow Serving,
ONNX, OpenVino, OPtimized tensorflow lite with the help of post training quantization.


# Currently the code doesn't support concrete function or using Select, Custom or Fused operators
For their implementations code might be updated in future or please help yourself to add the module to
this class method.

"""


class ModelConverter:
    def __init__(self, model_path):
        self.description = "Model Converter Code supporting conversions \
                            .h5 to .pb or .hdf5 to .pb \
                            .pb to .tfjs \
                            .pb to .tflite (with or without quantisation) \
                            .pb to .onnx "
        self.model_path = model_path

    def convert_h5_hdf5_to_pb(self, model_path=self.model_path):
        model = tf.keras.model.load_model(
            model_path, custom_objects=self.custom_objects
        )
        output_model_path = "".join(model_path.split(".")[:-1])
        tf.saved_model.save(model, output_model_path)

    def convert_h5_hdf5_to_tflite(self, model_path=self.model_path):
        model = tf.keras.model.load_model(
            model_path, custom_objects=self.custom_objects
        )
        converter = tf.lite.TFliteConverter.from_keras_model(model)
        output_model_path = "".join(model_path.split(".")[:-1]) + ".tflite"
        # save the Tensorflow lite model
        with open(output_model_path, "wb") as f:
            f.write(tflite_model)

    def convert_pb_to_tflite(self, model_path=self.model_path):
        converter = tf.lite.TFliteConverter.from_saved_model(model_path)
        tflite_model = converter.convert()
        output_model_path = "".join(model_path.split(".")[:-1]) + ".tflite"
        # Save the Tensorflow Lite model
        with open(output_model_path, "wb") as f:
            f.write(tflite_model)

    # def convert_pb_to_tfjs(self, model_path = self.model_path):

    # def convert_pb_to_onnx(self):
    #     import onnx
    #     import tf2onnx
    #     model

    # def convert_pb_to_tfjs(self):

    # def convert_pb_to_openvinoir(self):

    # def convert_pb_to_tensorrt(self):

    # def _quantize_model(self, dtype= 'np.float16'): # Float 16 Mixed Precision

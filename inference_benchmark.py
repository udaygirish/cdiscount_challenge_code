import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from infer_class import InferModel

model_path = "../cdiscount_challenge_code_local/models/model-best.h5"

infer_model = InferModel(model_path, "inceptionv3")


model = infer_model.direct_model_load()

# Class method implementation for Multiple Deployment str
import argparse
import asyncio
import os
import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import fastapi
import numpy as np
import pandas as pd
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, Query
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image
from starlette.middleware.cors import CORSMiddleware
from tensorflow import keras

from configs import config
from infer_class import InferModel
from src.dataprocessor import BSONIterator, CDiscountProcessor
from src.model import ModelMaker

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# For testing purpose setting model path
model_path = "../cdiscount_challenge_code_local/models/model-best.h5"

infer_model = InferModel(model_path, "inceptionv3")
model = infer_model.direct_model_load()
bson_iterator = BSONIterator
cdiscount_processor = CDiscountProcessor()
cdiscount_processor.generate_lookup_table()

category_csv = pd.read_csv(config.path_dict["categories_csv"])
print(category_csv.head(5))
app = fastapi.FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def entry_page():
    return "Welcome to the base page of Applied AI Thesis Project Demo - Multi Class Classifcation"


@app.post("/predict")
def predict_class(file: fastapi.UploadFile = fastapi.File(...)):
    t1 = time.time()
    print("===" * 10)
    with TemporaryDirectory() as td:
        filename = "temp.png"
        image = Image.open(file.file)
        filepath = Path(td) / filename

        print(type(image))
        image.save(filepath)
        print(filepath)
        image = cv2.imread(str(filepath))
        print("Image shape before preprocessing:{}".format(image.shape))
        image_resized = cv2.resize(
            image, (180, 180), interpolation=cv2.INTER_AREA)

        x = img_to_array(image_resized)
        x = ImageDataGenerator().random_transform(x)
        x = ImageDataGenerator().standardize(x)
        x = np.expand_dims(x, axis=0)
        print("Image shape after preprocessing:{}".format(x.shape))
        t2 = time.time()
        # Direct load API call (Custom TF Serving or TFLITE or TFJS still in implementation)
        output = model.predict(x)
        t3 = time.time()
        output_dict = dict()
        # for i in range(0,output):
        temp = list(output[0])
        print(type(temp))
        index = temp.index(max(temp))
        prob = round(max(temp), 2)
        category_id = cdiscount_processor.idx2cat[index]
        total_row = category_csv.loc[
            category_csv["category_id"] == category_id
        ].to_dict()
        print("OUTPUT:{}".format(total_row))
        category_l1 = total_row["category_level1"][index]
        category_l2 = total_row["category_level2"][index]
        category_l3 = total_row["category_level3"][index]
        # temp_out = {i: index, prob, category_id, category_l1, category_l2, category_l3)
        output_dict = {
            "category_index": str(index),
            "confidence": str(prob),
            "category_id": category_id,
            "category_l1": category_l1,
            "category_l2": category_l2,
            "category_l3": category_l3,
        }
        print("====" * 10)
        print(output_dict)
        print("====" * 10)

    t4 = time.time()
    return {
        "model_response": output_dict,
        "response_time": str(round((t4 - t1), 4)) + "s",
        "model_inference_time": str(round((t3 - t2), 4)) + "s",
    }


if __name__ == "__main__":
    uvicorn.run("model_api:app", host="0.0.0.0", port=5001, proxy_headers=True)

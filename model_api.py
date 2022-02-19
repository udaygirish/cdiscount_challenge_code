# Class method implementation for Multiple Deployment str
import tensorflow as tf
from tensorflow import keras
from src.model import ModelMaker
from infer_class import InferModel
import fastapi
from fastapi import FastAPI, Query
import numpy as np
import uvicorn
import cv2
from starlette.middleware.cors import CORSMiddleware
from contextlib import contextmanager
import asyncio
from tempfile import TemporaryDirectory
from pathlib import Path
from PIL import Image
from src.dataprocessor import BSONIterator, CDiscountProcessor
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
import argparse
import time
import pandas as pd
from configs import config

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# For testing purpose setting model path
model_path = "./models/model-best.h5"

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
        image_resized = cv2.resize(image, (180, 180), interpolation=cv2.INTER_AREA)

        x = img_to_array(image_resized)
        x = ImageDataGenerator().random_transform(x)
        x = ImageDataGenerator().standardize(x)
        x = np.expand_dims(x, axis=0)
        print("Image shape after preprocessing:{}".format(x.shape))
        t2 = time.time()
        # Direct load API call (Custom TF Serving or TFLITE or TFJS still in implementation)
        output = model.predict(x)
        t3 = time.time()
        output_list = []
        for i in output:
            temp = list(i)
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
            temp_out = (index, prob, category_id, category_l1, category_l2, category_l3)
            output_list.append(temp_out)

    t4 = time.time()
    return {
        "model_response": str(output_list),
        "response_time": str(t4 - t1),
        "model_inference_time": str(t3 - t2),
    }


if __name__ == "__main__":
    uvicorn.run("model_api:app", host="0.0.0.0", port=5001, proxy_headers=True)

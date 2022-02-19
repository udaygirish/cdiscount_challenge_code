import os, sys, math, io
import numpy as np
import pandas as pd
import multiprocessing as mp
import bson
import struct


import matplotlib.pyplot as plt
import skimage.io
from keras.preprocessing.image import load_img, img_to_array
from configs import config
from tqdm import tqdm
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from PIL import Image
from pymongo import MongoClient
import io
import pymongo
import skimage.io as skio
import threading
import time
from collections import defaultdict
from tqdm import *


class CDiscountProcessor:
    def __init__(self):
        self.description = " CDiscount Data Preprocessor "

        # Load Data Paths from the Config
        self.train_path = config.path_dict["train_path"]
        self.test_path = config.path_dict["test_path"]
        self.category_path = config.path_dict["category_path"]
        self.train_csv_path = config.path_dict["train_images_csv"]
        self.val_csv_path = config.path_dict["val_images_csv"]
        self.train_offsets_csv = config.path_dict["train_offsets_csv"]
        self.category_csv_path = config.path_dict["categories_csv"]
        self.sample_submission_path = config.path_dict["sample_submission_path"]

        # Load product counts or numerical parameters from the Config
        self.num_train_products = config.count_dict["num_train_products"]
        self.num_test_products = config.count_dict["num_test_products"]

        self.split_percentage = config.count_dict["split_percentage"]
        self.drop_percentage = config.count_dict["drop_percentage"]

        # Load some model parameters for Data Generator Initialisation
        self.num_classes = config.model_parameter_dict["num_classes"]
        self.batch_size = config.model_parameter_dict["batch_size"]
        self.target_size = config.model_parameter_dict["target_size"]

        self.output_base_path = "./outputs/"

    # Helper functions under same class

    def make_category_tables(self, categories_df):
        cat2idx = {}
        idx2cat = {}
        i = 0
        for ir in categories_df.itertuples():

            category_id = ir[0]
            category_idx = ir[4]
            cat2idx[category_id] = category_idx
            idx2cat[category_idx] = category_id

        return cat2idx, idx2cat

    # this takes a few minutes to execute, but we only have to do it once (we'll save the table to a CSV file afterwards).
    def read_bson(self, bson_path, num_records, with_categories):
        rows = {}
        with open(bson_path, "rb") as f, tqdm(total=num_records) as pbar:
            offset = 0
            while True:
                item_length_bytes = f.read(4)
                if len(item_length_bytes) == 0:
                    break

                length = struct.unpack("<i", item_length_bytes)[0]

                f.seek(offset)
                item_data = f.read(length)
                assert len(item_data) == length

                item = bson.BSON.decode(item_data)
                product_id = item["_id"]
                num_imgs = len(item["imgs"])

                row = [num_imgs, offset, length]
                if with_categories:
                    row += [item["category_id"]]
                rows[product_id] = row

                offset += length
                f.seek(offset)
                pbar.update()

        columns = ["num_imgs", "offset", "length"]
        if with_categories:
            columns += ["category_id"]

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "product_id"
        df.columns = columns
        df.sort_index(inplace=True)
        return df

    # Train validation split
    def make_val_set(self, df, split_percentage=0.2, drop_percentage=0.0):
        # Find the product_ids for each category.
        category_dict = defaultdict(list)
        for ir in tqdm(df.itertuples()):
            category_dict[ir[4]].append(ir[0])

        train_list = []
        val_list = []
        with tqdm(total=len(df)) as pbar:
            for category_id, product_ids in category_dict.items():
                category_idx = self.cat2idx[category_id]

                # Randomly remove products to make the dataset smaller.
                keep_size = int(len(product_ids) * (1.0 - drop_percentage))
                if keep_size < len(product_ids):
                    product_ids = np.random.choice(
                        product_ids, keep_size, replace=False
                    )

                # Randomly choose the products that become part of the validation set.
                val_size = int(len(product_ids) * split_percentage)
                if val_size > 0:
                    val_ids = np.random.choice(product_ids, val_size, replace=False)
                else:
                    val_ids = []

                # Create a new row for each image.
                for product_id in product_ids:
                    row = [product_id, category_idx]
                    for img_idx in range(df.loc[product_id, "num_imgs"]):
                        if product_id in val_ids:
                            val_list.append(row + [img_idx])
                        else:
                            train_list.append(row + [img_idx])
                    pbar.update()

        columns = ["product_id", "category_idx", "img_idx"]
        train_df = pd.DataFrame(train_list, columns=columns)
        val_df = pd.DataFrame(val_list, columns=columns)
        return train_df, val_df

    def generate_lookup_table(self):
        self.categories_df = pd.read_csv(self.category_path, index_col="category_id")
        self.categories_df["category_idx"] = pd.Series(
            range(len(self.categories_df)), index=self.categories_df.index
        )
        self.cat2idx, self.idx2cat = self.make_category_tables(self.categories_df)
        # Testing
        print(self.cat2idx[1000012755], self.idx2cat[4])

    def read_images_load_train_val(self):

        self.train_offsets_df = self.read_bson(
            self.train_path, num_records=self.num_train_products, with_categories=True
        )

        self.train_offsets_df.to_csv(self.train_offsets_csv)

        print(self.train_offsets_df["category_id"].value_counts())
        print(self.train_offsets_df["num_imgs"].value_counts())

        self.train_images_df, self.val_images_df = self.make_val_set(
            self.train_offsets_df, self.split_percentage, self.drop_percentage
        )

        print("Number of training images:", len(self.train_images_df))
        print("Number of validation images:", len(self.val_images_df))
        print("Total images:", len(self.train_images_df) + len(self.val_images_df))
        unique_train, unique_val = (
            len(self.train_images_df["category_idx"].unique()),
            len(self.val_images_df["category_idx"].unique()),
        )
        print(
            "The number of Unique Train and val category ids are :{},{}".format(
                unique_train, unique_val
            )
        )

        self.train_images_df.to_csv(self.train_csv_path)
        self.val_images_df.to_csv(self.val_csv_path)

    def initiate_data_generator(self):
        self.num_train_images = len(self.train_images_df)
        self.num_val_images = len(self.val_images_df)

        train_datagen = ImageDataGenerator()
        val_datagen = ImageDataGenerator()
        lock = threading.Lock()

        train_bson_file = open(self.train_path, "rb")
        train_gen = BSONIterator(
            train_bson_file,
            self.train_images_df,
            self.train_offsets_df,
            self.num_classes,
            train_datagen,
            lock,
            target_size=self.target_size,
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_gen = BSONIterator(
            train_bson_file,
            self.val_images_df,
            self.train_offsets_df,
            self.num_classes,
            val_datagen,
            lock,
            target_size=self.target_size,
            batch_size=self.batch_size,
            shuffle=True,
        )

        return train_gen, val_gen

    def data_gen_initialisation_and_check(self):
        train_gen, val_gen = self.initiate_data_generator()

        t1 = time.time()
        bx, by = next(train_gen)
        t2 = time.time()

        print(
            "Time took to load a batch of {} images through generator is {}".format(
                len(by), t2 - t1
            )
        )

        print("Loaded Image shape:{}".format(bx[0].shape))
        print("Class ID One Hot Encoded:{}".format(by))
        return train_gen, val_gen


# Custom Iterator Class
class BSONIterator(Iterator):
    def __init__(
        self,
        bson_file,
        images_df,
        offsets_df,
        num_class,
        image_data_generator,
        lock,
        target_size=(180, 180),
        with_labels=True,
        batch_size=32,
        shuffle=False,
        seed=None,
    ):

        self.file = bson_file
        self.images_df = images_df
        self.offsets_df = offsets_df
        self.with_labels = with_labels
        self.samples = len(images_df)
        self.num_class = num_class
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.image_shape = self.target_size + (3,)

        print(
            "Found %d images belonging to %d classes." % (self.samples, self.num_class)
        )

        super(BSONIterator, self).__init__(self.samples, batch_size, shuffle, seed)
        self.lock = lock

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        if self.with_labels:
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())

        for i, j in enumerate(index_array):
            # Protect file and dataframe access with a lock.
            with self.lock:
                image_row = self.images_df.iloc[j]
                product_id = image_row["product_id"]
                offset_row = self.offsets_df.loc[product_id]

                # Read this product's data from the BSON file.
                self.file.seek(offset_row["offset"])
                item_data = self.file.read(offset_row["length"])

            # Grab the image from the product.
            # print(j)
            # img_idx =0
            item = bson.BSON.decode(item_data)
            # item = train.find_one({'_id':int(j)})
            # print(item)
            img_idx = image_row["img_idx"]
            # print(img_idx)
            bson_img = item["imgs"][img_idx]["picture"]

            img = Image.open(io.BytesIO(bson_img))
            img = img.convert("RGB")
            img = img.resize(self.target_size, Image.NEAREST)
            # Preprocess the image.
            x = img_to_array(img)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)

            # Add the image and the label to the batch (one-hot encoded).
            batch_x[i] = x
            if self.with_labels:
                batch_y[i, image_row["category_idx"]] = 1

        if self.with_labels:
            return batch_x, batch_y
        else:
            return batch_x

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        # print(index_array)
        return self._get_batches_of_transformed_samples(index_array)

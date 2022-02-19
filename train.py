from src.callbacks import CallbacksFormatter
from src.dataprocessor import CDiscountProcessor
from src.metrics import precision_m, recall_m, f1_m
from src.model import ModelMaker
from configs import config
import tensorflow as tf
from tensorflow import keras
import argparse
import wandb
from wandb.keras import WandbCallback


cdiscount_processor = CDiscountProcessor()
modelmaker = ModelMaker()
callbacksformatter = CallbacksFormatter()

# Initialise Wandb for logging
wandb.init(
    project=config.model_parameter_dict["wandb_project_path"],
    entity=config.model_parameter_dict["wandb_entity"],
)

# Data Processing

# Generate lookup table
cdiscount_processor.generate_lookup_table()

# Read and split bson to train and val splits
cdiscount_processor.read_images_load_train_val()

# Create Train and Val Datagenerator
train_gen, val_gen = cdiscount_processor.data_gen_initialisation_and_check()

# Model maker
model = modelmaker.make_final_model()

# Callbacks Formatter

optimizer, callbacks = callbacksformatter.set_callbacks_and_optimizer()

# Compile the model
model.compile(
    optimizer=optimizer,
    loss=config.model_parameter_dict["loss"],
    metrics=["accuracy", precision_m, recall_m, f1_m],
)

# Call Model fit
H = model.fit(
    train_gen,
    epochs=config.model_parameter_dict["epochs"],
    callbacks=callbacks,
    verbose=0,
    validation_data=val_gen,
)

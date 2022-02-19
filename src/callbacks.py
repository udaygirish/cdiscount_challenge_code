# Class to define different callbacks and learning rate strategies
import tensorflow as tf
import wandb
from configs import config
import tensorflow_addons as tfa
from datetime import datetime
import os
import shutil
from wandb.keras import WandbCallback


# Custom class to log learning rate to Wandb
class LRLogger(tf.keras.callbacks.Callback):
    def __init__(self, optimizer):
        super(LRLogger, self).__init__
        self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs):
        lr = self.optimizer.learning_rate(epoch)
        wandb.log({"lr": lr}, commit=False)


class CallbacksFormatter:
    def __init__(self):
        self.description = "One place to get all callbacks and learning rate strategies"
        self.model_param_dict = config.model_parameter_dict

        self.checkpoints_base_path = self.model_param_dict["checkpoints_base_path"]

    def set_callbacks_and_optimizer(self):
        # Initialize a empty callback list
        callbacks_list = []

        # Learning rate Scheduler /Decay callback

        if self.model_param_dict["learning_rate_decay_strategy"] == "polynomial":
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=self.model_param_dict["base_learning_rate"],
                decay_steps=self.model_param_dict["decay_steps"],
                end_learning_rate=self.model_param_dict["end_learning_rate"],
                power=self.model_param_dict["decay_power"],
            )

        elif self.model_param_dict["learning_rate_decay_strategy"] == "exponential":
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.model_param_dict["base_learning_rate"],
                decay_steps=self.model_param_dict["decay_steps"],
                decay_rate=self.model_param_dict["decay_rate"],
            )

        elif self.model_param_dict["learning_rate_decay_strategy"] == "cyclic":
            lr_schedule = tfa.optimizers.CyclicalLearningRate(
                initial_learning_rate=self.model_param_dict["INIT_LR"],
                maximal_learning_rate=self.model_param_dict["MAX_LR"],
                scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
                step_size=self.model_param_dict["lr_schedule_step_size"],
            )

        else:
            lr_schedule = self.model_param_dict["base_learning_rate"]

        # Early Stopping callback
        earlystopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor=self.model_param_dict["early_stopping_monitor"],
            min_delta=self.model_param_dict["es_min_delta"],
            patience=self.model_param_dict["es_min_patience"],
            verbose=0,
            mode=self.model_param_dict["es_mode"],
            baseline=None,
            restore_best_weights=self.model_param_dict["es_restore_best_weights"],
        )

        # Reduce LR on monitor
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=self.model_param_dict["reduce_lr_monitor"],
            factor=self.model_param_dict["reduce_lr_factor"],
            patience=self.model_param_dict["reduce_lr_patience"],
            min_lr=self.model_param_dict["end_learning_rate"],
        )

        x = datetime.now()
        date_string = str(
            str(x.year)
            + "_"
            + str(x.month)
            + "_"
            + str(x.day)
            + "_"
            + str("_".join(str(x.time()).split(":")[:2]))
        )
        checkpoints_path = str(
            self.checkpoints_base_path
            + self.model_param_dict["model_name"]
            + "_"
            + str(date_string)
        )
        if os.path.isdir(checkpoints_path):
            shutil.rmtree(checkpoints_path)
            os.mkdir(checkpoints_path)
        else:
            os.mkdir(checkpoints_path)
        # Save checkpoint at every epoch callback
        checkpoint_ep = tf.keras.callbacks.ModelCheckpoint(
            checkpoints_path + "/" + "save_at_{epoch}.h5"
        )

        # Save best checkpoint based on metric monitoring

        filepath = (
            checkpoints_path + "/" + "weights.best_{epoch:02d}-{val_accuracy:.2f}.hdf5"
        )
        checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
            filepath,
            monitor=self.model_param_dict["checkpoint_monitor"],
            verbose=1,
            save_best_only=False,
            mode="max",
        )

        # Optimizer
        if self.model_param_dict["optimizer"] == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            raise Exception(
                "Please ensure the optimizer given in config is supported by the code- Read Config"
            )

        if self.model_param_dict["earlystopping_init"] == 1:
            callbacks_list.append(earlystopping_callback)
        if self.model_param_dict["reduce_lr_init"] == 1:
            callbacks_list.append(reduce_lr_callback)
        if self.model_param_dict["save_best_checkpoint"] == 1:
            callbacks_list.append(checkpoint_best)
        if self.model_param_dict["save_checkpoint_per_epoch"] == 1:
            callbacks_list.append(checkpoint_ep)

        callbacks_list.append(LRLogger(optimizer))

        wandb.config = self.model_param_dict
        wandb.config["learning_rate"] = lr_schedule

        # For Tensorflow based TQDM progress bar to show progress
        callbacks_list.append(tfa.callbacks.TQDMProgressBar())

        # Add WandbCallback()
        callbacks_list.append(WandbCallback())

        return optimizer, callbacks_list

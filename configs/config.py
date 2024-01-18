from typing import Dict

path_dict = {
    "train_path": "../cdiscount-image-classification-challenge/train.bson",
    "test_path": "../cdiscount-image-classification-challenge/test.bson",
    "category_path": "../cdiscount-image-classification-challenge/category_names.csv",
    "train_images_csv": "../cdiscount_challenge_code_local/data/train_images.csv",
    "val_images_csv": "../cdiscount_challenge_code_local/data/val_images.csv",
    "train_offsets_csv": "../cdiscount_challenge_code_local/data/train_offsets.csv",
    "categories_csv": "../cdiscount_challenge_code_local/data/categories.csv",
    "sample_submission_path": "../cdiscount-image-classification-challenge/sample_submission.csv",
}

count_dict = {
    "num_train_products": 7069896,
    "num_test_products": 1768182,
    "num_classes": 5270,
    "split_percentage": 0.2,
    "drop_percentage": 0.0,
}

# Ensure input shape and target size separately specified please ensure they are same except the channel dimensions
# Note exception is an example scratch level implementation so it can be only trained from scratch
model_parameter_dict = {
    "batch_size": 128,
    "target_size": (180, 180),
    "num_classes": 5270,
    # 'input_shape': (180,180,3)
    "train_from_scratch": 0,  # 0 Means Use pretrained weights 1 Means train from scratch
    # Finetunable layer count if pretrained weights are used.,
    "layer_count_trainable": 20,
    # Currently supported = "inception-v3, resnet-101, efficientnet-b7, exception_min"
    "model_name": "inception-v3",
    "INIT_LR": 1e-4,
    "MAX_LR": 1e-3,
    "base_learning_rate": 1e-3,
    "epochs": 50,
    "lr_schedule_step_size": 2,
    "end_learning_rate": 1e-4,
    "decay_steps": 50000,
    "decay_power": 0.5,
    "decay_rate": 0.9,
    "reduce_lr_factor": 0.2,
    "reduce_lr_patience": 5,
    "early_stopping_monitor": "val_accuracy",
    "reduce_lr_monitor": "val_accuracy",
    "checkpoint_monitor": "val_accuracy",
    "es_min_delta": 0,
    "es_min_patience": 3,
    "es_mode": "auto",
    "es_restore_best_weights": False,
    "optimizer": "adam",  # currently only adam is available
    # available  = polynomial, exponential, cyclic, no(if given no there wont be any strategy training will be on the base learning rate)
    "learning_rate_decay_strategy": "polynomial",
    "reduce_lr_init": 0,  # to use reduce lr or not if set to zero not used -> Train Callback
    # Focal loss can be also be used for focal loss please mention focalloss
    "loss": "categorical_crossentropy",
    "earlystopping_init": 0,  # Whether to use Early stopping or not if set to 0 not used
    "save_checkpoint_per_epoch": 0,
    "save_best_checkpoint": 1,
    "checkpoints_base_path": "./outputs/checkpoints/",
    "wandb_project_path": "cdiscount-challenge-diploma-project",
    "wandb_entity": "udaygirish",
}

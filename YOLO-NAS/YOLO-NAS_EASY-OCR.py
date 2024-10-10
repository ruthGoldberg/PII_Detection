import os
import torch
from super_gradients.training import Trainer, models
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

# Define the paths to your dataset
dataset_dir = '/home/noy/YOLO-NAS/PII/'
train_imgs_dir = '/home/noy/YOLO-NAS/PII/train/images'
train_labels_dir = '/home/noy/YOLO-NAS/PII/train/labels'
val_imgs_dir = '/home/noy/YOLO-NAS/PII/valid/images'
val_labels_dir = '/home/noy/YOLO-NAS/PII/valid/labels'

# Class names in your dataset
class_names =['Address', 'DOB', 'Exp date', 'details','name','number','photo','sign']

# Create a dataset object for training
dataset_params = {
    'data_dir': dataset_dir,
    'train_images_dir': train_imgs_dir,
    'train_labels_dir': train_labels_dir,
    'val_images_dir': val_imgs_dir,
    'val_labels_dir': val_labels_dir,
    'classes': class_names
}

# Define training parameters
BATCH_SIZE = 4
MAX_EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_DIR = './checkpoints'
EXPERIMENT_NAME = 'yolo_nas_s_experiment'

# Setting up dataloaders
train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': BATCH_SIZE,
        'num_workers': 2
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': BATCH_SIZE,
        'num_workers': 2
    }
)

# Initialize the YOLO-NAS-S model
model = models.get(
    "yolo_nas_s",
    num_classes=len(class_names),
    pretrained_weights="coco"
)

train_params = {
    "resume": False,
    'silent_mode': False,
    "average_best_models": True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 1e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    "max_epochs": MAX_EPOCHS,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        num_classes=len(class_names),
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(class_names),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.5,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50',
    "save_checkpoint_dir": CHECKPOINT_DIR,
    "log_dir": './logs'
}

trainer = Trainer(experiment_name=EXPERIMENT_NAME, ckpt_root_dir=CHECKPOINT_DIR)
trainer.train(model=model, training_params=train_params, train_loader=train_data, valid_loader=val_data)
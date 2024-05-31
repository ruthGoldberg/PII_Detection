import os
from super_gradients.training import Trainer
from super_gradients.training.models import YoloNAS_S
# print(dir(YoloNAS_S))

# Define the paths to your dataset
dataset_dir = '/mnt/c/Users/noyro/Documents/PII_Detection/YOLO-NAS'
train_imgs_dir = 'images/train'
train_labels_dir = 'labels/train'
val_imgs_dir = 'images/val'
val_labels_dir = 'labels/val'
# test_imgs_dir = 'images/test'
# test_labels_dir = 'labels/test'

# Class names in your dataset
class_names = ["back_side", "card_number", "exp_date", "front_side", "holder_name"]

# Create a dataset object for training
dataset_params = {
    'data_dir':dataset_dir,
    'train_images_dir':train_imgs_dir,
    'train_labels_dir':train_labels_dir,
    'val_images_dir':val_imgs_dir,
    'val_labels_dir':val_labels_dir,
    'classes':class_names
    # 'test_images_dir':test_imgs_dir,
    # 'test_labels_dir':test_labels_dir,
}

# Define the training configuration
config = {
    'batch_size': 4,
    'num_epochs': 20,
    'learning_rate': 1e-4,
    'dataset': dataset_params,
    'save_checkpoint_dir': './checkpoints',  # Directory to save checkpoints
    'log_dir': './logs',  # Directory to save logs
    'device': 'cpu'
}

# Define the arch_params for YoloNAS_S
arch_params = {
    'num_classes': len(class_names),
    'model_variant': "s"  # You can adjust this based on your specific model requirements
}

# Initialize the YOLO-NAS-S model
model = YoloNAS_S(arch_params=arch_params)

# Create a trainer object and train the model
trainer = Trainer(model=model, config=config)
trainer.train()

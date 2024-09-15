# train.py

from ultralytics import YOLOWorld
import supervision as sv
import cv2

# Define the paths to your dataset
dataset_dir = '/mnt/c/Users/noyro/Documents/PII_Detection/YOLO-NAS'
train_imgs_dir = 'train/images'
train_labels_dir = 'train/labels'
val_imgs_dir = 'valid/images'
val_labels_dir = 'valid/labels'

model = YOLOWorld(model_id="yolo_world/s") # There are multiple different sizes: s, m and l.

# Class names in your dataset
classes =['Address', 'DOB', 'Exp date','back_side','card_number', 'details', 'doc_quad','exp_date','front_side','holder_name','name','number','photo','sign']
# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="data_yoloworld.yaml", epochs=100, imgsz=640)

image = cv2.imread(IMAGE_PATH)
results = model.infer(image, confidence=0.2) # Infer several times and play around with this confidence threshold
detections = sv.Detections.from_inference(results)

import os
import cv2
import torch
from super_gradients.training import models
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

# Path to the directory containing test images
images_dir = '/home/noy/YOLO-NAS/PII/test/images/'
images_result = '/home/noy/YOLO-NAS/PII/test/cards_result'
annotations_dir = '/home/noy/YOLO-NAS/PII/test/annotations_result'  # Directory to save YOLO annotations

# Class names in your dataset
class_names =['Address', 'DOB', 'Exp date', 'details','name','number','photo','sign']

# Load the YOLO-NAS model
model_path = '/home/noy/YOLO-NAS/PII/checkpoints/yolo_nas_s_experiment/RUN_20240917_034858_081567/average_model.pth'
model = models.get("yolo_nas_s", num_classes=len(class_names), checkpoint_path=model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

# Threshold for object detection
threshold = 0.5

# Prepare post-prediction callback
post_prediction_callback = PPYoloEPostPredictionCallback(
    score_threshold=0.01,
    nms_threshold=0.01,  # Adjust according to your model's requirements
    nms_top_k=1000,     # Adjust according to your model's requirements
    max_predictions=300 # Adjust according to your model's requirements
)

# Ensure annotation directory exists
os.makedirs(annotations_dir, exist_ok=True)
os.makedirs(images_result, exist_ok=True)

# Get a list of all files in the directory
all_files = os.listdir(images_dir)

# Filter out non-image files
image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Process each image file in the directory
for image_name in image_files:
    # Read the test image
    image_path = os.path.join(images_dir, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to read image '{image_path}'")
        continue

    # Convert the image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to the desired size
    desired_size = (640, 640)  # Change this to the size required by your model
    resized_image = cv2.resize(rgb_image, desired_size)
    
    # Prepare the image for the model
    input_tensor = torch.from_numpy(resized_image).float().permute(2, 0, 1).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Perform object detection
    with torch.no_grad():
        predictions = model(input_tensor)

    # Post-process the predictions
    processed_predictions = post_prediction_callback(predictions)[0]

    # Debugging: Print the processed predictions
    print(f"Processed predictions for {image_name}: {processed_predictions}")

    # Save the annotation file
    annotation_path = os.path.join(annotations_dir, f"{os.path.splitext(image_name)[0]}.txt")
    
    with open(annotation_path, 'w') as f:
        # Draw bounding boxes on the image
        for result in processed_predictions:
            x1, y1, x2, y2, score, class_id = result

            # Debugging: Print the result details
            print(f"Result: x1={x1}, y1={y1}, x2={x2}, y2={y2}, score={score}, class_id={class_id}")

            if score > threshold:
                # Draw rectangle around the detected object
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

                # Ensure the class_id is within the range of class_names
                if 0 <= class_id < len(class_names):
                    class_name = class_names[int(class_id)].upper()
                    cv2.putText(image, class_name, (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                    # Check if the class is one of the specific ones
                    if class_name.lower() in class_names:
                        # Blacken the area within the bounding box
                        image[int(y1):int(y2), int(x1):int(x2)] = 0

                    # Convert coordinates to YOLO format
                    img_h, img_w, _ = image.shape
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h

                    # Write the annotation line
                    f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")
                else:
                    print(f"Error: class_id {class_id} is out of range")

    # Save the processed image
    result_path = os.path.join(images_result, f"result_{image_name}")
    cv2.imwrite(result_path, image)

    print(f"Processed image and annotations saved to '{result_path}' and '{annotation_path}'")

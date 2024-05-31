import os
import cv2
import torch
from super_gradients.training import models
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

# List of test images
test_images = ["2ajh064.jpg", "774328f6-f9fb-4e01-9690-40bae05a5cbf.jpeg", "aeonJAL[1].png"]

# Path to the directory containing test images
images_dir = '/mnt/c/Users/noyro/Documents/PII_Detection/YOLO-NAS/credit_card_resize_640'

# Load the YOLO-NAS model
model_path = '/mnt/c/Users/noyro/Documents/PII_Detection/YOLO-NAS/checkpoints/yolo_nas_s_experiment/RUN_20240531_150808_855290/average_model.pth'
model = models.get("yolo_nas_s", num_classes=5, checkpoint_path=model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

# Threshold for object detection
threshold = 0.5

# Prepare post-prediction callback
post_prediction_callback = PPYoloEPostPredictionCallback(
    score_threshold=0.01,
    nms_threshold=0.7,  # Adjust according to your model's requirements
    nms_top_k=1000,     # Adjust according to your model's requirements
    max_predictions=300 # Adjust according to your model's requirements
)

# Process each test image
for image_name in test_images:
    # Read the test image
    image_path = os.path.join(images_dir, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to read image '{image_path}'")
        continue

    # Convert the image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Prepare the image for the model
    input_tensor = torch.from_numpy(rgb_image).float().permute(2, 0, 1).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Perform object detection
    with torch.no_grad():
        predictions = model(input_tensor)

    # Post-process the predictions
    processed_predictions = post_prediction_callback(predictions)[0]

    # Draw bounding boxes on the image
    for result in processed_predictions:
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            # Draw rectangle around the detected object
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(image, processed_predictions.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            # Get the class name
            class_name = processed_predictions.names[int(class_id)].lower()

            # Check if the class is "card_number", "exp_dates", or "holder_name"
            if class_name in ["card_number", "exp_date", "holder_name"]:
                # Blacken the area within the bounding box
                image[int(y1):int(y2), int(x1):int(x2)] = 0

    # Save the processed image
    result_path = os.path.join(images_dir, f"result_{image_name}")
    cv2.imwrite(result_path, image)

    print(f"Processed image saved to '{result_path}'")

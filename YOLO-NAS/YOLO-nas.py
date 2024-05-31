import os
import cv2
from super_gradients.training import models

# List of test images
test_images = ["2ajh064.jpg", "774328f6-f9fb-4e01-9690-40bae05a5cbf.jpeg", "aeonJAL[1].png"]

# Path to the directory containing test images
images_dir = '/mnt/c/Users/noyro/Documents/PII_Detection/YOLO-NAS/credit_card_resize_640'

# Load the YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
model = models.get("yolo_nas_s", checkpoint_path=model_path)

# Threshold for object detection
threshold = 0.5

# Process each test image
for image_name in test_images:
    # Read the test image
    image_path = os.path.join(images_dir, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to read image '{image_path}'")
        continue

    # Perform object detection
    results = model(image)[0]

    # Draw bounding boxes on the image
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            # Draw rectangle around the detected object
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            # Get the class name
            class_name = results.names[int(class_id)].lower()

            # Check if the class is "card_number", "exp_dates", or "holder_name"
            if class_name in ["card_number", "exp_date", "holder_name"]:
                # Blacken the area within the bounding box
                image[int(y1):int(y2), int(x1):int(x2)] = 0

    # Save the processed image
    cv2.imwrite('result.jpg', image)



import os
import cv2
from ultralytics import YOLO
import datetime

start = datetime.datetime.now()

# Path to the directory containing test images
images_dir = '/home/noy/YOLOv8/test/images/'

# Output directory for processed images
output_dir = '/home/noy/YOLOv8/test/output_images_Yolov8'

# Output directory for text files
txt_output_dir = '/home/noy/YOLOv8/test/yolov8_labels'

class_names = ['address', 'dob', 'exp date', 'details', 'name', 'number', 'photo', 'sign']

# Load the YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'last.pt')
model = YOLO(model_path)

# Threshold for object detection
threshold = 0.3

# Text color dictionary for different labels (you can adjust these as needed)
text_colors = {
    "dob": (0, 255, 0),   # Green for DOB
    "name": (255, 0, 0),  # Blue for Name
    "number": (0, 0, 255) # Red for Number
}

# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(txt_output_dir, exist_ok=True)

# Process each image in the images_dir directory
for image_name in os.listdir(images_dir):
    if image_name.endswith(('.jpg', '.jpeg', '.png', '.webp')):  # Consider only image files
        input_filepath = os.path.join(images_dir, image_name)
        output_filepath = os.path.join(output_dir, image_name)
        labels_filepath = os.path.join(txt_output_dir, os.path.splitext(image_name)[0] + '.txt')
        image = cv2.imread(input_filepath)

        if image is None:
            print(f"Error: Unable to read image '{input_filepath}'")
            continue

        # Perform object detection
        results = model(image)[0]

        # Initialize labels for the current image
        labels = []

        image_height, image_width, _ = image.shape

        # Draw bounding boxes on the image
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            class_name = results.names[int(class_id)].lower()  # Ensure lowercase for consistency

            if score > threshold:
                x_center = (x1 + x2) / (2 * image_width)
                y_center = (y1 + y2) / (2 * image_height)
                box_width = (x2 - x1) / image_width
                box_height = (y2 - y1) / image_height

                
                if class_name in class_names:
                    # Add label to the text file
                    labels.append(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
                    # Draw rectangle on the image
                    color = text_colors.get(class_name, (0, 0, 0))  # Default to green if not found in text_colors
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,0), -1)
                    
                    # Put text (label) on the image
                    #cv2.putText(image, class_name.upper(), (int(x1), int(y1 - 10)), 
                                #cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        # Save the processed image to the output directory
        cv2.imwrite(output_filepath, image)

        # Save labels to text file if there are any
        if labels:
            with open(labels_filepath, 'w') as txt_file:
                for label in labels:
                    txt_file.write(label + '\n')

end = datetime.datetime.now()
elapsed_time = (end - start).total_seconds()
print(f"YOLOV8 runtime: {elapsed_time} seconds")

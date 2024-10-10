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

class_names =['Address', 'DOB', 'Exp date', 'details','name','number','photo','sign']

# Load the YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
model = YOLO(model_path)

# Threshold for object detection
threshold = 0.3

# Text color dictionary for different labels
text_colors = {
    "card_number": (255, 0, 0),   # Red
    "exp_date": (0, 255, 0),      # Green
    "holder_name": (0, 0, 255)    # Blue
}
# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(txt_output_dir, exist_ok=True)

# Process each image in the images_dir directory
#i =-1 
for image_name in os.listdir(images_dir):
    #i+=1
    #if i%10 != 0:
      #continue
    if image_name.endswith(('.jpg', '.jpeg', '.png', 'webp')):  # Consider only image files
        
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

            if score > threshold:
                x_center = (x1 + x2) / (2 * image_width)
                y_center = (y1 + y2) / (2 * image_height)
                box_width = (x2 - x1) / image_width
                box_height = (y2 - y1) / image_height

                # Append label to the list
                labels.append(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

                # Get the class name
                class_name = results.names[int(class_id)].lower()

                #Check if the class is "card_number" or "exp_date"
                if class_name in class_names:
                    # Draw rectangle around the detected object
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    # Get the color for the label
                    #color = text_colors[class_name]
                    
                    # Draw text with specified color
                    #cv2.putText(image, class_name.upper(), (int(x1), int(y1 - 10)),
                                #cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3, cv2.LINE_AA)
                    # cv2.putText(image, class_name.upper(), (int(x1), int(y1 - 10)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), -1)
                    
        # Save the processed image to the output directory
        #output_image_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_filepath, image)

        # Save labels to text file if there are any
        #if labels:
            # Replace the last occurrence of the file extension with ".txt"
        #txt_output_path = os.path.join(txt_output_dir, image_name.rsplit('.', 1)[0] + '.txt')
        with open(labels_filepath, 'w') as txt_file:
            for label in labels:
                txt_file.write(label + '\n')

end = datetime.datetime.now()
elapsed_time = (end - start).total_seconds()
print(f"YOLOV8 runtime: {elapsed_time} seconds")
import os
import cv2
from ultralytics import YOLO
import datetime
import easyocr
import re



start = datetime.datetime.now()

# Path to the directory containing test images
images_dir = '/home/noy/YOLOv8/test/images/'

# Output directory for processed images
output_dir = '/home/noy/YOLOv8/test/output_images_Yolov8_OCR'

# Output directory for text files
txt_output_dir = '/home/noy/YOLOv8/test/yolov8_OCR_labels'

class_names =['Address', 'DOB', 'Exp date', 'details','name','number','photo','sign']

# Flags to track detected classes
detected_classes = {
    "dob": False,
    "name": False,
    "number": False,
    "address": False,
    "exp date": False,
    "other_classes": False  # To track if other classes like "sign", "photo", etc., were detected
}

# Load the YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
model = YOLO(model_path)

# Threshold for object detection
threshold_YOLO = 0.3

# Function to perform object detection using YOLO
def detect_objects(image):
    return model(image)[0]

# Threshold for OCR
threshold_OCR = 0.12

# Initialize OCR reader
reader = easyocr.Reader(['en'])

# Function to recognize text in an image using OCR
def recognize_text(image):
    return reader.readtext(image)

def WriteLabels(labels, labels_filepath):        
    with open(labels_filepath, 'w') as txt_file:
        for label in labels:
            txt_file.write(label + '\n')


US_address = r'(\d{1,5}(?:\s?[A-Za-z]+(?:\s?[A-Za-z]+)?(?:\s?(?:ST|STREET|AVE|AVENUE|BLVD|BOULEVARD|RD|ROAD|DR|DRIVE|LN|LANE|CT|COURT|PL|PLACE))?)\s?(?:#?\s?[A-Za-z0-9]+)?(?:,\s)?(?:[A-Za-z\s]+)?(?:,\s)?(?:[A-Za-z]{2})?\s?\d{5}(?:-\d{4})?)'
dob_pattern = r'(?i)^(DOB|Date of Birth|Birthdate|DoB)[\s:]*\b(0[1-9]|[12][0-9]|3[01])[-/](0[1-9]|1[0-2])[-/](19|20)\d{2}\b'
card_expiry = r'^\d{2}/\d{2}$'
person_name = r'(?i)^(?!.*\b(BANK|DRIVING|LICENSE|PASSPORT|VISA|DEBIT|DATE|CREDIT|CARD|MONTH|GOVERNMENT|YEAR|VALID|STATE|OF|NO|EXPRESS|NUMBER)\b)[A-Za-z]+(?:\s+[A-Za-z]+){1,2}$'
number = r'\b(\d{9}|(?:\d{6,9})|(?:\d{5,12})|^\d{16}$)\b'


# Function to check if OCR text matches a pattern
def match_ocr_text(text, pattern):
    return re.match(pattern, text.strip())

# Function to perform OCR on an image and save the processed image - DEVIDE TO OCR PER CLASS
def perform_ocr_on_image(image, labels_filepath):
    labels = []
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = recognize_text(image_rgb)

    for (_, text, score) in results:
        print(f"Detected OCR text: {text} with score: {score}")

    for (bbox, text, prob) in results:
        if prob >= threshold_OCR:
            (top_left, _, bottom_right, _) = bbox
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

        # Check if OCR text matches any pattern and draw bounding boxes
            if match_ocr_text(text, US_address):
                print(f"OCR Detected Address: {text}")
                cv2.rectangle(image, pt1=top_left, pt2=bottom_right, color=(0, 0, 255), thickness=2)
                cv2.putText(image, 'address', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                labels.append(f"0 {top_left[0] / 640} {top_left[1] / 640} {bottom_right[0] / 640} {bottom_right[1] / 640}")
            elif match_ocr_text(text, dob_pattern):
                print(f"OCR Detected DOB: {text}")
                cv2.rectangle(image, pt1=top_left, pt2=bottom_right, color=(0, 0, 255), thickness=2)
                cv2.putText(image, 'dob', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                labels.append(f"1 {top_left[0] / 640} {top_left[1] / 640} {bottom_right[0] / 640} {bottom_right[1] / 640}")
            elif match_ocr_text(text, card_expiry):
                print(f"OCR Detected Expiry Date: {text}")
                cv2.rectangle(image, pt1=top_left, pt2=bottom_right, color=(0, 0, 255), thickness=2)
                cv2.putText(image, 'exp_date', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                labels.append(f"2 {top_left[0] / 640} {top_left[1] / 640} {bottom_right[0] / 640} {bottom_right[1] / 640}")
            elif match_ocr_text(text, person_name):
                print(f"OCR Detected Name: {text}")
                cv2.rectangle(image, pt1=top_left, pt2=bottom_right, color=(0, 0, 255), thickness=2)
                cv2.putText(image, 'name', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                labels.append(f"3 {top_left[0] / 640} {top_left[1] / 640} {bottom_right[0] / 640} {bottom_right[1] / 640}")
            elif match_ocr_text(text, number):
                print(f"OCR Detected Credit Card Number: {text}")
                cv2.rectangle(image, pt1=top_left, pt2=bottom_right, color=(0, 0, 255), thickness=2)
                cv2.putText(image, 'card_number', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                labels.append(f"4 {top_left[0] / 640} {top_left[1] / 640} {bottom_right[0] / 640} {bottom_right[1] / 640}")

    #WriteLabels(labels, labels_filepath)
    return (image, labels)

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
        results = detect_objects(image)

        # Initialize labels for the current image
        labels = []
        # Initialize a flag to keep track of the detected classes 
        other_classes_detected = False

        image_height, image_width, _ = image.shape
        print(f"YOLOv8 results for {image_name}:")

        # Draw bounding boxes on the image
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold_YOLO:
                x_center = (x1 + x2) / (2 * image_width)
                y_center = (y1 + y2) / (2 * image_height)
                box_width = (x2 - x1) / image_width
                box_height = (y2 - y1) / image_height

                # Append label to the list
                labels.append(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

                # Get the class name
                class_name = results.names[int(class_id)].lower()

                print(f"YOLO Detected: {class_name} with confidence {score}")

                 # Handle "sign", "photo", "details" - Draw bounding box, mark for OCR later
                if class_name in ["sign", "photo", "details"]:
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    detected_classes["other_classes"] = True
                    other_classes_detected = True
                
                if class_name in ["dob", "name", "number", "address", "exp date"]:
                    # Draw rectangle around the detected object
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    detected_classes[class_name] = True
                    print(f"YOLO Detected: {class_name}")
                    # Get the color for the label
                    #color = text_colors[class_name]
                    
                    # Draw text with specified color
                    #cv2.putText(image, class_name.upper(), (int(x1), int(y1 - 10)),
                                #cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3, cv2.LINE_AA)
                    # cv2.putText(image, class_name.upper(), (int(x1), int(y1 - 10)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                    # cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), -1)

         # If YOLO detected all the required classes, skip OCR
        if all(detected_classes[key] for key in ["dob", "name", "number", "address", "exp date"]):
            other_classes_detected = True  # No need for OCR as all important classes are detected
        else:
            other_classes_detected = False  # OCR is required to extract missing information
                    
        if not other_classes_detected:
            # Perform OCR on the image and save the processed image
            ocr_image, ocr_labels = perform_ocr_on_image(image, labels_filepath)
            labels.extend(ocr_labels)

        WriteLabels(labels, labels_filepath)

        # Save the processed image to the output directory
        #output_image_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_filepath, image)

        # Save labels to text file if there are any
        #if labels:
            # Replace the last occurrence of the file extension with ".txt"
        #txt_output_path = os.path.join(txt_output_dir, image_name.rsplit('.', 1)[0] + '.txt')
        # with open(labels_filepath, 'w') as txt_file:
        #     for label in labels:
        #         txt_file.write(label + '\n')

end = datetime.datetime.now()
elapsed_time = (end - start).total_seconds()
print(f"YOLOV8 runtime: {elapsed_time} seconds")
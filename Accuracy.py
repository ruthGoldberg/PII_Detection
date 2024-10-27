import os

def get_matching_files(file1_folder, file2_folder):
    matching_files = []
    for filename1 in os.listdir(file1_folder):
        if filename1.endswith('.txt'):
            filename2 = os.path.join(file2_folder, filename1)
            if os.path.isfile(filename2):
                matching_files.append((os.path.join(file1_folder, filename1), filename2))
    return matching_files

def read_annotations(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def parse_annotation(line):
    parts = line.split()
    label = parts[0]
    coordinates = [float(x) for x in parts[1:]]
    return label, coordinates

def compare_annotations(file1_path, file2_path, tolerance=0.1):
    annotations1 = read_annotations(file1_path)  # Ground truth annotations (test labels)
    annotations2 = read_annotations(file2_path)  # Model's annotations (YOLOv8, EasyOCR, etc.)
    
    common_annotations = 0
    total_annotations_file1 = len(annotations1)

    # For each ground truth annotation, find the best matching model annotation
    for annotation1 in annotations1:
        label1, coordinates1 = parse_annotation(annotation1)
        best_match = None
        best_match_similarity = 0

        for annotation2 in annotations2:
            label2, coordinates2 = parse_annotation(annotation2)
            if label1 == label2:
                # Calculate similarity based on coordinate proximity and tolerance
                similarity = sum(1 for c1, c2 in zip(coordinates1, coordinates2) if abs(c1 - c2) < tolerance) / len(coordinates1)
                if similarity > best_match_similarity:
                    best_match = annotation2
                    best_match_similarity = similarity

        # If a match is found within the tolerance, count it as a correctly detected annotation
        if best_match is not None:
            common_annotations += 1
            annotations2.remove(best_match)  # Remove to avoid double matching

    # Accuracy is calculated based on ground truth annotations (ignores extra detections by the model)
    accuracy = common_annotations / total_annotations_file1 * 100 if total_annotations_file1 != 0 else 0

    return accuracy

def calculate_average_accuracy(matching_files):
    accuracies = []
    for file1_path, file2_path in matching_files:
        accuracy = compare_annotations(file1_path, file2_path)
        accuracies.append(accuracy)
    return sum(accuracies) / len(accuracies)

# Paths to the folders containing test labels and YOLOv8 labels
test_labels_folder = '/home/noy/YOLOv8/test/labels/'
Yolov_8_labels_folder = '/home/noy/YOLOv8/test/yolov8_labels/'
easyocr_labels_folder = '/home/noy/YOLOv8/test/yolov8_OCR_labels/'
Yolo_Nas_labels_folder = '/home/noy/YOLO-NAS/PII/test/annotations_result/'

matching_files_yolov_8 = get_matching_files(test_labels_folder, Yolov_8_labels_folder)
matching_files_easyocr = get_matching_files(test_labels_folder, easyocr_labels_folder)
matching_files_yolo_nas = get_matching_files(test_labels_folder, Yolo_Nas_labels_folder)

average_accuracy_yolov_8 = calculate_average_accuracy(matching_files_yolov_8)
average_accuracy_easyocr = calculate_average_accuracy(matching_files_easyocr)
average_accuracy_yolo_nas = calculate_average_accuracy(matching_files_yolo_nas)


print("Average Accuracy YOLOv_8:", average_accuracy_yolov_8, "%")
print("Average Accuracy yolov8_ocr:", average_accuracy_easyocr, "%")
print("Average Accuracy yolo_nas:", average_accuracy_yolo_nas, "%")

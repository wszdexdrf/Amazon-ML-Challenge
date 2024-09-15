import os
import xml.etree.ElementTree as ET

# Define the class mapping
class_mapping = {
    'up arrow': 0
}

# Paths
annotations_file = 'annotations height.xml'
images_folder = '/Users/shubhendupandey/Desktop/pose-detection-keypoints-estimation-yolov8/bb'  # Folder containing your images
labels_folder = '/Users/shubhendupandey/Desktop/pose-detection-keypoints-estimation-yolov8/height labels'  # Folder to save YOLO label files

# Create labels folder if it doesn't exist
os.makedirs(labels_folder, exist_ok=True)

# Parse the XML file
tree = ET.parse(annotations_file)
root = tree.getroot()

# Iterate over each image in the XML
for image in root.findall('image'):
    image_name = image.attrib['name']
    image_width = float(image.attrib['width'])
    image_height = float(image.attrib['height'])
    label_filename = os.path.splitext(image_name)[0] + '.txt'
    label_filepath = os.path.join(labels_folder, label_filename)

    # List to hold all annotations for this image
    annotations = []

    # Iterate over each bounding box in the image
    for box in image.findall('box'):
        label_name = box.attrib['label']
        class_id = class_mapping[label_name]

        # Get bounding box coordinates
        xtl = float(box.attrib['xtl'])
        ytl = float(box.attrib['ytl'])
        xbr = float(box.attrib['xbr'])
        ybr = float(box.attrib['ybr'])

        # Convert to YOLO format
        x_center = (xtl + xbr) / 2 / image_width
        y_center = (ytl + ybr) / 2 / image_height
        width = (xbr - xtl) / image_width
        height = (ybr - ytl) / image_height

        # Ensure coordinates are between 0 and 1
        x_center = min(max(x_center, 0), 1)
        y_center = min(max(y_center, 0), 1)
        width = min(max(width, 0), 1)
        height = min(max(height, 0), 1)

        # Append annotation string
        annotation = f"{class_id} {x_center} {y_center} {width} {height}"
        annotations.append(annotation)

    # Write annotations to label file
    with open(label_filepath, 'w') as f:
        for ann in annotations:
            f.write(ann + '\n')

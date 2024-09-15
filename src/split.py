import os
import random
import shutil

# Paths
dataset_folder = '/Users/shubhendupandey/Desktop/pose-detection-keypoints-estimation-yolov8/height'
images_folder = os.path.join(dataset_folder, 'images')
labels_folder = os.path.join(dataset_folder, 'labels')
train_images_folder = os.path.join(dataset_folder, 'images', 'train')
val_images_folder = os.path.join(dataset_folder, 'images', 'val')
train_labels_folder = os.path.join(dataset_folder, 'labels', 'train')
val_labels_folder = os.path.join(dataset_folder, 'labels', 'val')

# Create train and val directories
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)

# Get list of all image files
image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Exclude 'train' and 'val' directories
image_files = [f for f in image_files if f not in ['train', 'val']]

# Shuffle and split the dataset
random.shuffle(image_files)
split_ratio = 0.8  # 80% training, 20% validation
split_index = int(len(image_files) * split_ratio)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# Function to copy files
def copy_files(file_list, src_image_folder, src_label_folder, dst_image_folder, dst_label_folder):
    for filename in file_list:
        base_name = os.path.splitext(filename)[0]
        image_src = os.path.join(src_image_folder, filename)
        label_src = os.path.join(src_label_folder, base_name + '.txt')
        image_dst = os.path.join(dst_image_folder, filename)
        label_dst = os.path.join(dst_label_folder, base_name + '.txt')

        shutil.copy(image_src, image_dst)
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)
        else:
            # If label file doesn't exist, create an empty file
            open(label_dst, 'w').close()

# Copy training files
copy_files(train_files, images_folder, labels_folder, train_images_folder, train_labels_folder)

# Copy validation files
copy_files(val_files, images_folder, labels_folder, val_images_folder, val_labels_folder)

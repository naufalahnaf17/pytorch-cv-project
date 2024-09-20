import os
import zipfile
import shutil
import xml.etree.ElementTree as ET
import cv2
import random

def download_dataset():
    if os.path.isdir("./data") == False:
        command_kaggle = "kaggle datasets download -d andrewmvd/face-mask-detection -p data/"
        os.system(command_kaggle)

        zip_file_path = "./data/face-mask-detection.zip"
        extract_folder = "./data"

        with zipfile.ZipFile(zip_file_path,'r') as zip_ref:
            zip_ref.extractall(extract_folder)

def setup_datasets():
    # Path ke folder VOC
    VOC_IMAGES_PATH = "data/images"
    VOC_ANNOTATIONS_PATH = "data/annotations"

    # Path ke folder YOLO
    YOLO_PATH = "facemask-dataset"
    TRAIN_PATH = os.path.join(YOLO_PATH, 'train')
    VALID_PATH = os.path.join(YOLO_PATH, 'valid')
    TEST_PATH = os.path.join(YOLO_PATH, 'test')

    # Persentase split dataset
    TRAIN_RATIO = 0.8
    VALID_RATIO = 0.2

    def parse_voc_annotation(xml_file, original_width, original_height):
        """Mengubah anotasi VOC menjadi format YOLO."""
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Ambil ukuran gambar dari anotasi
        image_size = root.find('size')
        img_w = int(image_size.find('width').text)
        img_h = int(image_size.find('height').text)

        # Hitung faktor skala untuk menyesuaikan bounding box
        scale_x = 448 / img_w
        scale_y = 448 / img_h

        yolo_data = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_id = class_name_to_id(class_name)  # Konversi nama kelas ke ID
            bbox = obj.find('bndbox')

            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            # Sesuaikan koordinat bounding box dengan ukuran gambar yang di-resize
            xmin, xmax = xmin * scale_x, xmax * scale_x
            ymin, ymax = ymin * scale_y, ymax * scale_y

            # Format YOLO: [class_id, x_center, y_center, width, height]
            x_center = (xmin + xmax) / 2.0 / 448
            y_center = (ymin + ymax) / 2.0 / 448
            width = (xmax - xmin) / 448
            height = (ymax - ymin) / 448

            yolo_data.append(f"{class_id} {x_center} {y_center} {width} {height}")
        return yolo_data

    def class_name_to_id(class_name):
        """Mengembalikan ID kelas berdasarkan nama kelas."""
        classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']
        return classes.index(class_name)

    def split_dataset(files, train_ratio, valid_ratio):
        """Membagi dataset menjadi train, valid, dan test berdasarkan rasio."""
        random.shuffle(files)
        total = len(files)
        train_split = int(total * train_ratio)
        valid_split = int(total * (train_ratio + valid_ratio))
        return files[:train_split], files[train_split:valid_split], files[valid_split:]

    def convert_voc_to_yolo():
        """Mengkonversi dataset VOC ke format YOLO."""
        files = [f.split('.')[0] for f in os.listdir(VOC_IMAGES_PATH) if f.endswith('.png')]
        train_files, valid_files, test_files = split_dataset(files, TRAIN_RATIO, VALID_RATIO)

        for file_list, dataset_type in [(train_files, 'train'), (valid_files, 'valid'), (test_files, 'test')]:
            for filename in file_list:
                image_src = os.path.join(VOC_IMAGES_PATH, f"{filename}.png")
                annotation_src = os.path.join(VOC_ANNOTATIONS_PATH, f"{filename}.xml")

                # Tentukan folder tujuan berdasarkan dataset type (train, valid, test)
                dest_image_path = os.path.join(YOLO_PATH, dataset_type, "images", f"{filename}.png")
                dest_label_path = os.path.join(YOLO_PATH, dataset_type, "labels", f"{filename}.txt")

                # Baca gambar menggunakan OpenCV dan resize ke 448x448
                image = cv2.imread(image_src)
                resized_image = cv2.resize(image, (448, 448))

                # Simpan gambar yang sudah di-resize
                cv2.imwrite(dest_image_path, resized_image)

                # Konversi anotasi VOC ke format YOLO
                original_width = image.shape[1]
                original_height = image.shape[0]
                yolo_data = parse_voc_annotation(annotation_src, original_width, original_height)

                # Simpan label ke file
                with open(dest_label_path, 'w') as f:
                    f.write("\n".join(yolo_data))

    def create_yaml_file():
        """Membuat file YAML untuk konfigurasi dataset YOLO."""
        yaml_content = f"""
        train: {os.path.abspath(TRAIN_PATH)}/images
        val: {os.path.abspath(VALID_PATH)}/images
        test: {os.path.abspath(TEST_PATH)}/images

        nc: 3  # Jumlah kelas
        names: ['with_mask', 'without_mask', 'mask_weared_incorrect']
        """
        with open(os.path.join(YOLO_PATH, 'yolo.yaml'), 'w') as f:
            f.write(yaml_content)

    # Jalankan konversi dan buat file YAML
    if not os.path.exists(YOLO_PATH):
        for folder in [TRAIN_PATH, VALID_PATH, TEST_PATH]:
            os.makedirs(os.path.join(folder, "images"), exist_ok=True)
            os.makedirs(os.path.join(folder, "labels"), exist_ok=True)
        
        convert_voc_to_yolo()
        create_yaml_file()
import cv2
import os
import json
import logging

logging.basicConfig(level=logging.INFO)

# Load configuration
with open('../config.json', 'r') as config_file:
    config = json.load(config_file)

lfw_dir = config['lfw_dir']
output_dir = config['output_dir']
os.makedirs(output_dir, exist_ok=True)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Loop through LFW dataset
for person_name in os.listdir(lfw_dir):
    person_path = os.path.join(lfw_dir, person_name)
    if os.path.isdir(person_path):
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            try:
                img = cv2.imread(image_path)
                if img is None:
                    logging.warning(f"Could not read image {image_path}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                output_image_path = os.path.join(output_dir, f"{person_name}_{image_name}")
                cv2.imwrite(output_image_path, img)

            except Exception as e:
                logging.error(f"Error processing {image_path}: {e}")

logging.info("Face detection completed. Check the output folder.")

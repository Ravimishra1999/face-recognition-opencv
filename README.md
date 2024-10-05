# Face Detection and Recognition Project

# Overview
This project uses Python and OpenCV to detect faces in images from the LFW (Labeled Faces in the Wild) dataset. The project is designed to be easy to set up and run on a Windows Server environment.

---

# Prerequisites
Python 3.11: Ensure Python is installed and added to your system's PATH.
Visual Studio Build Tools: Required for compiling some Python packages.
    - Download from [Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
    - During installation, select "Desktop Development with C++" and install.

# Setup Instructions

# Step 1: Download the LFW Dataset
1. Visit the [LFW Dataset Download Page](http://vis-www.cs.umass.edu/lfw/).
2. Download the "All images as .tgz" file.  you will need to extract files result will be in .tar.
3. Extract the contents of .tar to `C:\datasets\lfw`. You should see multiple folders named after individuals.

# Step 2: Create Project Structure
1. Open cmd as an administrator.
2. Create the project directory:
  on cmd
    cd C:\
    mkdir face_recognition_project
    cd face_recognition_project
    mkdir data output output\faces_detected scripts
    

# Step 3: Create and Activate a Virtual Environment
1. Create a virtual environment to isolate your project's dependencies:
    on cmd
    python -m venv env
    
2. Activate the virtual environment:
    on cmd
    env\Scripts\activate
    

# Step 4: Install Required Packages
1. With the virtual environment active, install necessary Python packages:
    on cmd
    pip install opencv-python opencv-python-headless face_recognition numpy python-dotenv
    

# Step 5: Set Up Configuration
1. In `C:\face_recognition_project`, create a file named `config.json`.
2. Paste the following configuration into `config.json`:
    json
    {
      "lfw_dir": "C:/datasets/lfw",
      "output_dir": "C:/face_recognition_project/output/faces_detected"
    }
    

# Step 6: Create the Face Detection Script
1. Navigate to the `scripts` folder:
    on cmd
    cd C:\face_recognition_project\scripts
    
2. Create a new file named `face_detection.py` and paste the following code:
    python
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
    

# Step 7: Run the Face Detection Script
1. Ensure your virtual environment is active:
    on cmd
    env\Scripts\activate
    
2. Navigate to the `scripts` folder and run the script:
    on cmd
    cd C:\face_recognition_project\scripts
    python face_detection.py
    

3. Check the `C:\face_recognition_project\output\faces_detected` folder for processed images with detected faces.


# TT
- Ensure Python and all packages are installed correctly & also upgrade PIP version it will trigger once.
- I installed cmake and create a new entry added as path variable in advance properties under system vairiable  and then tried dlib installation then it was successful.
- Verify the LFW dataset is properly extracted to `C:/datasets/lfw`.
- The virtual environment must be active when running scripts.

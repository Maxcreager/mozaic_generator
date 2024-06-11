import cv2
import logging
import random
import os
from PIL import Image, ImageOps
import numpy as np

def is_image_file(filename):
    """Check if a file is an image based on its extension / Vérifier si un fichier est une image en fonction de son extension."""
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))

def is_video_file(filename):
    """Check if a file is a video based on its extension / Vérifier si un fichier est une vidéo en fonction de son extension."""
    return filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm'))

def extract_frame(data):
    """Extract a random frame from a video and handle opening and reading errors / Extraire une image aléatoire d'une vidéo et gérer les erreurs d'ouverture et de lecture."""
    video_path, output_path = data
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            logging.error(f"Error opening video file: {video_path}")
            return None
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        random_frame = random.randint(0, total_frames - 1)
        video.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        success, image = video.read()
        video.release()
        if success:
            cv2.imwrite(output_path, image)
            return output_path
        else:
            logging.error(f"Failed to extract frame from video: {video_path}")
            return None
    except Exception as e:
        logging.error(f"Exception while processing video {video_path}: {e}")
        return None

def detect_and_crop(image_path, net, classes, target_class="person", confidence_threshold=0.5):
    """Detect the main object in the image and crop it accordingly / Détecter l'objet principal dans l'image et le recadrer en conséquence."""
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image {image_path}")
        return

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    
    net.setInput(blob)
    detections = net.forward()
    
    max_confidence = 0
    box = None
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            if classes[idx] == target_class:
                if confidence > max_confidence:
                    max_confidence = confidence
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    
    if box is not None:
        (startX, startY, endX, endY) = box.astype("int")

        # Center of the bounding box
        centerX = (startX + endX) // 2
        centerY = (startY + endY) // 2
        
        # Determine the size of the largest square
        box_width = endX - startX
        box_height = endY - startY
        max_dim = max(box_width, box_height)  # Use the largest dimension
        
        # Compute the coordinates of the square
        half_dim = max_dim // 2
        startX = max(centerX - half_dim, 0)
        startY = max(centerY - half_dim, 0)
        endX = min(centerX + half_dim, w)
        endY = min(centerY + half_dim, h)

        cropped_image = image[startY:endY, startX:endX]
        if cropped_image.size == 0:
            logging.error(f"Cropping resulted in an empty image for {image_path}")
        else:
            cv2.imwrite(image_path, cropped_image)
    else:
        logging.info(f"No main object found in image {image_path}")

def resize_to_square(image_path, size):
    """Resize the cropped image to a square of the given size / Redimensionner l'image recadrée en un carré de la taille donnée."""
    try:
        img = Image.open(image_path)
        img = ImageOps.fit(img, (size, size), Image.ANTIALIAS)
        img.save(image_path)
    except Exception as e:
        logging.error(f"Failed to resize image {image_path} to square: {e}")

def open_and_convert_image(img_path):
    """Open an image and ensure it is fully loaded, then convert to 'RGBA' if it uses a transparency palette / Ouvrir une image, s'assurer qu'elle est complètement chargée, puis la convertir en 'RGBA' si elle utilise une palette de transparence."""
    try:
        if not os.path.exists(img_path):
            logging.error(f"Image file does not exist: {img_path}")
            return None
        img = Image.open(img_path)
        img.load()
        if img.mode == 'P':
            alpha = img.info.get('transparency', -1)
            if alpha >= 0:
                img = img.convert('RGBA')
        return img
    except Exception as e:
        logging.error(f"Failed to open or convert image {img_path}: {e}")
        return None

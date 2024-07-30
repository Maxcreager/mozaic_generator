import os
import random
import logging
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from image_processing import extract_frame, is_image_file, is_video_file

def process_files(input_folder, output_folder, num_images=None):
    """Process files in the input folder to extract image and video files / Traiter les fichiers dans le dossier d'entrée pour extraire les fichiers image et vidéo."""
    file_paths = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if is_image_file(file) or is_video_file(file):
                file_paths.append(os.path.join(root, file))

    logging.debug(f"Found {len(file_paths)} files in total / {len(file_paths)} fichiers trouvés au total.")

    if num_images is not None:
        logging.debug(f"Selecting {num_images} random images / Sélection de {num_images} images aléatoires.")
        file_paths = random.sample(file_paths, min(num_images, len(file_paths))) if num_images < len(file_paths) else file_paths[:num_images]

    image_files = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for file_path in file_paths:
            if is_video_file(file_path):
                output_image_path = os.path.join(output_folder, f'image_from_video_{os.path.basename(file_path)}.jpg')
                futures.append(executor.submit(extract_frame, (file_path, output_image_path)))
            elif is_image_file(file_path):
                image_files.append(file_path)
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    image_files.append(result)
            except Exception as e:
                logging.error(f"Exception during processing: {e}")
        executor.shutdown(wait=True)

    logging.debug(f"Processed {len(image_files)} image files / {len(image_files)} fichiers image traités.")
    return image_files

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

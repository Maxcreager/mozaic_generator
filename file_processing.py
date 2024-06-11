import os
import random
import logging
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
            result = future.result()
            if result:
                image_files.append(result)
        executor.shutdown(wait=True)

    logging.debug(f"Processed {len(image_files)} image files / {len(image_files)} fichiers image traités.")
    return image_files

import argparse
import logging
import random
import yaml
import cv2
from file_processing import process_files
from mosaic_creation import create_segmented_mosaic, create_single_mosaic

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    config = load_config("config.yaml")

    MODEL_PROTO = config['model']['proto']
    MODEL_WEIGHTS = config['model']['weights']
    PAGE_SIZES = config['page_sizes']

    # Setup logging / Configurer la journalisation
    logging.basicConfig(filename=config['logging']['filename'], level=config['logging']['level'], 
                        format='%(asctime)s:%(levelname)s:%(message)s')

    # Load pre-trained model and configuration for MobileNet-SSD / Charger le modèle pré-entraîné et la configuration pour MobileNet-SSD
    NET = cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_WEIGHTS)
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    logging.debug(f"Input folder: {args.input}, Output folder: {args.output}, Columns: {args.cols}, Num images: {args.num_images}, Page format: {args.page_format}, DPI: {args.dpi}, Randomize: {args.random}")
    
    # Validate inputs / Valider les entrées
    if args.dpi <= 0:
        raise ValueError("DPI must be a positive integer.")
    if args.cols is not None and args.cols <= 0:
        raise ValueError("Number of columns must be a positive integer if specified.")
    if args.num_images is not None and args.num_images <= 0:
        raise ValueError("Number of images must be a positive integer if specified.")

    # Process files / Traiter les fichiers
    image_files = process_files(args.input, args.output, args.num_images)

    if args.random:
        logging.debug("Randomizing the order of images / Randomisation de l'ordre des images")
        random.shuffle(image_files)

    if image_files:
        if args.page_format:
            create_segmented_mosaic(image_files, args.cols, args.output, args.page_format, args.dpi, PAGE_SIZES, NET, CLASSES)
        else:
            create_single_mosaic(image_files, args.cols, args.output, NET, CLASSES)
    else:
        logging.error("No valid image files found to create a mosaic / Aucun fichier image valide trouvé pour créer une mosaïque.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a mosaic from videos and images in a folder / Créer une mosaïque à partir de vidéos et d\'images dans un dossier.')
    parser.add_argument('-i', '--input', required=True, help='Input folder containing videos and images / Dossier d\'entrée contenant des vidéos et des images.')
    parser.add_argument('-o', '--output', required=True, help='Output folder for the mosaic image / Dossier de sortie pour l\'image mosaïque.')
    parser.add_argument('-l', '--cols', type=int, help='Number of columns in the mosaic / Nombre de colonnes dans la mosaïque.', default=0)
    parser.add_argument('-p', '--page_format', type=str, help='Specify the page format for segmentation / Spécifiez le format de la page pour la segmentation.')
    parser.add_argument('-d', '--dpi', type=int, default=300, help='Specify the DPI for the page format / Spécifiez le DPI pour le format de la page.')
    parser.add_argument('-n', '--num_images', type=int, help='Specify the number of images to include in the mosaic / Spécifiez le nombre d\'images à inclure dans la mosaïque.')
    parser.add_argument('-r', '--random', action='store_true', help='Randomize the order of images in the mosaic / Randomiser l\'ordre des images dans la mosaïque.')
    args = parser.parse_args()

    main(args)

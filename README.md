# Mozaic Generator

This project is a Python script that creates image mosaics from images and videos in a specified folder. The script can detect main objects in images, crop them to squares, and arrange them in a mosaic format.

## Features

- Extract frames from videos
- Detect main objects in images using a pre-trained MobileNet-SSD model
- Crop images to the largest possible square centered on the main object
- Resize and arrange images into a mosaic format
- Support for multiple output formats and DPIs

## Requirements

- Python 3.x
- OpenCV
- Pillow
- NumPy
- aiofiles
- PyYAML

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/username/mozaic_generator.git
    cd mozaic_generator
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

```bash
python main.py -i <input_folder> -o <output_folder> -l <columns> [-p <page_format>] [-d <dpi>] [-n <num_images>] [-r]


#### Français

```markdown
# Générateur de Mosaïques

Ce projet est un script Python qui crée des mosaïques d'images à partir d'images et de vidéos dans un dossier spécifié. Le script peut détecter les objets principaux dans les images, les recadrer en carrés et les organiser dans un format de mosaïque.

## Fonctionnalités

- Extraction de cadres de vidéos
- Détection des objets principaux dans les images en utilisant un modèle MobileNet-SSD pré-entraîné
- Recadrage des images au plus grand carré possible centré sur l'objet principal
- Redimensionnement et organisation des images dans un format de mosaïque
- Prise en charge de plusieurs formats de sortie et DPI

## Exigences

- Python 3.x
- OpenCV
- Pillow
- NumPy
- aiofiles
- PyYAML

## Installation

1. Cloner le dépôt :
    ```bash
    git clone https://github.com/username/mozaic_generator.git
    cd mozaic_generator
    ```

2. Installer les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

```bash
python main.py -i <dossier_entrée> -o <dossier_sortie> -l <colonnes> [-p <format_page>] [-d <dpi>] [-n <num_images>] [-r]

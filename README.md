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

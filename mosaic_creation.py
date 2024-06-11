import os
import logging
from PIL import Image
from image_processing import detect_and_crop, resize_to_square, open_and_convert_image

def get_page_size(format_name, dpi, page_sizes):
    """Get page dimensions in pixels for a given format and DPI / Obtenir les dimensions de la page en pixels pour un format et une résolution (DPI) donnés."""
    if format_name in page_sizes:
        width_inch, height_inch = page_sizes[format_name]
        logging.debug(f"Page format: {format_name}, Width: {width_inch} inches, Height: {height_inch} inches, DPI: {dpi}")
        return (int(width_inch * dpi), int(height_inch * dpi))
    else:
        raise ValueError(f"Unsupported page format: {format_name}")

def create_segmented_mosaic(images, cols, output_folder, page_format, dpi, page_sizes, net, classes):
    """Create a mosaic of images segmented by page to avoid splitting images across pages / Créer une mosaïque d'images segmentée par page pour éviter de diviser les images entre les pages."""
    page_width, page_height = get_page_size(page_format, dpi, page_sizes)
    tile_size = page_width // cols
    logging.debug(f"Tile size: {tile_size}x{tile_size} pixels")

    pages = []
    current_page = []
    current_row = []
    y_pos = 0

    for img_path in images:
        detect_and_crop(img_path, net, classes)
        img = Image.open(img_path)
        if img.width < tile_size or img.height < tile_size:
            logging.info(f"Skipping image {img_path} due to insufficient resolution")
            continue
        resize_to_square(img_path, tile_size)
        img = open_and_convert_image(img_path)
        if img is None:
            continue

        if len(current_row) >= cols:
            current_page.append(current_row)
            current_row = []
            y_pos += tile_size
            if y_pos + tile_size > page_height:
                pages.append(current_page)
                current_page = []
                y_pos = 0
        current_row.append(img)

    if current_row:
        current_page.append(current_row)
    if current_page:
        pages.append(current_page)

    for i, page in enumerate(pages):
        mosaic = Image.new('RGB', (page_width, min(page_height, y_pos + tile_size)), (0, 0, 0))
        y_offset = 0
        for row in page:
            x_offset = 0
            for img in row:
                mosaic.paste(img, (x_offset, y_offset))
                x_offset += tile_size
            y_offset += tile_size
        output_file = os.path.join(output_folder, f'page_{i + 1}.jpg')
        mosaic.save(output_file)
        logging.debug(f"Saved mosaic page: {output_file}")

def create_single_mosaic(images, cols, output_folder, net, classes):
    """Create a single mosaic with all images arranged in the specified number of columns / Créer une seule mosaïque avec toutes les images disposées dans le nombre de colonnes spécifié."""
    tile_size = 256  # Example size for each tile
    logging.debug(f"Tile size: {tile_size}x{tile_size} pixels")

    # Determine the size of the mosaic / Déterminer la taille de la mosaïque
    rows = (len(images) + cols - 1) // cols  # Round up to ensure all images are included / Arrondir pour s'assurer que toutes les images sont incluses
    mosaic_width = cols * tile_size
    mosaic_height = rows * tile_size

    mosaic = Image.new('RGB', (mosaic_width, mosaic_height), (0, 0, 0))

    x_offset = 0
    y_offset = 0

    for img_path in images:
        detect_and_crop(img_path, net, classes)
        img = Image.open(img_path)
        resize_to_square(img_path, tile_size)
        img = open_and_convert_image(img_path)
        if img is None:
            continue

        mosaic.paste(img, (x_offset, y_offset))
        x_offset += tile_size

        if x_offset >= mosaic_width:
            x_offset = 0
            y_offset += tile_size

    output_file = os.path.join(output_folder, 'mosaic.jpg')
    mosaic.save(output_file)
    logging.debug(f"Saved single mosaic: {output_file}")

import json
import rasterio
import numpy as np 
import cv2
import tifffile
import matplotlib.pyplot as plt
import glob
from concurrent.futures import ProcessPoolExecutor
import os
import sys  

if len(sys.argv) != 3:
    print("Usage: python create_masks.py <input_directory> <output_directory>")
    sys.exit(1)
    
INPUT_DIRECTORY = sys.argv[1]
OUTPUT_DIRECTORY = sys.argv[2]

COLOR_MAPPING = {
    "tissue_tumor": (255, 0, 0),     
    "tissue_stroma": (0, 255, 0),   
    "other": (0, 0, 255)             
}


def generate_masks(geojson_path, img_shape):
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    masks = {}

    for feature in data['features']:
        original_label = feature['properties']['classification']['name'] 
        
        if original_label == "tissue_tumor":
            label = "tissue_tumor"
        elif original_label == "tissue_stroma":
            label = "tissue_stroma"
        else:
            label = "other"
        
        if label not in masks:
                masks[label] = np.zeros(img_shape, dtype=np.uint8)
        
        geometry_type = feature['geometry']['type']
        
        if geometry_type == "Polygon" :
            coords = feature['geometry']['coordinates']
            for polygon in coords:
                pts = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(masks[label], [pts], 255)
        
        elif geometry_type == "MultiPolygon":
            coords = feature['geometry']['coordinates']
            for multi_poly in coords:
                for polygon in multi_poly:
                    pts = np.array(polygon, dtype=np.int32)
                    cv2.fillPoly(masks[label], [pts], 255)

    return masks

def create_mask_image(all_masks_dict, img_height, img_width):
 
    rgb_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    for class_name, color in COLOR_MAPPING.items():

        if class_name in all_masks_dict:
            class_mask = all_masks_dict[class_name]
            
            rgb_mask[class_mask > 0] = color
            
    return rgb_mask



def process_single_file(geojson_path):
    try:
  
        base_name = os.path.basename(geojson_path)
        output_name = base_name.replace('.geojson', '.tif') 
        output_path = os.path.join(OUTPUT_DIRECTORY, output_name)
        
        
        masks_dict = generate_masks(geojson_path,(1024,1024))
       
        rgb_mask = create_mask_image(masks_dict, 1024,1024)
        

        tifffile.imwrite(output_path, rgb_mask)
        
        return True 

    except Exception as e:
        print(f"Error processing {geojson_path}: {e}")
        return False

def generate_segmentation_masks():

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    file_pattern = os.path.join(INPUT_DIRECTORY, "*.geojson")
    file_list = glob.glob(file_pattern)
   

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_file, file_list))

generate_segmentation_masks()

print("Colour mapping: Red for Tissue Tumor, Green for Tissue Stroma, Blue for Other")
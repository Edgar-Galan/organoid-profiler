"""Simple script to test CellPose models on dataset images."""

import os
import sys
from pathlib import Path

# Add parent directory to path to allow importing server
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from cellpose import models, io, plot
import argparse
import matplotlib.pyplot as plt
#import stackview
import time
from server import convert_rgb_to_grayscale_uint8

# Paths
MODEL_DIR = Path(__file__).parent.parent / "models" / "models"
DATASET_DIR = Path(__file__).parent.parent / "dataset"

def test_model(model, image_path, output_dir):
    # Load and process image
    img = io.imread(str(image_path))
    img = convert_rgb_to_grayscale_uint8(img)
    # print image 
    plt.imshow(img)
    plt.show()
    masks, flows, styles = model.eval(img, diameter=30,)    
    print(f"Found {len(np.unique(masks)) - 1} cells")
    print("Found cells")
        
def main():

    parser = argparse.ArgumentParser(description="Test CellPose models on dataset images")
    parser.add_argument("--model-name", type=str, default=None, help="Path to the model to test")
    parser.add_argument("--dataset-dir", type=str, default=DATASET_DIR, help="Path to the dataset directory")

    args = parser.parse_args()

    model_name = args.model_name
    dataset_dir = Path(args.dataset_dir)

    if model_name is None:

        model_paths = sorted(MODEL_DIR.glob("cellpose_*"))
        if not model_paths:
            raise FileNotFoundError(f"No models found in {MODEL_DIR}")

        for i, model_path in enumerate(model_paths):
            print(f"{i}: {model_path}")
        model_name = input("Enter the index of the model to test: ")
        model_name = model_paths[int(model_name)]

    ### LOADING MODEL ###
    model_path = MODEL_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_name} not found in {MODEL_DIR}")

    print(f"Testing model: {model_path}")
    model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))

    ### LOADING DATASET DIR ###
    print(f"Loading dataset from: {dataset_dir}")
    image_files = sorted(dataset_dir.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"No images found in {dataset_dir}")

    
    ### TESTING MODELS ON DATASET ###
    test_output_dir = dataset_dir / "test"
    for image_path in image_files:
        start_time = time.time()
        # get day 10 from the image path
        day = image_path.name.split("_")[1]
        if day != "d10":
            continue
        print(f"Testing on: {image_path.name}")
        test_model(model=model, image_path=image_path, output_dir=test_output_dir)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        break

    print("Test completed successfully!")


if __name__ == "__main__":
    main()
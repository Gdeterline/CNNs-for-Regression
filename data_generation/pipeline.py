import os, sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data_generation')))
from tools import generate_arrow_dataset

def create_arrow_dataset(n_samples=2000, output_dir="data", image_dir="images"):
    X, y = generate_arrow_dataset(n_samples=n_samples)
    
    # check for existing files to avoid overwriting
    max_index = check_max_in_directory(os.path.join(output_dir, image_dir))
    if max_index >= 0:
        start_index = max_index + 1
    else:
        start_index = 0
    
    # Create image names
    image_names = [f"arrow_{i}.png" for i in range(start_index, start_index + n_samples)]

    # save images in a directory
    os.makedirs(os.path.join(output_dir, image_dir), exist_ok=True)
    for i in range(n_samples):
        img_path = os.path.join(output_dir, image_dir, image_names[i])
        plt.imsave(img_path, X[i].squeeze(), cmap='gray')
        
    # Create a DataFrame for labels
    df = pd.DataFrame({
        'image_name': image_names,
        'angle': y
    })
    
    # Save labels to a CSV file
    csv_path = os.path.join(output_dir, "labels.csv")
    df.to_csv(csv_path, index=False)
    print(f"Dataset created with {n_samples} samples.")
    

def check_max_in_directory(directory):
    max_index = -1
    for filename in os.listdir(directory):
        if filename.startswith("arrow_") and filename.endswith(".png"):
            index_str = filename[len("arrow_"):-len(".png")]
            try:
                index = int(index_str)
                if index > max_index:
                    max_index = index
            except ValueError:
                continue
    return max_index
    
    
if __name__ == "__main__":
    create_arrow_dataset(n_samples=4900, output_dir="data", image_dir="images")
    

    
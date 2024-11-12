import os
import numpy as np 
import matplotlib.pyplot as plt 
import rasterio
from rasterio.plot import show
from sklearn.cluster import KMeans

DATA_DIR = "data"
OUTPUT_DIR = "output"

def load_satellite_image (file_path):
    """Load a satellite image using rasterio"""
    try:
        with rasterio.open(file_path) as src:
            image = src.read()
            profile = src.profile
        print(f"Loaded image iwth shape: {image.shape}")
        return image, profile
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None 
    
def display_image(image, title="Satellite Image"): 
    plt.figure(figsize=(10, 6))
    plt.imshow(np.moveaxis(image, 0, -1))
    plt.title(title)
    plt.axis('off')
    plt.show()

def preprocess_image(image):
    height, width = image.shape[1], image.shape [2]
    flattened_image = image.reshape(3, -1).T 
    print(f"Flattened image shape: {flattened_image.shape}")
    return flattened_image, height, width 

def perform_kmeans_clustering(flattened_image, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(flattened_image)
    print("Kmeans clustering completed.")
    return kmeans.labels_ 

def display_clustered_image(labels, height, width): 
    clustered_image = labels.reshape(height, width)
    plt.figure(figsize=(10, 6)) 
    plt.imshow(clustered_image, cmap='viridis') 
    plt.title("Clustered Satellite Image") 
    plt.axis('off')
    plt.show() 

if __name__ == "__main__":
    sample_image_path = os.path.join(DATA_DIR, "RGB.byte (1).tif") 
    image, profile = load_satellite_image(sample_image_path) 

    if image is not None: 
        display_image(image, "Original Satellite Image") 

        flattened_image, height, width = preprocess_image(image) 

        labels = perform_kmeans_clustering(flattened_image, n_clusters=4)

        display_clustered_image(labels, height, width) 

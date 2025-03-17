"""
K-Means Clustering for downsampled NIFTI Whole-Brain Microscopy Images
====================================================

Author: Ahmadreza Attarpour (a.attarpour@mail.utoronto.ca)
Date: 2025

This script applies K-means clustering to downsampled whole-brain light sheet 
microscopy images stored in NIfTI format. The goal is to segment the image into 
distinct clusters (AAV signal) and identify the largest ones for downstream processing.

#Key Features:
- Efficient Clustering: Uses GPU-accelerated K-means (via PyTorch) to segment high-resolution microscopy images.
- Flexible Clustering Parameters: Allows specifying the number of clusters and selecting the largest ones.
- Automated NIfTI Export: Saves each cluster and the mixed largest clusters as individual NIfTI files.

#Usage:
python kmeans_clustering.py -i input.nii -o output_dir -n num_clusters -l num_largest -g gpu_index

#Arguments:
-i, --input: Path to the input NIfTI file.
-o, --output_dir: Directory where output NIfTI files will be saved.
-n, --num_clusters: Number of clusters for K-means (default: 8).
-l, --num_largest: Number of largest clusters to mix (default: 2).
-g, --gpu_index: Index of the GPU to use (default: 0, set to CPU if unavailable).


#Outputs:
-Mixed Clusters: A NIfTI file (mixed_clusters.nii) where the largest clusters are merged.
-Individual Clusters: Each cluster is saved separately as cluster_X.nii in the output directory.

#Dependencies: pytorch, nibabel, numpy, kmeans_pytorch
"""

import argparse
import nibabel as nib
import numpy as np
import torch
from kmeans_pytorch import kmeans
import os

# function for loading NIfTI file
def load_nii(file_path):
    """Load a .nii file using nibabel."""
    img = nib.load(file_path)
    data = img.get_fdata()
    return img, data

# function for saving NIfTI file
def save_nii(file_path, data, nifti_img):
    """Save data as a .nii file using the provided header."""
    nib.save(nib.Nifti1Image(data, nifti_img.affine, nifti_img.header), file_path)

# function for performing K-means clustering
def perform_kmeans(data, num_clusters, device):
    """Perform K-means clustering on the data."""
    # Flatten the data for K-means
    print(f"Data shape: {data.shape}")
    flattened_data = data.reshape(-1, 1)
    print(f"Flattened data shape: {flattened_data.shape}")
    flattened_data = torch.tensor(flattened_data, dtype=torch.float32, device=device)

    # Perform K-means clustering
    cluster_ids, cluster_centers = kmeans(
        X=flattened_data, num_clusters=num_clusters, distance='euclidean', device=device
    )
    return cluster_ids, cluster_centers

# function for mixing the largest clusters
def save_clusters_as_nifti(data, cluster_ids, num_largest, output_dir, nii_img):
    """Mix the largest clusters and save the result as a NIfTI file."""
    # Count the size of each cluster
    cluster_sizes = np.bincount(cluster_ids.cpu().numpy())
    print(f"Cluster sizes: {cluster_sizes}")
    largest_clusters = np.argsort(cluster_sizes)[-num_largest:]
    print(f"Largest clusters: {largest_clusters}")

    # Reshape cluster_ids to match the original data shape
    cluster_ids_reshaped = cluster_ids.cpu().numpy().reshape(data.shape)

    # Mix the largest clusters
    mixed_data = np.zeros_like(data)
    for i in largest_clusters:
        mixed_data[cluster_ids_reshaped == i] = 1

    # save the mixed clusters as a NIfTI file
    output_path = f"{output_dir}/mixed_clusters.nii"
    save_nii(output_path, mixed_data, nii_img)
    print(f"Saved mixed clusters to {output_path}")


    # save each cluster as a NIfTI file in the output directory for the smallest cluster to the largest
    for i in np.argsort(cluster_sizes):
        cluster_data = np.zeros_like(data)
        cluster_data[cluster_ids_reshaped == i] = 1
        cluster_output_path = f"{output_dir}/cluster_{i}.nii"
        save_nii(cluster_output_path, cluster_data, nii_img)
        print(f"Saved cluster {i} to {cluster_output_path}") 

def main():
    parser = argparse.ArgumentParser(description="Perform K-means clustering on a .nii file and mix the largest clusters.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input .nii file.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to save the output .nii file.")
    parser.add_argument("-n", "--num_clusters", type=int, default=8, help="Number of clusters for K-means.")
    parser.add_argument("-l", "--num_largest", type=int, default=2, help="Number of largest clusters to mix.")
    parser.add_argument("-g", "--gpu_index", help="GPU index to be used; default is 0.", required=False, default=0)
    args = parser.parse_args()
    input_img = args.input
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_clusters = args.num_clusters
    num_largest = args.num_largest
    gpu_index = args.gpu_index

    # Load the .nii file
    img, data = load_nii(input_img)

    # set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_index}")
        print(f"Using GPU: {gpu_index}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Perform K-means clustering
    print("Performing K-means clustering...")
    cluster_ids, _ = perform_kmeans(data, num_clusters, device)
    print(f"Cluster IDs shape: {cluster_ids.shape}")

    # save the clusters as NIfTI files
    print("Saving clusters as NIfTI files...")
    save_clusters_as_nifti(data, cluster_ids, num_largest, output_dir, img)


if __name__ == "__main__":
    main()
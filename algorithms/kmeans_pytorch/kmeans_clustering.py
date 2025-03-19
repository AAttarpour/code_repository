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
-m, --brain_mask: Path to the brain mask (optional) to mask the final clusters.
-j, --metadata_json: Path to the metadata JSON file containing original dimensions for upsampling (optional).


#Outputs:
-Mixed Clusters: A NIfTI file (mixed_clusters.nii) where the largest clusters are merged.
-Individual Clusters: Each cluster is saved separately as cluster_X.nii in the output directory.
-Brain Mask: If provided, the brain mask is saved as brain_mask.nii in the output directory.
-Upsampled Clusters: If metadata JSON is provided, the mixed clusters are upsampled and saved as TIFF slices.

#Dependencies: pytorch, nibabel, numpy, kmeans_pytorch
"""

import argparse
import nibabel as nib
import numpy as np
import torch
from kmeans_pytorch import kmeans
import os
from scipy.ndimage import binary_erosion
from skimage.transform import resize
import json
import tifffile
from skimage.morphology import ball

# function for loading NIfTI file
def load_nii(file_path):
    """Load a .nii file using nibabel."""
    img = nib.load(file_path)
    data = img.get_fdata()
    return img, data

# erosion function
def erode_mask(mask, iterations=3):
    """Erode a binary mask."""
    #return binary_erosion(mask, iterations = iterations)
    return binary_erosion(mask, structure=ball(2), iterations=iterations)

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
def save_clusters_as_nifti(data, cluster_ids, num_largest, output_dir, nii_img, brain_mask):
    """Mix the largest clusters and save the result as a NIfTI file."""
    # Count the size of each cluster
    cluster_sizes = np.bincount(cluster_ids.cpu().numpy())
    print(f"Cluster sizes: {cluster_sizes}")
    largest_clusters = np.argsort(cluster_sizes)[-num_largest:]
    print(f"Largest clusters: {largest_clusters}")

    # Reshape cluster_ids to match the original data shape
    cluster_ids_reshaped = cluster_ids.cpu().numpy().reshape(data.shape)

    # Mix the largest clusters
    mixed_data = np.zeros(data.shape, dtype=np.bool_)
    for i in largest_clusters:
        mixed_data[cluster_ids_reshaped == i] = True

    # logical not of the mixed data
    mixed_data = np.logical_not(mixed_data)

    # save the mixed clusters as a NIfTI file
    if brain_mask is not None:
        mixed_data = mixed_data * brain_mask
    
    output_path = f"{output_dir}/mixed_clusters.nii"
    save_nii(output_path, mixed_data, nii_img)
    print(f"Saved mixed clusters to {output_path}")

    # save each cluster as a NIfTI file in the output directory for the smallest cluster to the largest
    for i in np.argsort(cluster_sizes):
        cluster_data = np.zeros(data.shape, dtype=np.bool_)
        cluster_data[cluster_ids_reshaped == i] = True
        # if brain_mask is not None:
        #     cluster_data = cluster_data * brain_mask
        cluster_output_path = f"{output_dir}/cluster_{i}.nii"
        save_nii(cluster_output_path, cluster_data, nii_img)
        print(f"Saved cluster {i} to {cluster_output_path}") 
    
    return mixed_data

# -------------------------------------------------------
# Save a single slice
# -------------------------------------------------------
def save_slice(z, slice_data, output_dir):
    """Save a single slice of the stitched volume."""
    raw_slice_name = f"mixed_clusters_slice{z:04d}.tif"  # Format slice name as mixed_clusters_slice0001, 0002, etc.
    img_filename = os.path.join(output_dir, raw_slice_name)
    print("Saving img: ", img_filename)
    tifffile.imwrite(
        img_filename,
        slice_data,
        metadata={
            'DimensionOrder': 'YX',
            'SizeC': 1,
            'SizeT': 1,
            'SizeX': slice_data.shape[1],
            'SizeY': slice_data.shape[0]
        }
    )
        
def main():
    parser = argparse.ArgumentParser(description="Perform K-means clustering on a .nii file and mix the largest clusters.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input .nii file.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to save the output .nii file.")
    parser.add_argument("-n", "--num_clusters", type=int, default=8, help="Number of clusters for K-means.")
    parser.add_argument("-l", "--num_largest", type=int, default=2, help="Number of largest clusters to mix.")
    parser.add_argument("-g", "--gpu_index", help="GPU index to be used; default is 0.", required=False, default=0)
    parser.add_argument("-m", "--brain_mask", help="brain labels (optional) if provided, the final clusters will be masked", required=False, default=None, type=str)
    parser.add_argument("-j", "--metadata_json", help="Path to the metadata JSON file containing original dimensions for upsampling.", required=False, default=None, type=str)

    args = parser.parse_args()
    input_img = args.input
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_clusters = args.num_clusters
    num_largest = args.num_largest
    gpu_index = args.gpu_index
    brain_mask = args.brain_mask
    metadata_json = args.metadata_json

    # Load the .nii file
    img, data = load_nii(input_img)

    if brain_mask is not None:
        brain_mask_img, brain_mask = load_nii(brain_mask)
        brain_mask = brain_mask > 0
        print(f"Brain mask shape: {brain_mask.shape}")
        # Erode the brain mask to avoid edge effects
        brain_mask = erode_mask(brain_mask, iterations=20)
        # save the brain mask as a NIfTI file
        output_path = f"{output_dir}/brain_mask.nii"
        save_nii(output_path, brain_mask, brain_mask_img)
        print(f"Saved brain mask to {output_path}")

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
    mixed_mask = save_clusters_as_nifti(data, cluster_ids, num_largest, output_dir, img, brain_mask)

    # Upsample the mixed mask if metadata JSON is provided
    if metadata_json is not None:
        print("Upscaling the mixed clusters...")
        with open(metadata_json, 'r') as f:
            metadata = json.load(f)
        original_dimensions = metadata["original_dimensions"]
        target_shape = (original_dimensions["height"], original_dimensions["width"], original_dimensions["depth"])
        
        # Upsample the mixed mask
        mixed_mask_upsampled = resize(mixed_mask, target_shape, order=0, anti_aliasing=False, preserve_range=True)
        print(f"Upsampled mixed clusters shape: {mixed_mask_upsampled.shape}")

        # Save the upsampled mixed mask
        print("Saving upsampled mixed clusters as TIFF slices...")
        output_path_tiff = f"{output_dir}/mixed_clusters_upsampled"
        if not os.path.exists(output_path_tiff):
            os.makedirs(output_path_tiff)
        for z in range(original_dimensions["depth"]):
            save_slice(z, mixed_mask_upsampled[:, :, z], output_path_tiff)  # Save each slice
        
        print(f"Saved upsampled mixed clusters as TIFF slices to {output_path_tiff}")


if __name__ == "__main__":
    main()
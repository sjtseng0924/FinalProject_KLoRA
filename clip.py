import torch
import os
import glob
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

# --- Configuration ---
# SET YOUR IMAGE PATHS HERE
REFERENCE_IMAGE = "./datasets/waterpainting/image_01_01.jpg"  # 参考图片
OUTPUT_DIRECTORY = "./outputs_cat"  # 要评估的图片目录

# For single image comparison
IMAGE_PATH_1 = "./datasets/waterpainting/image_01_01.jpg"
IMAGE_PATH_2 = "./outputs_cat/output_image_4.png"

# CHOOSE YOUR CLIP MODEL (ViT-B/32 is fast, ViT-L/14 is more accurate)
MODEL_NAME = "openai/clip-vit-base-patch32"
# --- End Configuration ---

def load_clip_model(model_name="openai/clip-vit-base-patch32"):
    """
    Load CLIP model and processor.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    return model, processor, device

def get_image_embedding(image_path, model, processor, device):
    """
    Get CLIP embedding for a single image.
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")

    # Process image
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Get image features
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        # Normalize the features
        image_features = F.normalize(image_features, p=2, dim=-1)

    return image_features

def calculate_image_similarity(image_path_1, image_path_2, model_name="openai/clip-vit-base-patch32"):
    """
    Calculate similarity between two images using CLIP.
    Returns a similarity score between 0 and 1 (cosine similarity).
    """
    # Check if files exist
    if not os.path.isfile(image_path_1):
        print(f"Error: Image not found: {image_path_1}")
        return None

    if not os.path.isfile(image_path_2):
        print(f"Error: Image not found: {image_path_2}")
        return None

    print(f"Loading CLIP model: {model_name}")
    model, processor, device = load_clip_model(model_name)

    print(f"Processing image 1: {image_path_1}")
    embedding_1 = get_image_embedding(image_path_1, model, processor, device)

    print(f"Processing image 2: {image_path_2}")
    embedding_2 = get_image_embedding(image_path_2, model, processor, device)

    # Calculate cosine similarity
    # Since features are already normalized, dot product = cosine similarity
    similarity = torch.mm(embedding_1, embedding_2.T).item()

    print("\n--- Results ---")
    print(f"Image 1: {image_path_1}")
    print(f"Image 2: {image_path_2}")
    print(f"CLIP Similarity Score: {similarity:.4f}")
    print(f"Similarity Percentage: {similarity * 100:.2f}%")

    return similarity

def get_image_files(directory):
    """Finds all common image files in a directory."""
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPG", "*.PNG"]
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(files)  # Sort for consistent ordering

def calculate_average_similarity(reference_image, output_directory, model_name="openai/clip-vit-base-patch32"):
    """
    Calculate average similarity between a reference image and all images in a directory.

    Args:
        reference_image: Path to the reference image
        output_directory: Directory containing images to compare
        model_name: CLIP model to use

    Returns:
        average_similarity: Average similarity score
        all_scores: List of individual similarity scores
    """
    # Check if reference image exists
    if not os.path.isfile(reference_image):
        print(f"Error: Reference image not found: {reference_image}")
        return None, None

    # Check if directory exists
    if not os.path.isdir(output_directory):
        print(f"Error: Directory not found: {output_directory}")
        return None, None

    # Get all image files
    image_files = get_image_files(output_directory)

    if not image_files:
        print(f"No images found in directory: {output_directory}")
        return None, None

    print(f"Found {len(image_files)} images in {output_directory}")
    print(f"Reference image: {reference_image}")
    print(f"Loading CLIP model: {model_name}\n")

    # Load model once
    model, processor, device = load_clip_model(model_name)

    # Get reference image embedding
    print(f"Processing reference image...")
    reference_embedding = get_image_embedding(reference_image, model, processor, device)

    # Calculate similarity for each image
    similarities = []
    print(f"\nProcessing {len(image_files)} images...")

    for i, img_path in enumerate(image_files, 1):
        try:
            img_embedding = get_image_embedding(img_path, model, processor, device)
            similarity = torch.mm(reference_embedding, img_embedding.T).item()
            similarities.append(similarity)
            print(f"[{i}/{len(image_files)}] {os.path.basename(img_path)}: {similarity:.4f}")
        except Exception as e:
            print(f"[{i}/{len(image_files)}] Error processing {img_path}: {e}")

    if not similarities:
        print("\nNo images were successfully processed.")
        return None, None

    # Calculate statistics
    average_similarity = sum(similarities) / len(similarities)
    max_similarity = max(similarities)
    min_similarity = min(similarities)

    print("\n" + "="*50)
    print("--- Summary Statistics ---")
    print(f"Total images processed: {len(similarities)}")
    print(f"Average similarity: {average_similarity:.4f} ({average_similarity*100:.2f}%)")
    print(f"Maximum similarity: {max_similarity:.4f} ({max_similarity*100:.2f}%)")
    print(f"Minimum similarity: {min_similarity:.4f} ({min_similarity*100:.2f}%)")
    print("="*50)

    return average_similarity, similarities

def compare_multiple_images(image_paths):
    """
    Compare multiple images pairwise and create a similarity matrix.

    Args:
        image_paths: List of image paths

    Returns:
        similarity_matrix: NxN matrix of similarity scores
    """
    n = len(image_paths)

    # Check if all files exist
    for path in image_paths:
        if not os.path.isfile(path):
            print(f"Error: Image not found: {path}")
            return None

    print(f"Loading CLIP model: {MODEL_NAME}")
    model, processor, device = load_clip_model(MODEL_NAME)

    # Get embeddings for all images
    embeddings = []
    for i, path in enumerate(image_paths):
        print(f"Processing image {i+1}/{n}: {path}")
        emb = get_image_embedding(path, model, processor, device)
        embeddings.append(emb)

    # Stack embeddings
    embeddings = torch.cat(embeddings, dim=0)

    # Compute similarity matrix
    similarity_matrix = torch.mm(embeddings, embeddings.T)

    print("\n--- Similarity Matrix ---")
    print("      ", end="")
    for i in range(n):
        print(f"Img{i+1:2d}  ", end="")
    print()

    for i in range(n):
        print(f"Img{i+1:2d} ", end="")
        for j in range(n):
            print(f"{similarity_matrix[i, j].item():5.3f} ", end="")
        print()

    return similarity_matrix.cpu().numpy()

# --- Run the script ---
if __name__ == "__main__":
    # Main task: Calculate average similarity for all images in a directory
    print("=== Calculating Average Similarity ===")
    avg_sim, all_scores = calculate_average_similarity(REFERENCE_IMAGE, OUTPUT_DIRECTORY, MODEL_NAME)

    # Optional: Compare two specific images (uncomment to use)
    # print("\n\n=== Comparing Two Images ===")
    # calculate_image_similarity(IMAGE_PATH_1, IMAGE_PATH_2, MODEL_NAME)

    # Optional: Compare multiple images with similarity matrix (uncomment to use)
    # print("\n\n=== Comparing Multiple Images ===")
    # image_list = [
    #     "./outputs_frobenius_fixed/output_image_1.png",
    #     "./outputs_cat/output_image_1.png",
    #     "./outputs_cat/output_image_2.png",
    # ]
    # compare_multiple_images(image_list)
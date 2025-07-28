import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw
import sys
import shutil
import cv2
import os
import matplotlib.cm as cm # For colormaps
import h5py
import pickle
from sklearn.neighbors import NearestNeighbors
# No explicit import for sklearn.decomposition.PCA is needed for this script
# as we are loading its parameters, not fitting a new instance.

# Ensure the ect library is installed and accessible
try:
    from ect import ECT, EmbeddedGraph
except ImportError:
    print("Error: The 'ect' library is not found. Please ensure it's installed and accessible.")
    print("Add its directory to PYTHONPATH or optionally install it using pip:")
    print("pip install ect-morphology")
    sys.exit(1)

# --- Configuration Parameters ---
BOUND_RADIUS = 1
# MODIFIED: NUM_ECT_DIRECTIONS increased to 360
NUM_ECT_DIRECTIONS = 360
# MODIFIED: ECT_THRESHOLDS to generate 360 samples
ECT_THRESHOLDS = np.linspace(0, BOUND_RADIUS, NUM_ECT_DIRECTIONS) # Using NUM_ECT_DIRECTIONS for consistency
IMAGE_SIZE = (256, 256) # Output size for all images (masks, ECT, RGB)

# MODIFIED: Set a global random seed for reproducibility
GLOBAL_RANDOM_SEED = 42

# --- Input Data Configuration ---
# MODIFIED: Paths relative to the script's execution location (assuming it's in a 'scripts' folder)
# Correctly points to ../outputs/saved_leaf_model_data
SAVED_MODEL_BASE_DIR = Path(__file__).parent.parent / "outputs"
SAVED_MODEL_SUB_DIR = SAVED_MODEL_BASE_DIR / "saved_leaf_model_data"
PCA_PARAMS_FILE = SAVED_MODEL_SUB_DIR / "leaf_pca_model_parameters.h5"
PCA_SCORES_LABELS_FILE = SAVED_MODEL_SUB_DIR / "original_pca_scores_and_geno_labels.h5"

# --- Output Data Configuration for Synthetic Samples ---
# MODIFIED: Correctly points to ../outputs/synthetic_leaf_data
SYNTHETIC_DATA_OUTPUT_DIR = SAVED_MODEL_BASE_DIR / "synthetic_leaf_data"

# Subdirectories for synthetic data
SYNTHETIC_BLADE_MASK_DIR = SYNTHETIC_DATA_OUTPUT_DIR / "blade_masks"
SYNTHETIC_BLADE_ECT_DIR = SYNTHETIC_DATA_OUTPUT_DIR / "blade_ects"
SYNTHETIC_VEIN_MASK_DIR = SYNTHETIC_DATA_OUTPUT_DIR / "vein_masks"
SYNTHETIC_VEIN_ECT_DIR = SYNTHETIC_DATA_OUTPUT_DIR / "vein_ects"
SYNTHETIC_COMBINED_VIZ_DIR = SYNTHETIC_DATA_OUTPUT_DIR / "combined_viz"
SYNTHETIC_TRANSFORMED_COORDS_DIR = SYNTHETIC_DATA_OUTPUT_DIR / "transformed_coords" # ADDED: Directory for saved transformed coordinates
SYNTHETIC_METADATA_FILE = SYNTHETIC_DATA_OUTPUT_DIR / "synthetic_metadata.csv"

# Temporary directory for ECT images before combining (will be deleted after run)
TEMP_ECT_VIZ_DIR = SYNTHETIC_DATA_OUTPUT_DIR / "temp_ect_viz"

# Pixel values for masks (consistent with user's previous code)
BACKGROUND_PIXEL = 0
BLADE_PIXEL = 1
VEIN_PIXEL = 2

# Grayscale values for output mask file
MASK_BACKGROUND_GRAY = 0    # Black background
MASK_BLADE_GRAY = 255       # White blade
MASK_VEIN_GRAY = 255        # White vein

# --- Coordinate Split Information (CRITICAL for reconstruction) ---
NUM_VEIN_COORDS = 1216
NUM_BLADE_COORDS = 456
TOTAL_COORDS = NUM_VEIN_COORDS + NUM_BLADE_COORDS # Should be 1672
FLATTENED_COORD_DIM = TOTAL_COORDS * 2 # 1672 * 2 = 3344

# --- SMOTE Augmentation Parameters ---
SAMPLES_PER_CLASS_TARGET = 400 # Desired number of synthetic samples for EACH genotype class
K_NEIGHBORS_SMOTE = 5 # Number of nearest neighbors to consider for SMOTE interpolation

# --- Helper Functions ---

def apply_transformation_with_affine_matrix(points: np.ndarray, affine_matrix: np.ndarray):
    """
    Splits the flattened coordinates back into (x, y) pairs and applies a 3x3 affine matrix.
    Points are expected as (N, 2) array.
    """
    if points.size == 0:
        return np.array([])
        
    if points.ndim == 1:
        if points.shape[0] == 2:
            points = points.reshape(1, 2)
        else:
            raise ValueError(f"Input 'points' is 1D but not a single (x,y) pair. Got shape: {points.shape}")
        
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Input 'points' must be a (N, 2) array. Got shape: {points.shape}")

    if affine_matrix.shape != (3, 3):
        raise ValueError(f"Input 'affine_matrix' must be (3, 3). Got shape: {affine_matrix.shape}")

    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # Check for potential dimension mismatch before matmul
    if points_homogeneous.shape[1] != affine_matrix.T.shape[0]:
        raise ValueError(f"matmul: Input operand 1 has a mismatch in its core dimension 0. Expected {points_homogeneous.shape[1]}, got {affine_matrix.T.shape[0]}.")

    transformed_homogeneous = points_homogeneous @ affine_matrix.T
    return transformed_homogeneous[:, :2]

def find_robust_affine_transformation_matrix(src_points: np.ndarray, dst_points: np.ndarray):
    """
    Finds a robust affine transformation matrix between source and destination points.
    It attempts to find 3 non-collinear points for cv2.getAffineTransform.
    """
    if len(src_points) < 3 or len(dst_points) < 3:
        if len(src_points) == 0:
            return np.eye(3) # Return identity for empty inputs, allows for graceful skipping
        raise ValueError(f"Need at least 3 points to compute affine transformation. Got {len(src_points)}.")

    chosen_src_pts = []
    chosen_dst_pts = []
    
    indices = np.arange(len(src_points))
    num_attempts = min(len(src_points) * (len(src_points) - 1) * (len(src_points) - 2) // 6, 1000) # Limit attempts

    for _ in range(num_attempts):
        if len(src_points) >= 3:
            selected_indices = np.random.choice(indices, 3, replace=False)
            p1_src, p2_src, p3_src = src_points[selected_indices]
            p1_dst, p2_dst, p3_dst = dst_points[selected_indices]
            
            # Check for collinearity by calculating area of triangle formed by points
            # Area is 0 if points are collinear
            area_val = (p1_src[0] - p3_src[0]) * (p2_src[1] - p1_src[1]) - \
                       (p1_src[0] - p2_src[0]) * (p3_src[1] - p1_src[1])
            
            if np.abs(area_val) > 1e-6: # Check if points are not collinear (using a small epsilon)
                chosen_src_pts = np.float32([p1_src, p2_src, p3_src])
                chosen_dst_pts = np.float32([p1_dst, p2_dst, p3_dst])
                break
    
    if len(chosen_src_pts) < 3:
        raise ValueError("Could not find 3 non-collinear points for affine transformation. Shape is likely degenerate or a line.")

    M_2x3 = cv2.getAffineTransform(chosen_src_pts, chosen_dst_pts)
    
    if M_2x3.shape != (2, 3):
        raise ValueError(f"cv2.getAffineTransform returned a non-(2,3) matrix: {M_2x3.shape}")

    affine_matrix_3x3 = np.vstack([M_2x3, [0, 0, 1]])
    
    return affine_matrix_3x3

def ect_coords_to_pixels(coords_ect: np.ndarray, image_size: tuple, bound_radius: float):
    """
    Transforms coordinates from ECT space (mathematical, Y-up, origin center, range [-R, R])
    to image pixel space (Y-down, origin top-left, range [0, IMAGE_SIZE]).
    """
    if len(coords_ect) == 0:
        return np.array([])
        
    display_x_conceptual = coords_ect[:, 1]
    display_y_conceptual = coords_ect[:, 0]

    scale_factor = image_size[0] / (2 * bound_radius)
    offset_x = image_size[0] / 2
    offset_y = image_size[1] / 2 

    pixel_x = (display_x_conceptual * scale_factor + offset_x).astype(int)
    pixel_y = (-display_y_conceptual * scale_factor + offset_y).astype(int)
    
    return np.column_stack((pixel_x, pixel_y))

def save_grayscale_shape_mask(transformed_coords: np.ndarray, pixel_identity: int, save_path: Path):
    """
    Saves a grayscale image representing a transformed contour/pixel set.
    """
    img = Image.new("L", IMAGE_SIZE, MASK_BACKGROUND_GRAY)
    draw = ImageDraw.Draw(img)

    if pixel_identity == BLADE_PIXEL:
        fill_color = MASK_BLADE_GRAY
    elif pixel_identity == VEIN_PIXEL:
        fill_color = MASK_VEIN_GRAY
    else:
        raise ValueError(f"Unknown pixel_identity: {pixel_identity}")

    if transformed_coords is not None and transformed_coords.size > 0:
        pixel_coords = ect_coords_to_pixels(transformed_coords, IMAGE_SIZE, BOUND_RADIUS)
        
        if len(pixel_coords) >= 3: # Draw as polygon if enough points
            polygon_points = [(int(p[0]), int(p[1])) for p in pixel_coords]
            draw.polygon(polygon_points, fill=fill_color)
        else: # Handle cases with 1 or 2 points (draw individual points)
            for x, y in pixel_coords:
                if 0 <= x < IMAGE_SIZE[0] and 0 <= y < IMAGE_SIZE[1]:
                    draw.point((x, y), fill=fill_color)
    
    img.save(save_path)

def save_radial_ect_image(ect_result, save_path: Path, cmap_name: str = "gray"):
    """
    Saves the radial ECT plot as an image with the specified colormap.
    """
    if ect_result is None:
        Image.new("L", IMAGE_SIZE, 0).save(save_path) # Save blank image if no ECT result
        return

    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"),
                           figsize=(IMAGE_SIZE[0]/100, IMAGE_SIZE[1]/100), dpi=100)
    thetas = ect_result.directions.thetas
    thresholds = ect_result.thresholds
    THETA, R = np.meshgrid(thetas, thresholds)
    im = ax.pcolormesh(THETA, R, ect_result.T, cmap=cmap_name) 
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim([0, BOUND_RADIUS])
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)

def create_combined_viz_from_images(ect_image_path: Path, overlay_coords: np.ndarray, 
                                     save_path: Path, overlay_color: tuple, overlay_alpha: float,
                                     overlay_type: str = "points"):
    """
    Creates a combined visualization by overlaying transformed elements (e.g., blade, veins)
    onto the ECT image. Overlayed elements are transformed to pixel space.
    """
    try:
        ect_img = Image.open(ect_image_path).convert("RGBA")
        img_width, img_height = ect_img.size

        composite_overlay = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        draw_composite = ImageDraw.Draw(composite_overlay)

        if overlay_coords is not None and overlay_coords.size > 0:
            pixel_coords = ect_coords_to_pixels(overlay_coords, IMAGE_SIZE, BOUND_RADIUS)
            fill_color_with_alpha = (overlay_color[0], overlay_color[1], overlay_color[2], int(255 * overlay_alpha))

            if overlay_type == "mask_pixels":
                if len(pixel_coords) >= 3:
                    polygon_points = [(int(p[0]), int(p[1])) for p in pixel_coords]
                    draw_composite.polygon(polygon_points, fill=fill_color_with_alpha)
                else: # Draw points if not enough for a polygon
                    for x, y in pixel_coords:
                        if 0 <= x < img_width and 0 <= y < img_height:
                            draw_composite.point((x, y), fill=fill_color_with_alpha)

            elif overlay_type == "points":
                point_radius = 2
                for x, y in pixel_coords:
                    if 0 <= x < img_width and 0 <= y < img_height:
                        draw_composite.ellipse([x - point_radius, y - point_radius,
                                                x + point_radius, y + point_radius],
                                               fill=fill_color_with_alpha)
            
        final_combined_img = Image.alpha_composite(ect_img, composite_overlay).convert("RGB")
        final_combined_img.save(save_path)

    except FileNotFoundError:
        print(f"Error: ECT image file not found: {ect_image_path}")
    except Exception as e:
        print(f"Error creating combined visualization for {ect_image_path.stem}: {e}")

def rotate_coords_2d(coords: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotates 2D coordinates (Nx2 array) around the origin (0,0).
    """
    if coords.size == 0:
        return np.array([])
    
    angle_rad = np.deg2rad(angle_deg)
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    
    rotated_coords = coords @ rot_matrix.T
    return rotated_coords

# --- Core Logic for Synthetic Data Generation ---

def load_pca_model_data(pca_params_file: Path, pca_scores_labels_file: Path):
    """
    Loads PCA model parameters and original PCA scores/labels.
    """
    pca_data = {}
    with h5py.File(pca_params_file, 'r') as f:
        pca_data['components'] = f['components'][:]
        pca_data['mean'] = f['mean'][:]
        pca_data['explained_variance'] = f['explained_variance'][:]
        pca_data['n_components'] = f.attrs['n_components']
        
    with h5py.File(pca_scores_labels_file, 'r') as f:
        pca_data['original_pca_scores'] = f['pca_scores'][:]
        pca_data['original_geno_labels'] = np.array([s.decode('utf-8') for s in f['geno_labels'][:]])
        
    print(f"Loaded PCA model parameters from {pca_params_file}.")
    print(f"Loaded original PCA scores and labels from {pca_scores_labels_file}.")
    return pca_data

def generate_synthetic_pca_samples(pca_data: dict, samples_per_class_target: int, k_neighbors: int, random_seed: int = None):
    """
    Generates synthetic PCA samples using a SMOTE-like approach based on class labels.
    """
    if random_seed is not None:
        np.random.seed(random_seed) # Set seed for SMOTE sampling

    print(f"\nStarting synthetic data generation (SMOTE-like) with {samples_per_class_target} samples per class...")
    
    original_pca_scores = pca_data['original_pca_scores']
    original_geno_labels = pd.Series(pca_data['original_geno_labels'])
    
    synthetic_X_pca = []
    synthetic_y = []
    
    class_counts = original_geno_labels.value_counts()
    all_classes = class_counts.index.tolist()
    
    total_generated_samples = 0

    for class_name in all_classes:
        class_pca_samples = original_pca_scores[original_geno_labels == class_name]
        
        if len(class_pca_samples) < 2:
            print(f"Warning: Class '{class_name}' has less than 2 samples ({len(class_pca_samples)}). Cannot perform SMOTE-like augmentation for interpolation. Skipping this class for synthetic generation.")
            continue
            
        n_neighbors_for_class = min(len(class_pca_samples) - 1, k_neighbors)
        if n_neighbors_for_class < 1:  
             print(f"Warning: Class '{class_name}' has insufficient samples ({len(class_pca_samples)}) for meaningful NearestNeighbors calculation (k={k_neighbors}). Skipping.")
             continue

        nn = NearestNeighbors(n_neighbors=n_neighbors_for_class + 1).fit(class_pca_samples)
        
        generated_count = 0
        while generated_count < samples_per_class_target:
            idx_in_class_samples = np.random.randint(0, len(class_pca_samples))
            sample = class_pca_samples[idx_in_class_samples]
            
            distances, indices = nn.kneighbors(sample.reshape(1, -1))
            
            available_neighbors_indices_in_class_pca = indices[0][1:] # Exclude the sample itself
            
            if len(available_neighbors_indices_in_class_pca) == 0:
                continue 
                
            neighbor_idx_in_class_pca_samples = np.random.choice(available_neighbors_indices_in_class_pca)
            neighbor = class_pca_samples[neighbor_idx_in_class_pca_samples]
            
            alpha = np.random.rand()
            synthetic_pca_sample = sample + alpha * (neighbor - sample)
            
            synthetic_X_pca.append(synthetic_pca_sample)
            synthetic_y.append(class_name)
            generated_count += 1
            total_generated_samples += 1
            
    print(f"Finished generating {total_generated_samples} synthetic samples across {len(all_classes)} classes.")
    return np.array(synthetic_X_pca), synthetic_y

def inverse_transform_pca(pca_scores: np.ndarray, pca_components: np.ndarray, pca_mean: np.ndarray):
    """
    Inverse transforms PCA scores back to the original flattened coordinate space.
    Assumes pca_components are (n_components, n_features) and pca_mean is (n_features,).
    """
    reconstructed_data = np.dot(pca_scores, pca_components) + pca_mean
    return reconstructed_data

def process_synthetic_leaf(
    synthetic_id: str,
    class_label: str,
    flat_coords: np.ndarray,
    ect_calculator: ECT,
    output_dirs: dict, # This dict now includes 'transformed_coords_dir'
    metadata_records: list,
    random_seed: int = None
):
    """
    Processes a single synthetic leaf's flattened coordinates to produce masks, ECTs, viz, and transformed coordinates.
    """
    if random_seed is not None:
        np.random.seed(random_seed) # Set seed for random rotation within processing

    current_metadata = {
        "synthetic_id": synthetic_id,
        "class_label": class_label,
        "is_processed_valid": False,
        "reason_skipped": "",
        "num_blade_coords": 0,
        "num_vein_coords": 0,
        "file_blade_mask": "",
        "file_blade_ect": "",
        "file_vein_mask": "",
        "file_vein_ect": "",
        "file_combined_viz_blade_on_ect": "", # Blade mask on blade ECT
        "file_combined_viz_vein_on_ect": "",    # Vein mask on vein ECT
        "file_transformed_blade_coords": "", # ADDED to metadata
        "file_transformed_vein_coords": ""   # ADDED to metadata
    }

    temp_blade_ect_inferno_path = None  
    temp_vein_ect_inferno_path = None

    transformed_blade_pixels = np.array([]) # Initialize outside try for finally block
    transformed_vein_pixels_for_vein_ect_viz = np.array([]) # Initialize outside try for finally block

    try:
        # Reshape to (N, 2) and split into vein and blade coordinates
        coords_2d = flat_coords.reshape(TOTAL_COORDS, 2)
        raw_vein_coords = coords_2d[:NUM_VEIN_COORDS]
        raw_blade_coords = coords_2d[NUM_VEIN_COORDS:]

        current_metadata["num_blade_coords"] = len(raw_blade_coords)
        current_metadata["num_vein_coords"] = len(raw_vein_coords)

        # Apply Random Rotation to Raw Coordinates
        RANDOM_ROTATION_RANGE_DEG = (-180, 180)
        random_angle_deg = np.random.uniform(*RANDOM_ROTATION_RANGE_DEG)

        # Rotate the raw coordinates for both blade and vein
        rotated_raw_blade_coords = rotate_coords_2d(raw_blade_coords, random_angle_deg)
        rotated_raw_vein_coords = rotate_coords_2d(raw_vein_coords, random_angle_deg)

        # Validate Blade Coordinates for ECT
        if len(np.unique(rotated_raw_blade_coords, axis=0)) < 3:
            raise ValueError(f"Synthetic leaf '{synthetic_id}' has too few distinct blade points ({len(np.unique(rotated_raw_blade_coords, axis=0))}) for ECT calculation or polygon drawing.")
        
        # Process Blade: Calculate its ECT and derive its transformation matrix
        G_blade = EmbeddedGraph()
        G_blade.add_cycle(rotated_raw_blade_coords) # Use rotated coords
        
        original_G_blade_coord_matrix = G_blade.coord_matrix.copy()

        G_blade.center_coordinates(center_type="origin")
        G_blade.transform_coordinates()
        G_blade.scale_coordinates(BOUND_RADIUS)

        if G_blade.coord_matrix.shape[0] < 3 or np.all(G_blade.coord_matrix == 0):
            raise ValueError(f"Degenerate blade shape for '{synthetic_id}' after ECT transformation.")

        ect_affine_matrix_blade = find_robust_affine_transformation_matrix(original_G_blade_coord_matrix, G_blade.coord_matrix)
        transformed_blade_pixels = apply_transformation_with_affine_matrix(rotated_raw_blade_coords, ect_affine_matrix_blade)
        
        ect_result_blade = ect_calculator.calculate(G_blade)

        # Process Vein for CNN Input Mask (aligned to Blade's ECT space)
        transformed_vein_pixels_for_cnn_mask = apply_transformation_with_affine_matrix(rotated_raw_vein_coords, ect_affine_matrix_blade)

        # Process Vein for its OWN ECT (for training guidance and its combined viz)
        ect_result_vein = None
        if len(rotated_raw_vein_coords) > 0:
            G_vein = EmbeddedGraph()
            if len(rotated_raw_vein_coords) >= 2: 
                G_vein.add_cycle(rotated_raw_vein_coords) # Still using add_cycle, assuming vein points can form a closed path or are dense enough
            else: 
                G_vein.add_points(rotated_raw_vein_coords) # For single points, add directly

            if G_vein.coord_matrix.shape[0] > 0:
                original_G_vein_coord_matrix = G_vein.coord_matrix.copy()

                G_vein.center_coordinates(center_type="origin")
                G_vein.transform_coordinates()
                G_vein.scale_coordinates(BOUND_RADIUS)

                if G_vein.coord_matrix.shape[0] < 1 or np.all(G_vein.coord_matrix == 0):
                    ect_result_vein = None
                else:
                    ect_result_vein = ect_calculator.calculate(G_vein)
                    ect_affine_matrix_vein = find_robust_affine_transformation_matrix(original_G_vein_coord_matrix, G_vein.coord_matrix)
                    transformed_vein_pixels_for_vein_ect_viz = apply_transformation_with_affine_matrix(rotated_raw_vein_coords, ect_affine_matrix_vein)
        
        # Define Output Paths
        blade_mask_path = output_dirs['blade_masks'] / f"{synthetic_id}_blade_mask.png"
        blade_ect_path = output_dirs['blade_ects'] / f"{synthetic_id}_blade_ect.png"
        vein_mask_path = output_dirs['vein_masks'] / f"{synthetic_id}_vein_mask.png"
        vein_ect_path = output_dirs['vein_ects'] / f"{synthetic_id}_vein_ect.png"
        
        combined_viz_blade_on_ect_path = output_dirs['combined_viz'] / f"{synthetic_id}_combined_blade.png"
        combined_viz_vein_on_vein_ect_path = output_dirs['combined_viz'] / f"{synthetic_id}_combined_vein.png"

        blade_coords_path = output_dirs['transformed_coords_dir'] / f"{synthetic_id}_blade_coords.npy" # ADDED path
        vein_coords_path = output_dirs['transformed_coords_dir'] / f"{synthetic_id}_vein_coords.npy"   # ADDED path

        temp_blade_ect_inferno_path = output_dirs['temp_ect_viz'] / f"{synthetic_id}_blade_ect_inferno.png"
        temp_vein_ect_inferno_path = output_dirs['temp_ect_viz'] / f"{synthetic_id}_vein_ect_inferno.png"

        # Save CNN Input/Output Files (Masks & Grayscale ECTs)
        save_grayscale_shape_mask(transformed_blade_pixels, BLADE_PIXEL, blade_mask_path)
        save_radial_ect_image(ect_result_blade, blade_ect_path, cmap_name="gray")
        
        save_grayscale_shape_mask(transformed_vein_pixels_for_cnn_mask, VEIN_PIXEL, vein_mask_path)
        save_radial_ect_image(ect_result_vein, vein_ect_path, cmap_name="gray")

        # Save Transformed Coordinates # ADDED SECTION
        if transformed_blade_pixels.size > 0:
            np.save(blade_coords_path, transformed_blade_pixels)
            current_metadata["file_transformed_blade_coords"] = str(blade_coords_path.relative_to(SYNTHETIC_DATA_OUTPUT_DIR))
        if transformed_vein_pixels_for_vein_ect_viz.size > 0:
            np.save(vein_coords_path, transformed_vein_pixels_for_vein_ect_viz)
            current_metadata["file_transformed_vein_coords"] = str(vein_coords_path.relative_to(SYNTHETIC_DATA_OUTPUT_DIR))
        # END ADDED SECTION

        # Create Combined Visualizations for Verification
        save_radial_ect_image(ect_result_blade, temp_blade_ect_inferno_path, cmap_name="inferno")
        create_combined_viz_from_images(
            temp_blade_ect_inferno_path, transformed_blade_pixels, combined_viz_blade_on_ect_path, 
            overlay_color=(255, 255, 255), overlay_alpha=0.2, overlay_type="mask_pixels"
        )
        
        save_radial_ect_image(ect_result_vein, temp_vein_ect_inferno_path, cmap_name="inferno")
        create_combined_viz_from_images(
            temp_vein_ect_inferno_path, transformed_vein_pixels_for_vein_ect_viz, combined_viz_vein_on_vein_ect_path,
            overlay_color=(255, 255, 255), overlay_alpha=1.0, overlay_type="mask_pixels"
        )

        # Populate metadata for successful processing
        current_metadata["is_processed_valid"] = True
        current_metadata["file_blade_mask"] = str(blade_mask_path.relative_to(SYNTHETIC_DATA_OUTPUT_DIR))
        current_metadata["file_blade_ect"] = str(blade_ect_path.relative_to(SYNTHETIC_DATA_OUTPUT_DIR))
        current_metadata["file_vein_mask"] = str(vein_mask_path.relative_to(SYNTHETIC_DATA_OUTPUT_DIR))
        current_metadata["file_vein_ect"] = str(vein_ect_path.relative_to(SYNTHETIC_DATA_OUTPUT_DIR))
        current_metadata["file_combined_viz_blade_on_ect"] = str(combined_viz_blade_on_ect_path.relative_to(SYNTHETIC_DATA_OUTPUT_DIR))
        current_metadata["file_combined_viz_vein_on_ect"] = str(combined_viz_vein_on_vein_ect_path.relative_to(SYNTHETIC_DATA_OUTPUT_DIR))

    except Exception as e:
        current_metadata["reason_skipped"] = f"Processing failed: {e}"
        print(f"Skipping synthetic leaf '{synthetic_id}' due to error: {e}")

    finally:
        metadata_records.append(current_metadata)
        # Clean up temporary ECT images
        if temp_blade_ect_inferno_path and temp_blade_ect_inferno_path.exists():
            os.remove(temp_blade_ect_inferno_path)
        if temp_vein_ect_inferno_path and temp_vein_ect_inferno_path.exists():
            os.remove(temp_vein_ect_inferno_path)


def main_synthetic_generation(clear_existing_data: bool = True):
    """
    Main function to orchestrate synthetic leaf data generation.
    """
    # Set global random seed at the start of the main function for overall reproducibility
    np.random.seed(GLOBAL_RANDOM_SEED)

    print("--- Starting Synthetic Leaf Data Generation Pipeline ---")

    # --- 1. Setup Output Directories ---
    if clear_existing_data and SYNTHETIC_DATA_OUTPUT_DIR.exists():
        print(f"Clearing existing synthetic data output directory: {SYNTHETIC_DATA_OUTPUT_DIR}")
        shutil.rmtree(SYNTHETIC_DATA_OUTPUT_DIR)
        
    SYNTHETIC_DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_BLADE_MASK_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_BLADE_ECT_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_VEIN_MASK_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_VEIN_ECT_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_COMBINED_VIZ_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_TRANSFORMED_COORDS_DIR.mkdir(parents=True, exist_ok=True) # ADDED: Create transformed coords dir
    TEMP_ECT_VIZ_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Created synthetic data output directories.")

    # --- 2. Load PCA Data ---
    pca_data = load_pca_model_data(PCA_PARAMS_FILE, PCA_SCORES_LABELS_FILE)

    # --- 3. Generate Synthetic PCA Samples ---
    # Pass the random_seed to generate_synthetic_pca_samples
    synthetic_X_pca, synthetic_y = generate_synthetic_pca_samples(
        pca_data, SAMPLES_PER_CLASS_TARGET, K_NEIGHBORS_SMOTE, random_seed=GLOBAL_RANDOM_SEED
    )

    # --- 4. Initialize ECT Calculator ---
    ect_calculator = ECT(num_dirs=NUM_ECT_DIRECTIONS, thresholds=ECT_THRESHOLDS, bound_radius=BOUND_RADIUS)
    print("Initialized ECT calculator.")

    metadata_records = []
    total_synthetic_samples = len(synthetic_X_pca)
    print(f"\nProcessing {total_synthetic_samples} synthetic leaves...")

    # Define the dictionary of output directories to pass to the processing function
    output_dirs_for_process = {
        'blade_masks': SYNTHETIC_BLADE_MASK_DIR,
        'blade_ects': SYNTHETIC_BLADE_ECT_DIR,
        'vein_masks': SYNTHETIC_VEIN_MASK_DIR,
        'vein_ects': SYNTHETIC_VEIN_ECT_DIR,
        'combined_viz': SYNTHETIC_COMBINED_VIZ_DIR,
        'temp_ect_viz': TEMP_ECT_VIZ_DIR,
        'transformed_coords_dir': SYNTHETIC_TRANSFORMED_COORDS_DIR # ADDED
    }

    # --- 5. Process Each Synthetic Sample ---
    for i in range(total_synthetic_samples):
        synthetic_id = f"synthetic_leaf_{i:05d}"
        class_label = synthetic_y[i]
        synthetic_pca_score = synthetic_X_pca[i]

        print(f"Processing synthetic leaf {i+1}/{total_synthetic_samples} ({synthetic_id}, Class: {class_label})")

        # Inverse transform PCA score to get flattened coordinates
        flat_coords = inverse_transform_pca(synthetic_pca_score.reshape(1, -1),
                                             pca_data['components'],
                                             pca_data['mean']).flatten()
        
        # Ensure the reconstructed flattened coordinates have the expected dimension
        if len(flat_coords) != FLATTENED_COORD_DIM:
            print(f"Error: Reconstructed flattened coordinates for {synthetic_id} have unexpected dimension {len(flat_coords)}. Expected {FLATTENED_COORD_DIM}. Skipping.")
            metadata_records.append({
                "synthetic_id": synthetic_id,
                "class_label": class_label,
                "is_processed_valid": False,
                "reason_skipped": f"Reconstructed coords dimension mismatch: {len(flat_coords)} != {FLATTENED_COORD_DIM}",
                "num_blade_coords": 0, "num_vein_coords": 0,
                "file_blade_mask": "", "file_blade_ect": "", "file_vein_mask": "", "file_vein_ect": "",
                "file_combined_viz_blade_on_ect": "", "file_combined_viz_vein_on_ect": "",
                "file_transformed_blade_coords": "", "file_transformed_vein_coords": "" # ADDED
            })
            continue # Skip to the next sample

        # Pass the random_seed to process_synthetic_leaf for reproducibility of random rotation
        process_synthetic_leaf(
            synthetic_id,
            class_label,
            flat_coords,
            ect_calculator,
            output_dirs_for_process, # Use the updated dictionary
            metadata_records,
            random_seed=GLOBAL_RANDOM_SEED + i # Vary seed per sample for rotation diversity, but keep overall reproducibility
        )

    # --- 6. Save Metadata ---
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(SYNTHETIC_METADATA_FILE, index=False)
    print(f"\nSaved synthetic leaf metadata to {SYNTHETIC_METADATA_FILE}")

    # --- 7. Clean up temporary directory ---
    # Only remove the temp directory if it's empty after processing
    if TEMP_ECT_VIZ_DIR.exists() and not os.listdir(TEMP_ECT_VIZ_DIR):
        shutil.rmtree(TEMP_ECT_VIZ_DIR)
        print(f"Cleaned up empty temporary directory: {TEMP_ECT_VIZ_DIR}")
    elif TEMP_ECT_VIZ_DIR.exists(): # If it still contains files, warn but don't delete
        print(f"Warning: Temporary directory {TEMP_ECT_VIZ_DIR} is not empty. Not deleting.")

    print("\n--- Synthetic Leaf Data Generation Pipeline Finished ---")

if __name__ == "__main__":
    main_synthetic_generation()
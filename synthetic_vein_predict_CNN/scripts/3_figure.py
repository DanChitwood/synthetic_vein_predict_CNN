import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import os

# --- Global Random Seed for Reproducibility for sample selection ---
GLOBAL_RANDOM_SEED_FIGURE = 7 # A different seed for figure reproducibility

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed for figure generation set to {seed}")


# --- Configuration Parameters ---
IMAGE_SIZE = (256, 256) # Consistent with synthetic data generation and training
NUM_FIGURE_SAMPLES = 40 # Number of random synthetic leaves to display

# Paths relative to the script's execution location (assuming it's in a 'scripts' folder)
BASE_OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
SYNTHETIC_DATA_OUTPUT_DIR = BASE_OUTPUTS_DIR / "synthetic_leaf_data"
SYNTHETIC_METADATA_FILE = SYNTHETIC_DATA_OUTPUT_DIR / "synthetic_metadata.csv"

MODEL_SAVE_DIR = BASE_OUTPUTS_DIR / "unet_model_results"
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "trained_leaf_segmentation_model.pth"

FIGURE_SAVE_DIR = BASE_OUTPUTS_DIR / "figure" # NEW: Folder for the final figure
FIGURE_SAVE_PATH = FIGURE_SAVE_DIR / "synthetic_vein_predictions_figure.png"


# --- Device Configuration (must match training script for loading) ---
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# --- 1. Custom Dataset Class (modified to load blade coordinates using metadata column) ---
class SyntheticLeafDataset(Dataset):
    def __init__(self, metadata_file: Path, base_dir: Path):
        self.metadata_df = pd.read_csv(metadata_file)
        # Ensure 'is_processed_valid' column exists and filter based on it
        if 'is_processed_valid' in self.metadata_df.columns:
            self.metadata_df = self.metadata_df[self.metadata_df['is_processed_valid']].reset_index(drop=True)
        else:
            print("Warning: 'is_processed_valid' column not found in metadata. Proceeding with all samples.")
        
        self.base_dir = base_dir

        if self.metadata_df.empty:
            raise ValueError(f"No valid processed samples found in metadata file: {metadata_file} after filtering (if 'is_processed_valid' exists).")
            
        print(f"Loaded {len(self.metadata_df)} valid synthetic samples for visualization.")

        # Determine filename column for image_name
        if 'image_name' in self.metadata_df.columns:
            self.filename_col = 'image_name'
        elif 'synthetic_id' in self.metadata_df.columns:
            self.filename_col = 'synthetic_id'
        else:
            raise ValueError(
                "Neither 'image_name' nor 'synthetic_id' column found in your metadata CSV. "
                "Please check your 'synthetic_metadata.csv' file to identify the correct column "
                "for unique image identifiers."
            )
        print(f"Using '{self.filename_col}' as the filename column from metadata.")

        # Ensure coordinate columns exist
        if 'file_transformed_blade_coords' not in self.metadata_df.columns:
            raise ValueError("Required column 'file_transformed_blade_coords' not found in metadata.")

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        image_name = row[self.filename_col]
        
        # Load Blade ECT (Input 1)
        blade_ect_path = self.base_dir / row['file_blade_ect']
        blade_ect_img = Image.open(blade_ect_path).convert("L")
        
        # Load Blade Mask (Input 2)
        blade_mask_path = self.base_dir / row['file_blade_mask']
        blade_mask_img = Image.open(blade_mask_path).convert("L")

        # Load Vein Mask (Target)
        vein_mask_path = self.base_dir / row['file_vein_mask']
        vein_mask_img = Image.open(vein_mask_path).convert("L")

        # Load Blade Coordinates using the specified metadata column
        blade_coords_path = self.base_dir / row['file_transformed_blade_coords']
        
        if not blade_coords_path.exists():
            raise FileNotFoundError(f"Blade coordinates file not found for {image_name} at {blade_coords_path}. "
                                    f"Please ensure the path in 'file_transformed_blade_coords' column is correct and the file exists.")
        
        blade_coords = np.load(blade_coords_path) # Assumed to be (N, 2) numpy array of (x, y) coordinates

        blade_ect_np = np.array(blade_ect_img) / 255.0
        blade_mask_np = np.array(blade_mask_img) / 255.0
        vein_mask_np = np.array(vein_mask_img) / 255.0

        inputs_stacked = np.stack([blade_ect_np, blade_mask_np], axis=0)
        inputs_tensor = torch.from_numpy(inputs_stacked).float()

        target_tensor = torch.from_numpy(vein_mask_np).float().unsqueeze(0) # Keep channel dim for consistent model input/output

        return inputs_tensor, target_tensor, image_name, blade_coords


# --- 2. Model Architecture: U-Net (identical to training script) ---
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels) 
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels) 

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 if necessary to match x2's spatial dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels_in, n_classes_out, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_classes_out = n_classes_out
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // (2 if bilinear else 1)) 
        
        self.up1 = Up(1024, 512 // (2 if bilinear else 1), bilinear)
        self.up2 = Up(512, 256 // (2 if bilinear else 1), bilinear)
        self.up3 = Up(256, 128 // (2 if bilinear else 1), bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1) # Corrected: from 'x,1' to 'x1'
        logits = self.outc(x)
        return logits


# --- Function to generate the combined figure ---
def generate_figure(model, dataset, num_samples, figure_save_path: Path):
    model.eval() # Set to evaluation mode
    figure_save_path.parent.mkdir(parents=True, exist_ok=True) # Create figure directory if it doesn't exist

    print(f"\nGenerating and saving the main figure to: {figure_save_path}")

    # Select random samples
    if num_samples > len(dataset):
        print(f"Warning: Requested {num_samples} samples, but only {len(dataset)} available. Using all available samples.")
        selected_indices = list(range(len(dataset)))
    else:
        selected_indices = random.sample(range(len(dataset)), num_samples)

    # Figure layout
    # 4 leaves per row, 10 rows. Each leaf has 2 panels. So 4*2 = 8 columns
    num_leaves_per_row = 4
    num_rows = num_samples // num_leaves_per_row
    if num_samples % num_leaves_per_row != 0:
        num_rows += 1 # Add an extra row if not perfectly divisible

    # Adjusted figsize and dpi for high-quality output
    fig, axes = plt.subplots(num_rows, num_leaves_per_row * 2, figsize=(num_leaves_per_row * 3.5, num_rows * 2), dpi=300)
    axes = axes.flatten() # Flatten array for easy indexing

    # Define colors (RGB 0-255)
    BACKGROUND_COLOR = [128, 128, 128] # Gray
    TP_COLOR = [255, 255, 255] # White for True Positives
    FP_COLOR = [255, 0, 255]   # Magenta for False Positives
    FN_COLOR = [65, 105, 225]  # DodgerBlue for False Negatives

    with torch.no_grad():
        for i, idx in enumerate(selected_indices):
            inputs, target_mask, image_name, blade_coords = dataset[idx]
            
            # Add batch dimension and move to device
            inputs_batch = inputs.unsqueeze(0).to(DEVICE)

            predicted_logits = model(inputs_batch)
            predicted_mask = torch.sigmoid(predicted_logits)
            predicted_mask = (predicted_mask > 0.5).float()

            # Move tensors back to CPU and convert to numpy for plotting
            blade_ect_viz = inputs[0].cpu().numpy() # inputs[0] is Blade ECT
            ground_truth_mask_viz = target_mask.squeeze().cpu().numpy() # Remove channel dim
            predicted_mask_viz = predicted_mask.squeeze().cpu().numpy() # Remove channel dim

            # --- Apply 90-degree CCW rotation followed by 180-degree rotation ---
            # Original blade_coords are (x, y)
            # Step 1: 90-deg CCW rotation in Cartesian (Y-up): (x, y) -> (-y, x)
            # Step 2: 180-deg rotation: (-y, x) -> (-(-y), -x) = (y, -x)
            
            final_x_normalized = blade_coords[:, 1]  # New X is original Y
            final_y_normalized = -blade_coords[:, 0] # New Y is negative of original X

            # Apply scaling and translation to pixel coordinates
            pixel_blade_coords_x = (final_x_normalized * (IMAGE_SIZE[0] / 2)) + (IMAGE_SIZE[0] / 2)
            pixel_blade_coords_y = (final_y_normalized * (IMAGE_SIZE[1] / 2)) + (IMAGE_SIZE[1] / 2)
            
            pixel_blade_coords = np.stack((pixel_blade_coords_x, pixel_blade_coords_y), axis=1)

            # --- Left Panel: Blade ECT with Blade Outline ---
            ax_left = axes[i * 2]
            ax_left.imshow(blade_ect_viz, cmap='gray_r') # Reverse grayscale
            
            # Plot blade outline using transformed pixel coordinates
            if pixel_blade_coords.shape[0] > 0:
                ax_left.plot(pixel_blade_coords[:, 0], pixel_blade_coords[:, 1], color='black', linewidth=0.8, zorder=10)
            
            ax_left.axis('off')

            # --- Right Panel: TP, FP, FN Visualization + Blade Outline ---
            ax_right = axes[i * 2 + 1]
            
            # Initialize with background color (gray)
            composite_colored_mask = np.full((*ground_truth_mask_viz.shape, 3), BACKGROUND_COLOR, dtype=np.uint8)

            # Convert to boolean masks for clear logic
            gt_mask_bool = ground_truth_mask_viz == 1
            pred_mask_bool = predicted_mask_viz == 1

            # False Negatives (GT is 1, Pred is 0) - DodgerBlue
            composite_colored_mask[gt_mask_bool & ~pred_mask_bool] = FN_COLOR
            
            # False Positives (GT is 0, Pred is 1) - Magenta
            composite_colored_mask[~gt_mask_bool & pred_mask_bool] = FP_COLOR
            
            # True Positives (GT is 1, Pred is 1) - White
            composite_colored_mask[gt_mask_bool & pred_mask_bool] = TP_COLOR
            
            ax_right.imshow(composite_colored_mask)

            # Plot blade outline on the right panel too using transformed pixel coordinates
            if pixel_blade_coords.shape[0] > 0:
                ax_right.plot(pixel_blade_coords[:, 0], pixel_blade_coords[:, 1], color='black', linewidth=0.8, zorder=10)
            
            ax_right.axis('off')

    # Remove any unused subplots if num_samples is not a perfect multiple of num_leaves_per_row
    for j in range(len(selected_indices) * 2, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=0.0) # Set pad to 0 for minimal spacing between subplots
    plt.savefig(figure_save_path, bbox_inches='tight', pad_inches=0.0) # Save with no extra padding
    plt.close(fig) # Close the figure to free up memory
    print(f"Figure saved successfully to {figure_save_path}")


# --- Main Execution ---
if __name__ == "__main__":
    set_seed(GLOBAL_RANDOM_SEED_FIGURE) # Set seed for random sample selection

    # Create dataset
    try:
        dataset = SyntheticLeafDataset(SYNTHETIC_METADATA_FILE, SYNTHETIC_DATA_OUTPUT_DIR)
    except ValueError as e:
        print(f"Error creating dataset: {e}")
        print("Please ensure the synthetic data generation script has been run successfully and the metadata file exists.")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error finding a required file: {e}")
        print("Please ensure all expected data files (ECTs, masks, and transformed_coords for blade outlines) exist and paths in metadata are correct.")
        sys.exit(1)

    if not MODEL_SAVE_PATH.exists():
        print(f"Error: Trained model not found at {MODEL_SAVE_PATH}")
        print("Please ensure the training script (2_train_unet.py) has been run successfully.")
        sys.exit(1)

    if len(dataset) < NUM_FIGURE_SAMPLES:
        print(f"Warning: Not enough samples ({len(dataset)}) available to select {NUM_FIGURE_SAMPLES} for the figure.")
        print("The figure will be generated with all available samples.")

    # Initialize model
    inference_model = UNet(n_channels_in=2, n_classes_out=1).to(DEVICE)
    inference_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    inference_model.eval() # Set to evaluation mode

    print(f"Loaded model from: {MODEL_SAVE_PATH}")

    # Generate the combined figure
    generate_figure(inference_model, dataset, NUM_FIGURE_SAMPLES, FIGURE_SAVE_PATH)

    print("\n--- Figure generation complete ---")
    print(f"Check the '{FIGURE_SAVE_DIR}' folder for the saved figure.")
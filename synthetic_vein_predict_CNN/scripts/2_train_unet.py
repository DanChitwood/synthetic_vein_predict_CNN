import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms 
import os
import pandas as pd # Already imported, good!
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm.auto import tqdm 

# --- Global Random Seed for Reproducibility ---
GLOBAL_RANDOM_SEED = 42

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

# --- Configuration Parameters (consistent with synthetic data generation) ---
IMAGE_SIZE = (256, 256) # Output size for all images (masks, ECT, RGB)

# Paths relative to the script's execution location (assuming it's in a 'scripts' folder)
BASE_OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
SYNTHETIC_DATA_OUTPUT_DIR = BASE_OUTPUTS_DIR / "synthetic_leaf_data"
SYNTHETIC_METADATA_FILE = SYNTHETIC_DATA_OUTPUT_DIR / "synthetic_metadata.csv"

# Model save path now points to ../outputs/unet_model_results
MODEL_SAVE_DIR = BASE_OUTPUTS_DIR / "unet_model_results"
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "trained_leaf_segmentation_model.pth"
DICE_HISTORY_SAVE_PATH = MODEL_SAVE_DIR / "validation_dice_history.csv" # MODIFIED: Changed to .csv

# --- Training Parameters (Adjust these!) ---
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 0.0005 
VALIDATION_SPLIT_RATIO = 0.2 # 20% of data for validation
NUM_WORKERS = 0 # Set to 0 for initial stability, increase for performance if needed
                      # (Note: num_workers > 0 can be tricky with MPS backend)

# --- Device Configuration ---
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# --- 1. Custom Dataset Class ---
class SyntheticLeafDataset(Dataset):
    def __init__(self, metadata_file: Path, base_dir: Path, transform=None):
        self.metadata_df = pd.read_csv(metadata_file)
        # Filter for successfully processed samples
        self.metadata_df = self.metadata_df[self.metadata_df['is_processed_valid']].reset_index(drop=True)
        self.base_dir = base_dir
        self.transform = transform

        if self.metadata_df.empty:
            raise ValueError(f"No valid processed samples found in metadata file: {metadata_file}")
            
        print(f"Loaded {len(self.metadata_df)} valid synthetic samples for training.")

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        
        # Load Blade ECT (Input 1)
        blade_ect_path = self.base_dir / row['file_blade_ect']
        blade_ect_img = Image.open(blade_ect_path).convert("L") # Ensure grayscale
        
        # Load Blade Mask (Input 2)
        blade_mask_path = self.base_dir / row['file_blade_mask']
        blade_mask_img = Image.open(blade_mask_path).convert("L") # Ensure grayscale

        # Load Vein Mask (Target)
        vein_mask_path = self.base_dir / row['file_vein_mask']
        vein_mask_img = Image.open(vein_mask_path).convert("L") # Ensure grayscale

        # Convert to numpy arrays first for channel stacking, then to PIL for transform if needed, then to tensor
        blade_ect_np = np.array(blade_ect_img) / 255.0 # Normalize to [0, 1]
        blade_mask_np = np.array(blade_mask_img) / 255.0 # Normalize to [0, 1]
        vein_mask_np = np.array(vein_mask_img) / 255.0 # Normalize to [0, 1]

        # Stack inputs along a new channel dimension
        # (H, W, 2) -> (2, H, W) for PyTorch convention
        inputs_stacked = np.stack([blade_ect_np, blade_mask_np], axis=0) # Shape: (2, H, W)
        inputs_tensor = torch.from_numpy(inputs_stacked).float()

        # Target mask is single channel
        target_tensor = torch.from_numpy(vein_mask_np).float().unsqueeze(0) # Add channel dimension (1, H, W)

        # --- Future Improvement: Data Augmentation ---
        # If you decide to add rotational augmentation (or other types),
        # this is where you would apply your 'transform' logic.
        # Ensure augmentations are applied identically to 'inputs_tensor' and 'target_tensor'.
        # Example: if self.transform: inputs_tensor, target_tensor = self.transform(inputs_tensor, target_tensor)
        if self.transform:
             # Placeholder for where image transformations/augmentations would go.
             # You'd typically need to apply the same random transform to both inputs and target mask.
             pass 

        return inputs_tensor, target_tensor

# --- 2. Model Architecture: U-Net ---
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
        x = torch.cat([x2, x1], dim=1) # Concatenate along channel dimension
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
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# --- 3. Loss Function: Binary Cross-Entropy + Dice Loss ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: prediction tensor, targets: ground truth tensor
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice # Return (1 - Dice) as it's a loss

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss() # Uses logits directly, applies sigmoid internally
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        # For Dice loss, inputs should be probabilities, not logits.
        # BCEWithLogitsLoss combines sigmoid and BCE, so we manually apply sigmoid for Dice.
        dice_loss = self.dice(torch.sigmoid(inputs), targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

# --- New: Dice Coefficient Metric Function ---
def calculate_dice_coefficient(predictions, targets, smooth=1e-6):
    """
    Calculates the Dice Coefficient for a batch of predictions.
    Assumes predictions are probabilities (after sigmoid).
    """
    # Threshold predictions to get binary masks
    predictions = (predictions > 0.5).float()
    
    # Flatten tensors
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    intersection = (predictions * targets).sum()
    dice = (2. * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
    return dice.item() # Return scalar value

# --- Training and Validation Functions ---
def train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS, model_save_path=MODEL_SAVE_PATH, dice_history_save_path=DICE_HISTORY_SAVE_PATH):
    best_dice = -1.0 # Initialize with a value lower than any possible Dice (Dice is between 0 and 1)
    val_dice_history = [] # List to store validation Dice for each epoch

    # Ensure the model save directory exists
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Model will be saved to: {model_save_path}")
    print(f"Validation Dice history will be saved to: {dice_history_save_path}")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        # Wrap the DataLoader with tqdm for a progress bar
        train_loader_tqdm = tqdm(dataloaders['train'], desc=f"Training Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (inputs, targets) in enumerate(train_loader_tqdm):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            # Update tqdm postfix with current batch loss
            train_loader_tqdm.set_postfix(batch_loss=f"{loss.item():.4f}")

        epoch_train_loss = running_loss / len(dataloaders['train'].dataset)
        print(f"Train Loss: {epoch_train_loss:.4f}")

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        running_val_dice = 0.0
        # Wrap the DataLoader with tqdm for a progress bar
        val_loader_tqdm = tqdm(dataloaders['val'], desc=f"Validation Epoch {epoch+1}/{num_epochs}")
        with torch.no_grad():
            for inputs, targets in val_loader_tqdm:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, targets) # Use the combined loss for validation monitoring
                
                # Calculate Dice for evaluation (after sigmoid and thresholding)
                # Apply sigmoid to outputs to get probabilities for Dice calculation
                probs = torch.sigmoid(outputs) 
                dice = calculate_dice_coefficient(probs, targets)

                running_val_loss += loss.item() * inputs.size(0)
                running_val_dice += dice * inputs.size(0) # Accumulate weighted by batch size
                # Update tqdm postfix with current batch dice
                val_loader_tqdm.set_postfix(batch_dice=f"{dice:.4f}")


        epoch_val_loss = running_val_loss / len(dataloaders['val'].dataset)
        epoch_val_dice = running_val_dice / len(dataloaders['val'].dataset)
        print(f"Validation Loss: {epoch_val_loss:.4f}, Validation Dice: {epoch_val_dice:.4f}")
        
        # Store validation Dice for this epoch
        val_dice_history.append(epoch_val_dice)

        # Save the best model based on Dice Coefficient
        if epoch_val_dice > best_dice:
            best_dice = epoch_val_dice
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path} with Validation Dice: {best_dice:.4f}")
            
        print() # Newline for readability

    print("Training complete!")
    
    # Save the validation Dice history to a CSV
    df_dice_history = pd.DataFrame({
        'Epoch': range(1, len(val_dice_history) + 1),
        'Validation_Dice': val_dice_history
    })
    df_dice_history.to_csv(dice_history_save_path, index=False)
    print(f"Validation Dice history saved to {dice_history_save_path}")

    return model

# --- Main Execution ---
if __name__ == "__main__":
    # Set random seeds for reproducibility
    set_seed(GLOBAL_RANDOM_SEED)

    # Create dataset
    try:
        dataset = SyntheticLeafDataset(SYNTHETIC_METADATA_FILE, SYNTHETIC_DATA_OUTPUT_DIR)
    except ValueError as e:
        print(f"Error creating dataset: {e}")
        print("Please ensure the synthetic data generation script has been run successfully.")
        sys.exit(1)

    # Split dataset into training and validation
    # Use torch.manual_seed for reproducibility of random_split
    train_size = int((1 - VALIDATION_SPLIT_RATIO) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(GLOBAL_RANDOM_SEED))

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                            # Use custom worker_init_fn for per-worker reproducibility if num_workers > 0
                            worker_init_fn=lambda worker_id: np.random.seed(GLOBAL_RANDOM_SEED + worker_id) if NUM_WORKERS > 0 else None),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                          worker_init_fn=lambda worker_id: np.random.seed(GLOBAL_RANDOM_SEED + worker_id) if NUM_WORKERS > 0 else None)
    }

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Initialize model
    # n_channels_in = 2 (Blade ECT + Blade Mask)
    # n_classes_out = 1 (Vein Mask)
    model = UNet(n_channels_in=2, n_classes_out=1).to(DEVICE)

    # Loss function and optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    # Pass the dice_history_save_path to the train_model function
    trained_model = train_model(model, dataloaders, criterion, optimizer, NUM_EPOCHS, MODEL_SAVE_PATH, DICE_HISTORY_SAVE_PATH)

    print("\n--- Model Training Finished ---")

    # --- Example Inference (using the best saved model) ---
    print("\n--- Example Inference ---")
    if MODEL_SAVE_PATH.exists():
        # Load the best model's state
        inference_model = UNet(n_channels_in=2, n_classes_out=1).to(DEVICE)
        inference_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        inference_model.eval() # Set to evaluation mode

        # Take one sample from the validation set for demonstration
        if len(val_dataset) > 0:
            sample_inputs, sample_target_mask = val_dataset[0]
            
            # Add batch dimension and move to device
            sample_inputs = sample_inputs.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                predicted_logits = inference_model(sample_inputs)
                predicted_mask = torch.sigmoid(predicted_logits) # Convert logits to probabilities
                predicted_mask = (predicted_mask > 0.5).float() # Threshold to binary (0 or 1)

            # Move tensors back to CPU for plotting
            sample_inputs_cpu = sample_inputs.squeeze(0).cpu() # Remove batch dim
            blade_ect_viz = sample_inputs_cpu[0].numpy()
            blade_mask_viz = sample_inputs_cpu[1].numpy()
            sample_target_mask_viz = sample_target_mask.squeeze(0).cpu().numpy()
            predicted_mask_viz = predicted_mask.squeeze(0).cpu().numpy()

            # Plotting
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 4, 1)
            plt.imshow(blade_ect_viz, cmap='gray')
            plt.title("Input: Blade ECT")
            plt.axis('off')

            plt.subplot(1, 4, 2)
            plt.imshow(blade_mask_viz, cmap='gray')
            plt.title("Input: Blade Mask")
            plt.axis('off')

            plt.subplot(1, 4, 3)
            plt.imshow(sample_target_mask_viz, cmap='gray')
            plt.title("Ground Truth: Vein Mask")
            plt.axis('off')

            plt.subplot(1, 4, 4)
            plt.imshow(predicted_mask_viz, cmap='gray')
            plt.title("Predicted: Vein Mask")
            plt.axis('off')

            plt.tight_layout()
            plt.show()
            print("Displayed an example inference. Look for the plot window.")
        else:
            print("Validation dataset is empty, cannot perform example inference.")
    else:
        print("No trained model found to perform inference. Ensure training completed successfully.")
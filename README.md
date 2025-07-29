# Predicting veins from the blade of synthetic leaves
code for predicting leaf venation from the blade outline

![alt](https://github.com/DanChitwood/synthetic_vein_predict_CNN/blob/main/synthetic_vein_predict_CNN/outputs/figure/synthetic_vein_predictions_figure.png)  
**Prediciton of segmented vein masks from corresponding radial ECTs and shape masks of the blade using a CNN.** For 40 randomly created synthetic grapevine leaves, their radial ECT of the blade with superimposed blade outline (left) that was used as the input to the CNN, and the predicted pixels of the veins with blade outline for reference (right), with true positives (white), false positives (magenta), false negatives (dodgerblue), and background (gray) indicated by color.

## Data inputs  
Unzip the compressed file `data.zip` and keep in place to access the raw data.

## Geometric Morphometric Analysis and Synthetic Data Generation  
`0_morphometric_leaf_processing.ipynb`  
Leaf shape data, comprising both vein and blade coordinate landmarks, were prepared for geometric morphometric analysis. Each leaf's coordinate set was initially reshaped into a two-dimensional array by concatenating vein and blade coordinates. A Generalized Procrustes Analysis (GPA) was then performed to remove non-shape variation (position, scale, and orientation), yielding a mean shape and a set of Procrustes-aligned shapes.

Principal Component Analysis (PCA) was subsequently applied to the flattened, Procrustes-aligned shape coordinates to reduce dimensionality and capture the primary modes of shape variation. The number of principal components retained for analysis was dynamically determined by the minimum dimension of the input data, effectively encompassing the maximum possible variance (i.e., min(number of samples, number of flattened features)).

All critical data, including the calculated GPA mean shape, the fitted PCA model parameters (components, mean, explained variance, and number of components), the original PCA scores, and their corresponding genotype labels, were persistently saved. These outputs were stored in a structured directory ../outputs/saved_leaf_model_data, utilizing HDF5 format for numerical arrays (PCA parameters, scores) and Python's pickle for the leaf_indices dictionary, which defines the anatomical segments of the leaf. This robust saving strategy ensures data integrity and facilitates the subsequent generation of synthetic leaf shapes for downstream applications, such as convolutional neural network (CNN) training.

## Synthetic Leaf Data Generation  
`1_generate_synthetic_leaves.py`  
This Python script generates synthetic leaf morphological data, including blade and vein structures, in a manner suitable for training machine learning models. The process leverages a pre-trained Principal Component Analysis (PCA) model of real leaf shapes and a Synthetic Minority Over-sampling Technique (SMOTE)-like approach for data augmentation, followed by morphological feature extraction using the Euler Characteristic Transform (ECT).

1. Data Loading and PCA Model Initialization
The script initiates by loading parameters from a pre-trained PCA model. This includes the PCA components, mean, and explained variance, as well as the original PCA scores and corresponding genotype labels. These data are critical for reconstructing leaf coordinate data from the PCA latent space.

2. Synthetic Sample Generation via SMOTE-like Augmentation
Synthetic leaf shapes are generated within the PCA latent space to augment the dataset, particularly for under-represented genotype classes. A SMOTE-like algorithm is employed, where for each original leaf's PCA score, k nearest neighbors within the same genotype class are identified. New synthetic PCA scores are then interpolated between an original sample and one of its randomly selected neighbors. This process is repeated to achieve a target number of synthetic samples per genotype class, ensuring a balanced dataset.

3. Coordinate Reconstruction and Morphological Processing
Each synthetic PCA score is inverse-transformed back into its original flattened two-dimensional coordinate representation, which comprises distinct sets of coordinates for the blade outline and the vein structure. To introduce variability and improve model robustness, a random rotation (between -180 and 180 degrees is applied to the raw coordinates of each synthetic leaf.

Subsequently, the Euler Characteristic Transform (ECT) is applied to both the blade and vein coordinates. The ECT, parameterized with a BOUND_RADIUS of 1 unit, 360 radial ECT_THRESHOLDS (equally spaced from 0 to BOUND_RADIUS), and 360 NUM_ECT_DIRECTIONS, quantifies shape information by calculating the Euler characteristic at each direction and threshold. For superimposition with the original shape mask from which the ECT is derived, the ECT is converted into polar coordinates. To standardize the ECT output and prepare for visualization and downstream machine learning tasks, each transformed shape (blade and vein) undergoes centering, scaling, and re-orientation within the ECT's normalized space. An affine transformation matrix, derived from the ECT's internal coordinate normalization, is then applied to the original (randomly rotated) blade and vein coordinates. This aligns the generated 2D shapes to their corresponding ECT representations.

4. Output Generation
For each synthetic leaf, the script generates a suite of output files:

Blade and Vein Masks: Grayscale PNG images of the blade and vein structures, rendered in the ECT-aligned pixel space. The vein mask is specifically aligned to the blade's ECT space for potential use in a multi-channel CNN input.

Blade and Vein ECTs: Grayscale PNG images representing the radial ECT plots for both the blade and the vein.

Combined Visualizations: Colorized visualizations that overlay the blade outline on its respective ECT plot and the vein structure on its own ECT plot, facilitating visual inspection and quality control.

Transformed Coordinates: NumPy binary files (.npy) containing the precise floating-point coordinates of the blade outline and vein structure after ECT-specific transformations. These are saved to enable accurate reconstruction and detailed visualization in subsequent analyses without relying on rasterized mask data.

Metadata: A comprehensive CSV file (synthetic_metadata.csv) is generated, detailing each synthetic sample's ID, assigned genotype class, processing status (valid/skipped), reasons for skipping (if any), and file paths for all generated outputs.

All generated data are organized into a structured directory hierarchy, with an option to clear previous runs for clean regeneration. A fixed GLOBAL_RANDOM_SEED of 42 is employed for reproducibility, ensuring consistent synthetic data generation across multiple executions.  

## Convolutional Neural Network Architecture and Training  
`2_train_unet.py`  
A U-Net convolutional neural network (CNN) architecture was employed for the semantic segmentation of grapevine vein structures. The model was configured to accept two input channels: one for the blade's Euler Characteristic Transform (ECT) and another for the blade's binary mask. It outputs a single channel representing the predicted binary mask of the vein structure.

The network was trained using a combined loss function, consisting of Binary Cross-Entropy (BCE) Loss and Dice Loss, each weighted equally at 0.5. Optimization was performed using the Adam optimizer with a learning rate of 0.0005. Training was conducted over 100 epochs, utilizing a batch size of 16. The dataset was split into training and validation sets with an 80/20 ratio, respectively. Model performance was monitored on the validation set, and the best-performing model, as determined by the highest Dice Coefficient, was saved. All training processes were made reproducible by setting a global random seed to 42. The training leveraged available hardware acceleration, automatically selecting between MPS (Metal Performance Shaders), CUDA, or CPU.  

## Prediction and visualizaiton of random grapevine leaves  
`3_figure.py`  
This script generates a comprehensive visualization figure to evaluate the synthetic leaf vein segmentation process. It loads a selection of synthetic leaf images (comprising radial ECT and blade mask inputs), their corresponding ground truth vein masks, and the algorithmically generated blade outlines. A pre-trained U-Net model performs inference on these samples to predict vein patterns. The script then creates a multi-panel figure that displays the input radial ECT (with overlaid blade outline for context), the ground truth vein mask, and a composite prediction map that color-codes True Positives, False Positives, and False Negatives for detailed error analysis. The final high-resolution figure is saved as a PNG image, offering a visual summary of the synthetic dataset and the segmentation model's performance.

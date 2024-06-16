## Brief Description of the Problem and Data 

### Problem Description

In this project, we aim to generate images in the style of Claude Monet's paintings using Generative Adversarial Networks (GANs). Monet's artwork is known for its unique style, characterized by vibrant colors and distinct brush strokes. Our task is to transform photos into Monet-style paintings. This challenge is not only interesting from an artistic perspective but also demonstrates the power of GANs in image style transfer. The generated images will be evaluated using the MiFID (Memorization-informed Fréchet Inception Distance) score to assess the quality of the transformation.

### Dataset

We utilize the dataset provided by Kaggle for the GAN Getting Started competition. The dataset consists of two main components:

1. **Monet Paintings**:
   - Directory: `monet_jpg` and `monet_tfrec`
   - Description: Contains images of Monet paintings in JPEG and TFRecord formats.
   - Count: 300 Monet paintings sized 256x256 pixels

2. **Photos**:
   - Directory: `photo_jpg` and `photo_tfrec`
   - Description: Contains photos in JPEG and TFRecord formats.
   - Count: 7028 photos sized 256x256 pixels

The dataset structure ensures that we have a balanced and comprehensive set of images for training and validating our GAN model.

### Code to Load and Explore the Data

Here is the code to load and explore the dataset:

```python
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the path to the dataset directories
monet_jpg_dir = 'monet_jpg'
photo_jpg_dir = 'photo_jpg'

# List files in the Monet directory
monet_images = os.listdir(monet_jpg_dir)
print(f"Number of Monet images: {len(monet_images)}")

# List files in the Photo directory
photo_images = os.listdir(photo_jpg_dir)
print(f"Number of Photo images: {len(photo_images)}")

# Function to load and display images
def display_images(image_paths, title):
    plt.figure(figsize=(10, 10))
    for i, img_path in enumerate(image_paths[:9]):
        img = plt.imread(os.path.join(monet_jpg_dir, img_path))
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# Display sample Monet images
display_images(monet_images, 'Sample Monet Images')

# Display sample Photo images
display_images(photo_images, 'Sample Photo Images')
```

This code snippet demonstrates how to load the dataset, count the number of images in each category, and display a few sample images from the Monet paintings and photos. The function `display_images` helps in visualizing the images to understand the dataset better.

## Exploratory Data Analysis (EDA) — Inspect, Visualize and Clean the Data (15 pts)

### Load the Data

We begin by loading the necessary libraries and the dataset. The images are loaded from the respective directories provided in the last section.

### Data Cleaning

Next, we inspect and clean the data to ensure it is suitable for training our GAN model. We check the dimensions and data types of the images to ensure consistency.

```python
from PIL import Image

# Function to check the properties of images
def check_image_properties(image_dir, image_paths):
    for img_path in image_paths[:5]:
        img = Image.open(os.path.join(image_dir, img_path))
        print(f"Image: {img_path}, Size: {img.size}, Mode: {img.mode}")

# Check properties of Monet images
print("Monet Images:")
check_image_properties(monet_jpg_dir, monet_images)

# Check properties of Photo images
print("Photo Images:")
check_image_properties(photo_jpg_dir, photo_images)
```

Results:

```
Monet Images:
Image: 0a5075d42a.jpg, Size: (256, 256), Mode: RGB
Image: 0bd913dbc7.jpg, Size: (256, 256), Mode: RGB
Image: 0c1e3bff7f.jpg, Size: (256, 256), Mode: RGB
Image: 0e3b3292da.jpg, Size: (256, 256), Mode: RGB
Image: 1a127acf4d.jpg, Size: (256, 256), Mode: RGB

Photo Images:
Image: 0a0c3a6d07.jpg, Size: (256, 256), Mode: RGB
Image: 0a0d36ea7.jpg, Size: (256, 256), Mode: RGB
Image: 0a1dbf98e.jpg, Size: (256, 256), Mode: RGB
Image: 0a2e5f728.jpg, Size: (256, 256), Mode: RGB
Image: 0a6e92d928.jpg, Size: (256, 256), Mode: RGB
```
The analysis of the results from the image properties check for both Monet images and Photo images reveals the following insights:

1. **Image Size Consistency**:
   - All Monet images and Photo images have a consistent size of \(256 \times 256\) pixels. This uniformity in image dimensions is crucial for the training of GANs as it ensures that the input data to the model is standardized, which can help improve the efficiency and effectiveness of the training process.

2. **Image Mode**:
   - Both Monet and Photo images are in RGB mode, which means they contain three color channels (Red, Green, and Blue). This is important for the style transfer task as the model needs to learn the color distributions and patterns specific to Monet’s style and apply them to the photos.

3. **Implications for Model Training**:
   - The consistency in image size and mode indicates that the dataset is well-prepared for immediate use in training the GAN model without requiring additional preprocessing steps to resize or convert image modes. This can streamline the workflow and focus the efforts on model architecture and training strategies.

In summary, the uniformity in the size and mode of the images ensures that the dataset is ready for GAN training, which is critical for the success of the style transfer task.

### Data Distribution Analysis

We analyze the distribution of the images to understand the balance of the dataset. This includes visualizing the distribution of image sizes and the number of images in each category.

```python
import pandas as pd

# Function to analyze the distribution of image sizes
def analyze_image_sizes(image_dir, image_paths):
    sizes = [Image.open(os.path.join(image_dir, img_path)).size for img_path in image_paths]
    sizes_df = pd.DataFrame(sizes, columns=['Width', 'Height'])
    return sizes_df

# Analyze Monet image sizes
monet_sizes_df = analyze_image_sizes(monet_jpg_dir, monet_images)
print(monet_sizes_df.describe())

# Analyze Photo image sizes
photo_sizes_df = analyze_image_sizes(photo_jpg_dir, photo_images)
print(photo_sizes_df.describe())

# Plot distribution of image sizes
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(monet_sizes_df['Width'], bins=20, alpha=0.7, label='Width')
plt.hist(monet_sizes_df['Height'], bins=20, alpha=0.7, label='Height')
plt.title('Monet Image Sizes')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(photo_sizes_df['Width'], bins=20, alpha=0.7, label='Width')
plt.hist(photo_sizes_df['Height'], bins=20, alpha=0.7, label='Height')
plt.title('Photo Image Sizes')
plt.legend()

plt.show()
```

Results:

Monet Image Sizes:
```
            Width  Height
count  300.000000   300.0
mean   256.000000   256.0
std      0.000000     0.0
min    256.000000   256.0
25%    256.000000   256.0
50%    256.000000   256.0
75%    256.000000   256.0
max    256.000000   256.0
```

Photo Image Sizes:
```
            Width  Height
count  7028.000000   7028.0
mean    256.000000   256.0
std       0.000000     0.0
min      256.000000   256.0
25%      256.000000   256.0
50%      256.000000   256.0
75%      256.000000   256.0
max      256.000000   256.0
```

![Monet Image Sizes](./images/monet_image.png)

The analysis of the image size distributions for both Monet and Photo images reveals a high level of consistency. Every image in both datasets measures \(256 \times 256\) pixels, as shown by the descriptive statistics and the histograms. The lack of variation in the sizes, evidenced by the zero standard deviation, ensures that the entire dataset is uniform in dimensions.

For the Monet images, we have 300 samples, each with a mean size of 256 pixels in both width and height. This uniformity extends across all percentiles (25th, 50th, and 75th), indicating no deviation from this size. Similarly, the Photo dataset comprises 7028 images, all with identical dimensions. The mean, median, and other statistical measures confirm this consistency, as every image is exactly 256 pixels wide and 256 pixels tall.

This uniformity in image size is highly beneficial for training our Generative Adversarial Network (GAN). Since all images are already resized to the desired dimensions, there is no need for additional preprocessing to standardize the image sizes. This not only saves computational resources but also streamlines the data preparation process, allowing us to focus more on model development and training. The consistent size across the dataset ensures that the model receives uniform input, which can enhance the training efficiency and overall performance of the GAN in style transfer tasks.


### Sample Images

We display a few sample images from both Monet paintings and photos to get an idea of the data. This helps us understand the visual characteristics of the images we will be working with.

```python
# Function to display sample images with titles
def display_sample_images(image_dir, image_paths, title):
    plt.figure(figsize=(10, 10))
    for i, img_path in enumerate(image_paths[:9]):
        img = plt.imread(os.path.join(image_dir, img_path))
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(f"{title} {i+1}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# Display sample Monet images
display_sample_images(monet_jpg_dir, monet_images, 'Monet Sample Image')

# Display sample Photo images
display_sample_images(photo_jpg_dir, photo_images, 'Photo Sample Image')
```

Results:

Monet Sample Images:

![Monet Sample Images](./images/sample1.png)

Photo Sample Images:

![Photo Sample Images](./images/sample2.png)

This code snippet includes functions to inspect, visualize, and clean the data. We load and display sample images, check their properties, analyze their size distributions, and visualize the data to ensure it is ready for model training.
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

## Model Architecture 

In this section, we will describe and implement the architecture of our Generative Adversarial Network (GAN) for style transfer, focusing on generating images in the style of Monet's paintings. We will utilize a CycleGAN architecture, which is particularly well-suited for style transfer tasks as it allows for image-to-image translation without the need for paired examples. This architecture consists of two main components: Generators and Discriminators.

1. **Generators**: The generators are responsible for creating images in the target style (Monet) from the source images (photos). We use two generators:
   - **G\_A**: Transforms photos into Monet-style paintings.
   - **G\_B**: Transforms Monet-style paintings back into photos.

2. **Discriminators**: The discriminators evaluate the authenticity of the generated images. We use two discriminators:
   - **D\_A**: Distinguishes between real Monet paintings and those generated by G\_A.
   - **D\_B**: Distinguishes between real photos and those generated by G\_B.

### CycleGAN Implementation

We will implement the CycleGAN architecture using TensorFlow and Keras. The model includes building and compiling the generators and discriminators, setting up the loss functions, and training the model.

### Generator Model (U-Net)

The generator model is designed as a U-Net, which consists of an encoder (downsampling) and a decoder (upsampling) with skip connections to preserve spatial information.

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Activation, Concatenate, Input
from tensorflow.keras.models import Model

# Define the generator model (U-Net)
def build_generator():
    def conv2d(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = Conv2DTranspose(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        u = Activation('relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    img_input = Input(shape=(256, 256, 3))

    # Downsampling
    d1 = conv2d(img_input, 64, bn=False)
    d2 = conv2d(d1, 128)
    d3 = conv2d(d2, 256)
    d4 = conv2d(d3, 512)
    d5 = conv2d(d4, 512)
    d6 = conv2d(d5, 512)
    d7 = conv2d(d6, 512)
    d8 = conv2d(d7, 512, bn=False)

    # Upsampling
    u1 = deconv2d(d8, d7, 512)
    u2 = deconv2d(u1, d6, 512)
    u3 = deconv2d(u2, d5, 512)
    u4 = deconv2d(u3, d4, 512)
    u5 = deconv2d(u4, d3, 256)
    u6 = deconv2d(u5, d2, 128)
    u7 = deconv2d(u6, d1, 64)

    u8 = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same')(u7)
    output_img = Activation('tanh')(u8)

    return Model(img_input, output_img)
```

### Discriminator Model (PatchGAN)

The discriminator model is designed as a PatchGAN, which classifies image patches instead of the whole image, providing better stability and performance in GAN training.

```python
# Define the discriminator model (PatchGAN)
def build_discriminator():
    def d_layer(layer_input, filters, f_size=4, bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    img_input = Input(shape=(256, 256, 3))
    d1 = d_layer(img_input, 64, bn=False)
    d2 = d_layer(d1, 128)
    d3 = d_layer(d2, 256)
    d4 = d_layer(d3, 512)
    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model(img_input, validity)
```

### Building and Compiling the Models

We build and compile the generator and discriminator models. The generators use the Adam optimizer with a learning rate of 0.0002 and a beta_1 of 0.5.

```python
from tensorflow.keras.optimizers import Adam

# Build the generators
G_A = build_generator()
G_B = build_generator()

# Build the discriminators
D_A = build_discriminator()
D_B = build_discriminator()

# Compile the discriminators
D_A.compile(loss='mse', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
D_B.compile(loss='mse', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Define the combined model for CycleGAN
def build_combined(G, D, image_shape):
    D.trainable = False
    img = Input(shape=image_shape)
    fake_img = G(img)
    validity = D(fake_img)
    return Model(img, validity)

# Combined model
combined_A = build_combined(G_A, D_A, (256, 256, 3))
combined_B = build_combined(G_B, D_B, (256, 256, 3))

# Compile the combined model
combined_A.compile(loss='mse', optimizer=Adam(0.0002, 0.5))
combined_B.compile(loss='mse', optimizer=Adam(0.0002, 0.5))
```

### Hyperparameter Tuning

To optimize the performance of our model within the computational constraints, we performed a simplified hyperparameter tuning. We tested a smaller number of hyperparameter combinations and reduced the number of epochs and batch sizes for faster training.

```python
from sklearn.model_selection import ParameterSampler
import random
import numpy as np
from tensorflow.keras.optimizers import Adam

# Define the parameter grid with fewer options for faster computation
param_grid = {
    'learning_rate': [0.0001, 0.0002],
    'batch_size': [1, 4],
    'epochs': [1000, 2000]  # Reduced number of epochs
}

# Number of parameter settings to sample
n_iter_search = 2
random.seed(42)
param_list = list(ParameterSampler(param_grid, n_iter=n_iter_search, random_state=42))

# Function to load and preprocess a smaller subset of the data
def load_data(image_dir, image_paths, sample_size=100):
    data = []
    for img_path in image_paths[:sample_size]:  # Use only a subset of the data
        img = Image.open(os.path.join(image_dir, img_path))
        img = img.resize((256, 256))
        img = np.array(img) / 127.5 - 1.0
        data.append(img)
    return np.array(data)

# Load a smaller subset of the datasets for faster computation
monet_data = load_data(monet_jpg_dir, monet_images, sample_size=100)
photo_data = load_data(photo_jpg_dir, photo_images, sample_size=100)

# Hyperparameter tuning loop
best_params = None
best_loss = float('inf')

for params in param_list:
    print(f"Testing parameters: {params}")
    
    # Build and compile models with the current parameters
    G_A = build_generator()
    G_B = build_generator()
    D_A = build_discriminator()
    D_B = build_discriminator()
    
    D_A.compile(loss='mse', optimizer=Adam(params['learning_rate'], 0.5), metrics=['accuracy'])
    D_B.compile(loss='mse', optimizer=Adam(params['learning_rate'], 0.5), metrics=['accuracy'])
    
    combined_A = build_combined(G_A, D_A, (256, 256, 3))
    combined_B = build_combined(G_B, D_B, (256, 256, 3))
    
    combined_A.compile(loss='mse', optimizer=Adam(params['learning_rate'], 0.5))
    combined_B.compile(loss='mse', optimizer=Adam(params['learning_rate'], 0.5))
    
    # Training parameters
    epochs = params['epochs']
    batch_size = params['batch_size']
    
    # Calculate the patch size for the discriminator output
    patch = int(256 / 2**4)  # Assuming 4 down-sampling layers
    disc_patch = (patch, patch, 1)
    
    # Training loop
    for epoch in range(epochs):
        for batch in range(len(photo_data) // batch_size):
            # Select a batch of photos
            idx_photos = np.random.randint(0, len(photo_data), batch_size)
            idx_monet = np.random.randint(0, len(monet_data), batch_size)
            real_photos = photo_data[idx_photos]
            real_monet = monet_data[idx_monet]
            
            # Generate fake Monet images
            fake_monet = G_A.predict(real_photos)
            
            # Reshape the fake and real labels to match the output shape of the discriminator
            valid = np.ones((batch_size,) + disc_patch)
            fake = np.zeros((batch_size,) + disc_patch)
            
            # Train the discriminators
            D_A_loss_real = D_A.train_on_batch(real_monet, valid)
            D_A_loss_fake = D_A.train_on_batch(fake_monet, fake)
            D_A_loss = 0.5 * np.add(D_A_loss_real, D_A_loss_fake)
            
            # Train the generators
            G_A_loss = combined_A.train_on_batch(real_photos, valid)
            
            # Print the progress
            if batch % 10 == 0:  # Print progress less frequently
                print(f"[Epoch {epoch}/{epochs}] [Batch {batch}/{len(photo_data) // batch_size}] "
                      f"[D_A loss: {D_A_loss[0]}, acc: {100 * D_A_loss[1]}] [G_A loss: {G_A_loss}]")
    
    # Evaluate the model performance on a validation set or some metric
    total_loss = D_A_loss[0] + G_A_loss
    if total_loss < best_loss:
        best_loss = total_loss
        best_params = params
        G_A.save('best_G_A_model.h5')
        G_B.save('best_G_B_model.h5')
        D_A.save('best_D_A_model.h5')
        D_B.save('best_D_B_model.h5')

print(f"Best parameters found: {best_params} with loss: {best_loss}")
```
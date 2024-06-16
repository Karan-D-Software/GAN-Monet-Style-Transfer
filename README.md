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


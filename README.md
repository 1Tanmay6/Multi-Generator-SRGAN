# Multi-Generator SRGAN

## Overview

Welcome to the Multi-Generator Super-Resolution Generative Adversarial Network (SRGAN) project! This repository showcases an implementation that I have developed, drawing inspiration from the original SRGAN paper and enhancing it with a multi-generator approach to advance image super-resolution. The model is designed to upscale low-resolution images into high-resolution outputs using a powerful ensemble of generators. Additionally, I've included a scaled-down version of the SRGAN model, optimized to generate images from 64x64 resolution to 128x128 resolution, perfect for environments with limited computational resources.

## SRGAN Architecture

Super-Resolution Generative Adversarial Networks (SRGAN) are designed to transform low-resolution (LR) images into high-resolution (HR) counterparts. The architecture consists of four main components:

1. **Dataset**: The dataset includes both high-resolution (HR) and low-resolution (LR) images, forming the foundation of the model's training process.
2. **The Generator**: A deep convolutional neural network that takes in a low-resolution image and outputs a high-resolution version. The generator is trained using content loss (measured using a pre-trained VGG network, such as VGG19) and adversarial loss, which drives the network to create images indistinguishable from real HR images.
3. **The Discriminator**: A deep neural network that distinguishes between real images from the dataset and fake images generated by the generator. It’s trained to maximize its classification accuracy.
4. **VGGx Model**: A Convolutional Neural Network (CNN) used for feature extraction, where 'x' denotes the specific version (e.g., VGG19, VGG54) used to compute content loss.

### Loss Functions

- **Content Loss**: Evaluates the difference in high-level features between the generated and real images using a pre-trained VGG network, with Mean Squared Error (MSE) as the metric.
- **Adversarial Loss**: Encourages the generator to produce images that the discriminator cannot differentiate from real ones, typically measured using Binary Cross-Entropy (BCE).

## Multi-Generator Ensemble

To elevate the performance of traditional SRGANs, I integrated multiple generators into the model. Each generator is trained independently to create high-resolution images. During inference, the outputs from these generators are combined—using methods such as averaging—to produce the final image. This approach is particularly useful when working with less powerful hardware, as it leverages multiple, less complex models to achieve high-quality results.

### Why Multiple Generators?

- **Enhanced Detail**: The ensemble approach captures a broader range of details and reduces artifacts.
- **Improved Robustness**: By combining the outputs of multiple generators, the likelihood of individual generator errors impacting the final result is reduced.

## Generator Updates per Discriminator Update

In traditional GAN training, the generator and discriminator are updated alternately. My implementation allows for a configurable ratio of generator to discriminator updates (`n:m`). For example, a `2:1` ratio means the generator is updated twice for each discriminator update, helping to stabilize training and enabling the generator to keep pace with a powerful discriminator.

## Scaled-Down SRGAN

For those working in environments with limited resources, I’ve included a scaled-down version of the SRGAN. This lightweight model is designed to generate images with resolutions from 64x64 to 128x128, making it suitable for quicker training and lower memory consumption.

### Key Features:

- **Optimized Memory Usage**: Ideal for setups with limited GPU memory.
- **Faster Training**: The reduced complexity leads to shorter training times.
- **Maintained Quality**: Despite being scaled down, this model strives to retain the detail and perceptual quality of the generated images.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/1Tanmay6/Multi-Generator-SRGAN.git
cd Multi-Generator-SRGAN
pip install -r requirements.txt
```

## Usage

This project is packaged for ease of use, with modules designed for each key component:

- **Data**: Handles loading and preprocessing of HR and LR images.
- **Generator**: Contains the implementation of the generator models.
- **Discriminator**: Implements the discriminator network.
- **LossUtils**: Utility functions for computing content and adversarial losses.
- **Utils**: General utility functions for model training and evaluation.
- **Inferencers**: Modules for generating high-resolution images from LR inputs during inference.
- **Trainers**: Tools for managing the training loop, including configurable update ratios for generators and discriminators.

These modules are designed to be intuitive and flexible, allowing you to adapt the pipeline to your specific needs.

## Configuration

Adjust training parameters, including the ratio of generator to discriminator updates, in the `config.ini` file to customize your training process.

## Conclusion

This Multi-Generator SRGAN implementation offers a robust solution for super-resolution tasks, harnessing the power of multiple models to produce superior results. Whether you're working with limited hardware or require high-fidelity images, this repository provides the tools you need. I welcome contributions—feel free to open issues or submit pull requests!

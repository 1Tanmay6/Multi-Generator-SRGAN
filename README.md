# Multi-Generator SRGAN

## Overview

This repository contains the implementation of a Multi-Generator Super-Resolution Generative Adversarial Network (SRGAN) directly inspired by the original SRGAN paper. The implementation is designed to generate high-resolution images from low-resolution inputs using an ensemble of generators. Additionally, we provide a scaled-down version of the SRGAN model, which can generate images from 64x64 resolution to 128x128 resolution.

## SRGAN Architecture

Super-Resolution Generative Adversarial Networks (SRGAN) are designed to generate high-resolution (HR) images from low-resolution (LR) images. The model is composed of two main components: the Generator and the Discriminator.

### Generator

- The Generator is a deep convolutional neural network that takes a low-resolution image as input and outputs a high-resolution image. It is trained to minimize the content loss (typically using a pre-trained VGG network to extract features) and an adversarial loss, which encourages the generator to produce images indistinguishable from real high-resolution images.

### Discriminator

- The Discriminator is a deep neural network that classifies images as real (from the dataset) or fake (generated by the generator). It is trained to maximize its accuracy in this binary classification task.

### Losses

- **Content Loss:** Measures the difference in high-level features between the generated image and the real image using a pre-trained VGG network.
- **Adversarial Loss:** Encourages the generator to produce images that the discriminator cannot distinguish from real images.

## Multi-Generator Ensemble

In this implementation, we extend the traditional SRGAN by employing multiple generators. Each generator is independently trained to create high-resolution images. During inference, the outputs from multiple generators are combined, typically using averaging or a more sophisticated ensemble strategy, to produce the final high-resolution image.

This approach aims to capture a wider variety of details and reduce artifacts by leveraging the strengths of multiple models.

## Generator Updates per Discriminator Update

In traditional GAN training, the generator and discriminator are updated alternately. In this implementation, the generator is updated multiple times for every discriminator update. This ratio is configurable and is set to `n:m`, where `n` is the number of generator updates and `m` is the number of discriminator updates.

For example, a ratio of `2:1` means that the generator will be updated twice for every discriminator update. This strategy helps stabilize the training process and allows the generator to learn faster than the discriminator, especially in cases where the discriminator becomes too strong and easily classifies generated images as fake.

## Scaled-Down SRGAN

In addition to the full-scale SRGAN, this repository includes a scaled-down version designed to generate images with resolutions ranging from 64x64 to 128x128. This version is more lightweight and resource-efficient, making it suitable for environments with limited computational resources or when working with smaller image resolutions.

### Key Features of the Scaled-Down Version:

- **Lower Memory Consumption:** Optimized for environments with limited GPU memory.
- **Faster Training:** Reduced model size and complexity lead to faster training times.
- **Preserved Quality:** Despite the reduced scale, the model aims to maintain a high level of detail and perceptual quality in the generated images.

## Installation

```bash
git clone https://github.com/1Tanmay6/Multi-Generator-SRGAN.git
cd Multi-Generator-SRGAN
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the multi-generator SRGAN, use the following command:

```bash
python train.py --config config.yaml
```

### Inference

To generate high-resolution images using the ensemble of generators:

```bash
python inference.py --input_dir low_res_images/ --output_dir high_res_images/
```

### Scaled-Down Version

To train or perform inference with the scaled-down version:

```bash
python train.py --config config_scaled.yaml
```

## Configuration

You can adjust various training parameters, including the number of generator updates per discriminator update, in the `config.yaml` file.

## Conclusion

This implementation of SRGAN with a multi-generator ensemble offers enhanced performance by combining the strengths of multiple models. The scaled-down version provides a lightweight alternative for generating lower-resolution images, making it adaptable to various use cases and environments.

Contributions are welcome! Please feel free to open issues or submit pull requests.

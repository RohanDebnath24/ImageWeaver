# ImageWeaver: A PyTorch Implementation of Stable Diffusion

A PyTorch implementation of Stable Diffusion from scratch, inspired by various open-source implementations and designed to be flexible and easy to use.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Training](#training)
- [Inference](#inference)
- [Contributing](#contributing)
- [License](#license)

## Introduction

ImageWeaver is a PyTorch implementation of Stable Diffusion, a powerful generative model that combines diffusion-based image synthesis with a text encoder. This project aims to provide a clear, well-documented implementation suitable for both research and practical applications.


## Key Components

### Diffusion Process

ImageWeaver implements the diffusion process, gradually adding noise to an image until it becomes completely random. This process is reversible, allowing us to start with noise and progressively refine it into an image.

### Denoising Process

The denoising process removes noise from an image until it becomes recognizable. This is achieved through a series of forward diffusion steps and reverse diffusion steps.

### U-Net Architecture

ImageWeaver uses a U-Net as the primary neural network architecture. U-Nets are particularly well-suited for image-to-image translation tasks and image denoising.

### Attention Mechanism

We employ both regular attention and cross-attention mechanisms. Regular attention allows ImageWeaver to focus on different parts of the input image, while cross-attention enables the model to combine information from both the image and text prompt.

### Text Encoder

Our text encoder is responsible for encoding text prompts into embeddings that can be used by ImageWeaver. It's implemented using a Transformer architecture.

## Implementation Details

### Rotary Positional Encodings

ImageWeaver utilizes rotary positional encodings to handle long sequences effectively, improving the model's ability to capture position information.

### Normalization Layers

Layer normalization is used throughout ImageWeaver to stabilize training and improve performance.

### Sampling Process

Our implementation includes various sampling methods, including DDIM (Denoising Diffusion Implicit Model) and PLMS (Probabilistic Latent Variable Models).


## Training

Training ImageWeaver involves optimizing the loss function that measures the difference between the predicted noise and the actual noise added during the forward diffusion process. Our implementation supports training on custom datasets.

## Inference

Once trained, ImageWeaver can be used for various tasks:

1. Image Generation from Text Prompts
2. Image-to-Image Translation
3. Fine-tuning on Custom Datasets

Example scripts for these tasks are provided in the `examples` directory.

## Contributing

Contributions are welcome! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to ImageWeaver.

## License

ImageWeaver is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the following repositories for inspiration and reference:

- https://github.com/CompVis/stable-diffusion/
- https://github.com/divamgupta/stable-diffusion-tensorflow
- https://github.com/kjsman/stable-diffusion-pytorch
- https://github.com/huggingface/diffusers/

## Installation

To install required dependencies:
pip install torch torchvision numpy matplotlib


## Downloading Pre-trained Weights

Download the necessary files from Hugging Face:

1. `vocab.json` and `merges.txt` from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/tokenizer
2. `v1-5-pruned-emaonly.ckpt` from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main

Place these files in the `data` folder of your ImageWeaver project.

## Tested Fine-tuned Models

ImageWeaver has been tested with several fine-tuned models:

- InkPunk Diffusion: https://huggingface.co/Envvi/Inkpunk-Diffusion/tree/main
- Illustration Diffusion (Hollie Mengert): https://huggingface.co/ogkalu/Illustration-Diffusion/tree/main

Simply download the `.ckpt` file from any fine-tuned SD model (up to v1.5) and place it in the `data` folder.

Thank you for exploring ImageWeaver! Feel free to experiment and push the boundaries of what's possible with this powerful generative model.





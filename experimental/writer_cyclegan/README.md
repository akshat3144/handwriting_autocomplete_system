# Writer CycleGAN

A PyTorch implementation of CycleGAN and Pix2Pix models for handwriting style transfer and image-to-image translation. This project extends the original CycleGAN architecture with writer-specific conditioning and OCR-based loss functions for improved handwriting synthesis.

## Features

- **CycleGAN**: Unpaired image-to-image translation for handwriting style transfer
- **Pix2Pix**: Paired image-to-image translation
- **Writer Conditioning**: Style transfer conditioned on specific writer identities
- **OCR Loss Integration**: Maintains text legibility during style transfer
- **Multi-GPU Training**: Distributed data parallel (DDP) support
- **Wandb Integration**: Experiment tracking and visualization

## Models

### Available Models

- `cycle_gan`: Cycle-consistent adversarial networks for unpaired translation
- `pix2pix`: Conditional GANs for paired image-to-image translation
- `test`: Single-direction generation (inference only)
- `colorization`: Image colorization model

### Network Architectures

- **Generators**: ResNet (6/9 blocks), U-Net (128/256)
- **Discriminators**: PatchGAN (70x70), n-layers, pixel-wise
- **Writer Encoder**: Embedding network for writer identity conditioning

## Installation

### Prerequisites

- Python 3.11
- CUDA 12.1 (for GPU training)
- PyTorch 2.4.0

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate pytorch-img2img
```

### Manual Installation

```bash
pip install torch==2.4.0 torchvision==0.19.0
pip install dominate Pillow scikit-image wandb
```

## Dataset Preparation

Organize your dataset in the following structure:

```
datasets/
└── your_dataset/
    ├── trainA/          # Source domain training images
    ├── trainB/          # Target domain training images
    ├── testA/           # Source domain test images
    └── testB/           # Target domain test images
```

For the IAM handwriting dataset, use the provided preparation notebook:

```bash
jupyter notebook iam_dataset_preparation.ipynb
```

## Training

### Basic CycleGAN Training

```bash
python train.py --dataroot ./datasets/your_dataset \
                --name experiment_name \
                --model cycle_gan \
                --batch_size 4 \
                --n_epochs 100 \
                --n_epochs_decay 100
```

### Training with Writer Conditioning

```bash
python train.py --dataroot ./datasets/iam_handwriting \
                --name writer_conditioned \
                --model cycle_gan \
                --num_writers 50 \
                --embed_dim 128 \
                --lambda_OCR 1.0
```

### Pix2Pix Training

```bash
python train.py --dataroot ./datasets/paired_dataset \
                --name pix2pix_experiment \
                --model pix2pix \
                --direction AtoB
```

### Training Options

Key parameters:

- `--dataroot`: Path to dataset directory
- `--name`: Experiment name (saves to `./checkpoints/<name>/`)
- `--model`: Model type (cycle_gan, pix2pix, etc.)
- `--gpu_ids`: GPU IDs to use (e.g., `0,1,2,3`)
- `--batch_size`: Batch size per GPU
- `--n_epochs`: Number of epochs with initial learning rate
- `--n_epochs_decay`: Number of epochs to decay learning rate
- `--lambda_A`, `--lambda_B`: Cycle consistency loss weights
- `--lambda_identity`: Identity loss weight
- `--lambda_OCR`: OCR loss weight (for writer conditioning)

### Multi-GPU Training

```bash
python train.py --dataroot ./datasets/your_dataset \
                --name multi_gpu_experiment \
                --model cycle_gan \
                --gpu_ids 0,1,2,3 \
                --batch_size 16
```

## Testing/Inference

### Test a Trained Model

```bash
python test.py --dataroot ./datasets/your_dataset/testA \
               --name experiment_name \
               --model cycle_gan \
               --epoch latest
```

### Single-Direction Testing

```bash
python test.py --dataroot ./datasets/your_dataset/testA \
               --name experiment_name \
               --model test \
               --no_dropout
```

### Test Options

- `--results_dir`: Directory to save results (default: `./results/`)
- `--num_test`: Number of test images (default: 50)
- `--epoch`: Which checkpoint to load (latest, 100, 200, etc.)
- `--phase`: Dataset phase to use (test, val, train)

## Project Structure

```
writer_cyclegan/
├── train.py                    # Training script
├── test.py                     # Testing/inference script
├── environment.yml             # Conda environment configuration
├── models/                     # Model definitions
│   ├── cycle_gan_model.py     # CycleGAN implementation
│   ├── pix2pix_model.py       # Pix2Pix implementation
│   ├── networks.py            # Network architectures
│   ├── writer_encoder.py      # Writer embedding network
│   └── ocr_loss.py            # OCR-based loss functions
├── data/                       # Dataset loaders
│   ├── unaligned_dataset.py   # Unpaired data loader
│   ├── aligned_dataset.py     # Paired data loader
│   └── single_dataset.py      # Single-direction loader
├── options/                    # Command-line options
│   ├── base_options.py        # Common options
│   ├── train_options.py       # Training-specific options
│   └── test_options.py        # Testing-specific options
├── util/                       # Utility functions
│   ├── image_pool.py          # Image buffer for discriminator
│   ├── visualizer.py          # Training visualization
│   └── html.py                # HTML result generation
└── scripts/                    # Helper scripts
    ├── train_cyclegan.sh      # CycleGAN training script
    └── test_cyclegan.sh       # CycleGAN testing script
```

## Notebooks

- `CycleGAN.ipynb`: Interactive CycleGAN training and testing
- `pix2pix.ipynb`: Interactive Pix2Pix training and testing
- `iam_dataset_preparation.ipynb`: IAM dataset preprocessing

## Advanced Features

### Writer-Conditioned Style Transfer

The model supports writer-specific conditioning using learnable embeddings:

```python
# Writer embedding is automatically injected into generator
fake_B = netG_A(real_A, writer_embedding)
```

This allows the model to:

- Generate text in specific writer styles
- Interpolate between different writing styles
- Control style transfer with writer identity

### OCR Loss

The OCR loss ensures generated handwriting remains readable:

```bash
python train.py --dataroot ./datasets/iam \
                --name ocr_loss_experiment \
                --lambda_OCR 1.0 \
                --ocr_model_path ./pretrained_ocr.pth
```

### Weights & Biases Integration

Track experiments with Wandb:

```bash
python train.py --dataroot ./datasets/your_dataset \
                --name wandb_experiment \
                --use_wandb \
                --wandb_project_name my_cyclegan_project
```

## Results

Results are saved in the following locations:

- **Checkpoints**: `./checkpoints/<experiment_name>/`
- **Test Results**: `./results/<experiment_name>/`
- **Training Visualizations**: `./checkpoints/<experiment_name>/web/`

Each checkpoint directory contains:

- `latest_net_G.pth`: Latest generator weights
- `latest_net_D.pth`: Latest discriminator weights
- `loss_log.txt`: Training loss history
- `web/`: HTML visualization of training progress

## Tips and Best Practices

1. **Data Preprocessing**: Ensure images are properly aligned and normalized
2. **Learning Rate**: Start with `lr=0.0002` and decay after 100 epochs
3. **Identity Loss**: Use `--lambda_identity 0.5` for style transfer tasks
4. **Batch Size**: Increase batch size for stable training (4-16 depending on GPU)
5. **Training Time**: CycleGAN typically requires 100-200 epochs
6. **GPU Memory**: Use `--crop_size 256` or smaller if running out of memory

## Common Issues

### CUDA Out of Memory

- Reduce `--batch_size`
- Reduce `--crop_size`
- Use fewer layers: `--netG resnet_6blocks`

### Training Instability

- Enable identity loss: `--lambda_identity 0.5`
- Adjust learning rate: `--lr 0.0001`
- Use instance normalization: `--norm instance`

### Poor Quality Results

- Train longer (200+ epochs)
- Increase cycle consistency weight: `--lambda_A 20 --lambda_B 20`
- Use larger networks: `--ngf 128`

## Citation

If you use this code for your research, please cite:

```bibtex
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}

@inproceedings{pix2pix2017,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```

## License

This project builds upon the original CycleGAN and Pix2Pix implementations. See LICENSE file for details.

## Related Projects

- [Original CycleGAN and Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)

## Acknowledgments

- Original CycleGAN and Pix2Pix implementation by Jun-Yan Zhu et al.
- Writer conditioning and OCR loss extensions for handwriting synthesis
- IAM Handwriting Database for training data

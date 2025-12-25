# Audio-Visual Sound Source Localization with Slot Attention

Official implementation of **"Improving Sound Source Localization with Joint Slot Attention on Image and Audio"**.

[PAPER](https://arxiv.org/abs/2504.15118)

## Features

- **Slot Attention Mechanism**: Decomposes audio-visual features into discrete slots representing different sound sources
- **Multi-Modal Learning**: Combines visual (ResNet) and audio encoders for joint representation learning
- **Multiple Evaluation Metrics**: Supports cIoU, AUC, and mAP for comprehensive performance assessment
- **Flexible Training**: Supports multiple datasets and training configurations

## Architecture

- **Visual Encoder**: ResNet-based image feature extractor
- **Audio Encoder**: ResNet-based spectrogram feature extractor
- **Slot Attention Module**: Iterative attention mechanism for object-centric representations
- **Object Saliency Integration**: Combines learned features with object saliency maps

## Supported Datasets

- **VGG-Sound**: 10k and 144k training splits
- **Flickr SoundNet**: 10k and 144k training splits
- **AVSBench**: MS3 and S4 test sets for evaluation

## Project Structure

```
.
├── train_slot.py           # Main training script
├── test_model.py           # Testing and evaluation script
├── model_slot.py           # Slot attention model architecture
├── dataset.py              # VGG-Sound and Flickr dataset loaders
├── dataset_avs.py          # AVSBench dataset loaders
├── utils.py                # Utility functions and evaluators
├── resnet.py               # ResNet backbone implementation
├── audio_io.py             # Audio processing utilities
├── train_script.sh         # Training script with configurations
├── metadata/               # Dataset metadata files
└── checkpoints/            # Model checkpoints (created during training)
```

## Training

Use the provided shell script for training:

```bash
# Train on VGG-Sound 10k dataset, test on VGG-Sound
bash train_script.sh vggss_10k vggss 0

# Train on Flickr 10k dataset, test on Flickr
bash train_script.sh flickr_10k flickr 0 experiment_name

# Train on VGG-Sound 144k dataset
bash train_script.sh vggss_144k vggss 0
```

### Training Parameters

- `$1`: Training dataset (vggss_10k, vggss_144k, flickr_10k, flickr_144k)
- `$2`: Test dataset (vggss, vggss_heard, vggss_unheard, flickr, ms3, s4, all)
- `$3`: GPU device ID
- `$4`: Experiment name (optional)

### Key Hyperparameters

## Testing

```bash
python test_model.py \
    --model_dir ./checkpoints \
    --experiment_name your_experiment \
    --testset vggss \
    --test_data_path /path/to/test/data \
    --test_gt_path metadata/vggss.json \
    --gpu 0
```

## Logging

Training progress is logged using Weights & Biases (wandb). Set `--wandb true` to enable online logging.

## Dataset Preparation

Update the dataset paths in `train_script.sh`:

```bash
# For VGG-Sound
train_path="/path/to/VGGSound/"
test_data_path="/path/to/VGGSound/"
test_gt_path="metadata/vggss.json"

# For Flickr
train_path="/path/to/flickr_trainset/"
test_data_path="/path/to/Flickr/test/"
test_gt_path="/path/to/Flickr/test/Annotations/"
```

## Citation
If you use this code in your research, please cite:

```bibtex
@inproceedings{kim2025improving,
  title={Improving Sound Source Localization with Joint Slot Attention on Image and Audio},
  author={Kim, Inho and Song, Youngkil and Park, Jicheol and Kim, Won Hwa and Kwak, Suha},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={3121--3130},
  year={2025}
}
```

## Acknowledgments
We sincerely thank [EZ-VSL](https://github.com/stoneMo/EZ-VSL) and [FNAC_AVL](https://github.com/OpenNLPLab/FNAC_AVL) for their great codebase.

# leaffliction

An innovative computer vision project utilizing leaf image analysis for disease recognition.

## How to use

### Download dataset

```bash
# Download image dataset and generate distribution chart image
python 01.Distribution.py apple grape
```

### Data augmentation

```bash
# Augment unbalanced image dataset
python 02.Augmentation.py
```

```
Auto image augmentation...

Augmenting "Apple" images...
========================
Summary of augmentation:
========================
/Users/woolim/Documents/leaffliction/images/Apple_scab: 629 -> 1640
/Users/woolim/Documents/leaffliction/images/Apple_healthy: 1640 -> 1640
/Users/woolim/Documents/leaffliction/images/Apple_rust: 275 -> 1640
/Users/woolim/Documents/leaffliction/images/Apple_Black_rot: 620 -> 1640

Augmenting "Grape" images...
========================
Summary of augmentation:
========================
/Users/woolim/Documents/leaffliction/images/Grape_Esca: 1382 -> 1382
/Users/woolim/Documents/leaffliction/images/Grape_healthy: 422 -> 1382
/Users/woolim/Documents/leaffliction/images/Grape_Black_rot: 1178 -> 1382
/Users/woolim/Documents/leaffliction/images/Grape_spot: 1075 -> 1382
```

# Check image distributions are well balanced

```bash
python 01.Distribution.py apple grape
```

|           apple before            |          apple after           |
| :-------------------------------: | :----------------------------: |
| ![](./demo/apple_non_blanced.png) | ![](./demo/apple_balanced.png) |

|            grape before            |          grape after           |
| :--------------------------------: | :----------------------------: |
| ![](./demo/grape_non_balanced.png) | ![](./demo/grape_balanced.png) |

# Save transformed image plot

```bash
python 03.Transformation.py -src [SRC_PATH] -dst [DST_PATH]
```

|           image transformed            |           image transformed            |
| :------------------------------------: | :------------------------------------: |
| ![](./demo/image_transformation_1.png) | ![](./demo/image_transformation_2.png) |

# Predict an image

```bash
python predict.py [image_path]
```

![](./demo/predicted.png)

# Predict all images and check prediction accuracy.

```bash
python 04.Classification.py
```

```
Validation Progress: 100%|███████████████████████████████████████████| 10/10 [01:10<00:00,  7.01s/it]
Accuracy of the model on the validation set: 92.31%
```

## Tensorboard

```bash
tensorboard --logdir runs
```

|         Train loss         |         Train vs Validation Loss         |
| :------------------------: | :--------------------------------------: |
| ![](./demo/train_loss.png) | ![](./demo/train_vs_validation_loss.png) |

## Resources

- [Youtube Coursera CNN](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)

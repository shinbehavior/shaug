# shaug
Data Augmentation Tool

### req
```
python 3.8+
opencv
pillow
albumentations
```

### Usage
For now image augmentation only:
```
python image.py
```
It will open the next window:

![alt text](/source/full.png)

Choose params, tweak probs and range:

![alt text](/source/params.png)

Without checkbox, it will create 5 copies of the image with different augmentation. With checkbox, it will apply 5 random augmentations for ONE image copy.

![alt text](/source/batch_process.png)


import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import albumentations as albu


# affine変換
# flop
# bright
# shade
# 背景と合成

IMG_DIR = './image/'
MASK_DIR = './label/'

class SegmentationDataset():
    def __init__(self, img_dir, mask_dir, transforms):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.bg_dir = 'background'
        self.transforms = transforms
        self.img_ids = sorted(os.listdir(self.img_dir))
        self.mask_ids = sorted(os.listdir(self.mask_dir))
        self.bg_ids = sorted(os.listdir(self.bg_dir))


    def getitem(self, idx):
        img_name = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_name = self.mask_ids[idx]
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # 背景合成
        bg_name = self.bg_ids[idx]
        img_bg = cv2.imread(os.path.join(self.bg_dir, bg_name))
        img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB)
        # マスク画像の白黒を反転
        img_maskn = cv2.bitwise_not(mask)

        # 背景からimg_msknの部分を切り出す
        img_bg = cv2.bitwise_and(img_bg, img_maskn)

        # 背景と合成
        img = cv2.bitwise_or(img_bg, img)

        augmented = self.transforms(image=img, mask=mask)
        img, mask = augmented['image'], augmented['mask']


        return img, mask

    def random_erasing(self, x):
        image = np.zeros_like(x)
        size = x.shape[2]
        offset = np.random.randint(-4, 5, size=(2,))
        mirror = np.random.randint(2)
        remove = np.random.randint(2)
        top, left = offset
        left = max(0, left)
        top = max(0, top)
        right = min(size, left + size)
        bottom = min(size, top + size)
        if mirror > 0:
            x = x[:, :, ::-1]
        image[:, size - bottom:size - top, size - right:size - left] = x[:, top:bottom, left:right]
        # Remove erasing start
        if remove > 0:
            while True:
                s = np.random.uniform(0.02, 0.4) * size * size
                r = np.random.uniform(-np.log(3.0), np.log(3.0))
                r = np.exp(r)
                w = int(np.sqrt(s / r))
                h = int(np.sqrt(s * r))
                left = np.random.randint(0, size)
                top = np.random.randint(0, size)
                if left + w < size and top + h < size:
                    break
            c = np.random.randint(-128, 128)
            image[:, top:top + h, left:left + w] = c
        # Remove erasing end
        return image


def get_augmentation():
    train_transform = [
        albu.Blur(),
        albu.GaussNoise(),
        albu.ShiftScaleRotate(shift_limit=0.4, scale_limit=0.2, rotate_limit=90),
        albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10),
        albu.RandomBrightnessContrast(),
        albu.RandomShadow(shadow_dimension=4),
    ]
    return albu.Compose(train_transform)

transforms = get_augmentation()

dataset = SegmentationDataset(IMG_DIR, MASK_DIR, transforms)

img, mask = dataset.getitem(1)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].imshow(img)
axes[1].imshow(mask)
plt.show()

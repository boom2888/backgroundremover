from albumentations import HorizontalFlip, GridDistortion, OpticalDistortion, ChannelShuffle, CoarseDropout, CenterCrop, Crop, Rotate
from tqdm import tqdm
import cv2
import os
from src.utils.utility import load_data_raw,create_dir
from const.constants import raw_data_path,process_data_path
def augment_data(images, masks, save_path, augment=True):
    H = 512
    W = 512

    for x, y in tqdm(zip(images, masks), total=len(images)):
        name = x.split("/")[-1].split(".")[0]
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)
        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            x2 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            y2 = y

            aug = ChannelShuffle(p=1)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = CoarseDropout(
            num_holes_range=(3, 10),
            hole_height_range=(8, 32),   # pixels
            hole_width_range=(8, 32),    # pixels
            p=1.0
             )

            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented["image"]
            y5 = augmented["mask"]

            X = [x, x1, x2, x3, x4, x5]
            Y = [y, y1, y2, y3, y4, y5]

        else:
            X = [x]
            Y = [y]


        index = 0
        for i, m in zip(X, Y):
            try:
                """ Center Cropping """
                aug = CenterCrop(H, W, p=1.0)
                augmented = aug(image=i, mask=m)
                i = augmented["image"]
                m = augmented["mask"]

            except Exception as e:
                i = cv2.resize(i, (W, H))
                m = cv2.resize(m, (W, H))

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_image_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1

(train_x, train_y), (test_x, test_y) = load_data_raw(raw_data_path,0.1)
create_dir(f"{process_data_path}/train/image/")
create_dir(f"{process_data_path}/train/mask/")
create_dir(f"{process_data_path}/test/image/")
create_dir(f"{process_data_path}/test/mask/")

augment_data(train_x, train_y, f"{process_data_path}/train/", augment=True)
augment_data(test_x, test_y, f"{process_data_path}/test/", augment=False)
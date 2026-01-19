import os
from sklearn.model_selection import train_test_split
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import cv2

import matplotlib.pyplot as plt

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data_raw(path, split=0.1):
     X = sorted(glob(os.path.join(path, "images", "*.jpg")))
     Y = sorted(glob(os.path.join(path, "masks", "*.png")))
     split_size = int(len(X) * split)
     train_x, test_x = train_test_split(X, test_size=split_size, random_state=42)
     train_y, test_y = train_test_split(Y, test_size=split_size, random_state=42)
     return (train_x, train_y), (test_x, test_y)
def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_data_processed(path):
    x = sorted(glob(os.path.join(path, "image", "*png")))
    y = sorted(glob(os.path.join(path, "mask", "*png")))
    return x, y

def read_image(path,H,W):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))  # Resize to (512, 512)
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(path,H,W):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))  # Resize to (512, 512)
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y,H,W):
    def _parse(x, y):
        x = read_image(x,H,W)
        y = read_mask(y,H,W)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y
    
def save_results(H,image, mask, y_pred, save_image_path):
    ## i - m - yp - yp*i
    line = np.ones((H, 10, 3)) * 128

    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    mask = mask * 255

    y_pred = np.expand_dims(y_pred, axis=-1)    ## (512, 512, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)  ## (512, 512, 3)

    masked_image = image * y_pred
    y_pred = y_pred * 255

    cat_images = np.concatenate([image, line, mask, line, y_pred, line, masked_image], axis=1)
    cv2.imwrite(save_image_path, cat_images)

def save_training_plots(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # ---- Loss ----
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.close()

    # ---- Dice Coefficient ----
    if "dice_coef" in history.history:
        plt.figure()
        plt.plot(history.history["dice_coef"], label="Train Dice")
        plt.plot(history.history["val_dice_coef"], label="Val Dice")
        plt.xlabel("Epochs")
        plt.ylabel("Dice Coef")
        plt.legend()
        plt.title("Dice Coefficient")
        plt.savefig(os.path.join(save_dir, "dice_coef.png"))
        plt.close()

    # ---- IoU ----
    if "iou" in history.history:
        plt.figure()
        plt.plot(history.history["iou"], label="Train IoU")
        plt.plot(history.history["val_iou"], label="Val IoU")
        plt.xlabel("Epochs")
        plt.ylabel("IoU")
        plt.legend()
        plt.title("Intersection over Union")
        plt.savefig(os.path.join(save_dir, "iou.png"))
        plt.close()

    # ---- Precision ----
    if "precision" in history.history:
        plt.figure()
        plt.plot(history.history["precision"], label="Train Precision")
        plt.plot(history.history["val_precision"], label="Val Precision")
        plt.xlabel("Epochs")
        plt.ylabel("Precision")
        plt.legend()
        plt.title("Precision")
        plt.savefig(os.path.join(save_dir, "precision.png"))
        plt.close()

    # ---- Recall ----
    if "recall" in history.history:
        plt.figure()
        plt.plot(history.history["recall"], label="Train Recall")
        plt.plot(history.history["val_recall"], label="Val Recall")
        plt.xlabel("Epochs")
        plt.ylabel("Recall")
        plt.legend()
        plt.title("Recall")
        plt.savefig(os.path.join(save_dir, "recall.png"))
        plt.close()

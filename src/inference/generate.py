from const.constants import process_data_path,ARTIFACTS_PATH
from src.utils.utility import save_results,load_data_processed,create_dir
import os
from tqdm import tqdm
import cv2
import  numpy as np
import random
import matplotlib.pyplot as plt

def  validation(model,H,no_show=5):
    valid_path = os.path.join(process_data_path, "test")
    test_x, test_y = load_data_processed(valid_path)
    test_x=test_x[:5]
    test_y=test_y[:5]
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x = image/255.0
        x = np.expand_dims(x, axis=0)

        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        """ Prediction """
        y_pred = model.predict(x)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)

        """ Saving the prediction """
        folder_path=os.path.join(ARTIFACTS_PATH,"test_images_result")
        create_dir(folder_path)
        save_image_path = f"{folder_path}/{name}.png"
        save_results(H,image, mask, y_pred, save_image_path)
   
            
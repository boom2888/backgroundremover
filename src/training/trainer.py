
from const.constants import model_config_path,train_config_path,ARTIFACTS_PATH
import yaml
from tensorflow.keras.callbacks import ModelCheckpoint


from src.utils.utility import load_data_processed,shuffling,create_dir
from src.data.dataset import tf_dataset
from src.model.deep3lab import deeplabv3_plus
from src.training.loss import dice_loss,dice_coef,iou
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from const.constants import CHECKPOINT_PTH
class Trainer():
    def __init__(self,batch,lr,epochs,train_path,validation_path,H,W):
        self.batch=batch
        self.lr=lr
        self.epochs=epochs
        self.train_path=train_path
        self.validation_path=validation_path
        self.H=H
        self.W=W
        self.model=deeplabv3_plus((self.H,self.W, 3))
    
    def train(self):
        train_x, train_y = load_data_processed(self.train_path)
        train_x, train_y = shuffling(train_x, train_y)
        valid_x, valid_y = load_data_processed(self.validation_path)
        train_dataset = tf_dataset(train_x, train_y, batch=self.batch,H=self.H,W=self.W)
        valid_dataset = tf_dataset(valid_x, valid_y, batch=self.batch,H=self.H,W=self.W)
        self.model.compile(loss=dice_loss, optimizer=Adam(learning_rate=self.lr), metrics=[dice_coef, iou, Recall(), Precision()])
        create_dir(ARTIFACTS_PATH)
        checkpoint = ModelCheckpoint(
        f"{ARTIFACTS_PATH}/best_model.h5",
        monitor="val_dice_coef",
        mode="max",
        save_best_only=True,
        verbose=1
        )
        history=self.model.fit(
        train_dataset,
        epochs=self.epochs,
        validation_data=valid_dataset,
        steps_per_epoch=5, 
        callbacks=[checkpoint]
        

       ) 
        return history

    


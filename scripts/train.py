
from const.constants import model_config_path,train_config_path,ARTIFACTS_PATH,process_data_path
import yaml
from src.training.trainer import Trainer
import os
from src.utils.utility import save_training_plots
from src.inference.generate import validation
with open(train_config_path, 'r') as file:
    data = yaml.safe_load(file)

batch=data["batch"]
lr=data["lr"]
num_epochs=data["num_epochs"]
H=data["H"]
W=data["W"]
train_path=os.path.join(process_data_path,"train")
validation_path=os.path.join(process_data_path,"test")
plot_save_path=os.path.join(ARTIFACTS_PATH,"plots")
trainer=Trainer(batch,lr,num_epochs,train_path,validation_path,H,W)
history=trainer.train()
save_training_plots(history,plot_save_path)
validation(trainer.model,H)
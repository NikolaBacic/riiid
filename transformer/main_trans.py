# IMPORTS

import os
import pickle

import torch

# my packages
from dataset import RiiidData
from model_trans import SaintNikolaTransformer
from train import train_model
from utils import seed_everything, get_ma

# dataset config
from config_trans import max_quest, num_quest, start_response_token, sequel_response_token, batch_size
# model config
from config_trans import head_dim, nhead, dim_feedforward, num_encoder_layers
# training config
from config_trans import lr, epochs, warmup_steps

from config_trans import random_seed
seed_everything(random_seed)

# visible GPU card
gpu_idx = 1
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_idx}"
device = torch.device("cuda")

# load data container
with open("../models/riiid_container/questions_container.pickle", "rb") as handle:
    questions_container = pickle.load(handle)

# TRAIN AND VALIDATION DATALOADERS
data = RiiidData(questions_container, max_quest, num_quest, start_response_token, sequel_response_token, batch_size)
data.sampling_process()
train_dataloader, val_dataloader = data.get_dataloaders(val_size=0.025)

# INITIALIZE A MODEL INSTANCE
model = SaintNikolaTransformer(device, num_quest, max_quest, head_dim, nhead, dim_feedforward, num_encoder_layers)
model.init_weights()
model.to(device)
print(f"The model has {model.num_parameters()} of trainable parameters!")

# TRAIN THE MODEL
model, train_loss = train_model(model, device, train_dataloader, val_dataloader, lr, epochs, warmup_steps)

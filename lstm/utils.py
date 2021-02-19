"""misc of useful functions"""

import pandas as pd
import numpy as np
import random
import torch
import os

from tqdm import tqdm

from config_gru import no_elapsed


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_ma(train_losses, window_size=40):
    train_losses_pd = pd.Series(train_losses)
    windows = train_losses_pd.rolling(window_size)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    without_nans = moving_averages_list[window_size - 1:]
    
    return without_nans


def csv_to_dict(df):
    """maps the training data from csv to dictionary"""
    # separate question events
    questions_df = df[df["content_type_id"] == False]
        
    # fill nans
    questions_df["prior_question_elapsed_time"] = questions_df["prior_question_elapsed_time"].fillna(no_elapsed)
        
    # questions history container
    questions_container = dict(tuple(questions_df.groupby(["user_id"])))
    # df -> tensors
    for user_id in questions_container.keys():
        tmp_df = questions_container[user_id]
        questions_container[user_id] = {
            "content_id": torch.LongTensor(tmp_df["content_id"].values),
            "timestamp": torch.LongTensor(tmp_df["timestamp"].values),
            "prior_question_elapsed_time": torch.LongTensor(tmp_df["prior_question_elapsed_time"].values),
            "answered_correctly": torch.ShortTensor(tmp_df["answered_correctly"].values),
            }
            
    return questions_container


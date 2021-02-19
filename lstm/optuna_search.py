# IMPORTS

import os
import pickle

import torch

# my packages
from dataset import RiiidData
from model_lstm import SaintNikolaLSTM
from train import evaluate
from utils import seed_everything, get_ma

# dataset config
from config_lstm import max_quest, num_quest, start_response_token, sequel_response_token, batch_size
# model config
from config_lstm import input_size_lstm, hidden_size_lstm, num_layers_lstm
# training config
from config_lstm import epochs

from config_lstm import random_seed
seed_everything(random_seed)

# visible GPU card
gpu_idx = 2
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_idx}"
device = torch.device("cuda")

import optuna

from transformers import get_cosine_schedule_with_warmup
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
loss_bce = BCEWithLogitsLoss()

# load data container
with open("../data/questions_container_20m.pickle", "rb") as handle:
    questions_container = pickle.load(handle)

# TRAIN AND VALIDATION DATALOADERS
data = RiiidData(questions_container, max_quest, num_quest, start_response_token, sequel_response_token, batch_size)
data.sampling_process()
train_dataloader, val_dataloader = data.get_dataloaders(val_size=0.125)


def define_model(trial):
    
    # INITIALIZE A MODEL INSTANCE
    model = SaintNikolaLSTM(device, num_quest, max_quest, input_size_lstm, hidden_size_lstm, num_layers_lstm)
    model.init_weights()
    model.to(device)

    return model


def objective(trial):

    # Generate the model.
    model = define_model(trial).to(device)
    
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    # define optimizer - Adam
    optimizer = Adam(model.parameters(), lr=lr)
    # # define lr scheduler
    num_train_steps = epochs * len(train_dataloader)
    warmup_steps = trial.suggest_int("warmup_steps", 0, 300, 10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_train_steps)
    
    
    for epoch in range(epochs):
        
        model.train()
        
        for batch in train_dataloader:
            
            # clean the gradients
            model.zero_grad()

            # add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            quest_ids, prior_elapsed, quest_lag, target, response, _, loss_mask = batch

            # get model's prediction
            output  = model(quest_ids, prior_elapsed, quest_lag, response)

            # mask padded tgt elements
            loss_mask = loss_mask.view(-1)
            
            # calculate loss 1
            loss = loss_bce(output.view(-1)[loss_mask], target.view(-1)[loss_mask])
                                                
            # calculate gradients
            loss.backward()
            
            # update model param
            optimizer.step()

            # update the learning rate
            scheduler.step()

        val_auc = evaluate(model, device, val_dataloader)
    
        trial.report(val_auc, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_auc


if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, timeout=60000000)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    # save the study
    with open("./study.pickle", "wb") as handle:
        pickle.dump(study, handle, protocol=pickle.HIGHEST_PROTOCOL)


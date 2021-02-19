"""
Train and validation loops
"""

from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import pickle
import torch

from loss import loss_bce, init_opt_sch


def train_model(model, device, train_dataloader, val_dataloader, lr, epochs, warmup_steps):
    
    print("model training process...")

    train_losses = []
    
    optimizer, scheduler = init_opt_sch(model, train_dataloader, lr, epochs, warmup_steps)
    
    for epoch in range(epochs):
        
        model.train()
        
        for batch in tqdm(train_dataloader):
            
            # clean the gradients
            model.zero_grad()

            # add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            quest_ids, prior_elapsed, quest_lag, target, response, _, loss_mask = batch

            # get model's prediction
            output  = model(quest_ids, prior_elapsed, quest_lag, response)

            # mask padded tgt elements
            loss_mask = loss_mask.view(-1)
            
            # calculate loss
            loss = loss_bce(output.view(-1)[loss_mask], target.view(-1)[loss_mask])
                                    
            # remember train loss
            train_losses.append(loss.item())
            
            # calculate gradients
            loss.backward()
            
            # update model param
            optimizer.step()

            # update the learning rate
            scheduler.step()
            
#         # TRAIN AND VALIDATION ROC AUC SCORE
#         train_auc = evaluate(model, device, train_dataloader)
#         val_auc = evaluate(model, device, val_dataloader)
#         print(f"Train auc: {round(train_auc, 4)} ### Validation auc: {round(val_auc, 4)}")

        torch.save(model.state_dict(), f"./model_gru_{epoch}.pt")
        
        # VALIDATION ROC AUC SCORE
        val_auc = evaluate(model, device, val_dataloader)
        print(f"Validation auc: {round(val_auc, 4)}")
        
    return model, train_losses


def evaluate(model, device, dataloader):
    
    targets = []
    preds = []
    model.eval()

    for batch in dataloader:

        # add batch to GPU
        batch = tuple(t.to(device) for t in batch) 

        quest_ids, prior_elapsed, quest_lag, target, response, _, loss_mask = batch

        with torch.no_grad():
            # get model's prediction
            output  = model(quest_ids, prior_elapsed, quest_lag, response)

        # mask padded elements
        loss_mask = loss_mask.view(-1)

        targets.append(target.view(-1)[loss_mask].cpu())
        preds.append(output.view(-1)[loss_mask].cpu())
    
    targets = torch.cat(targets)
    preds = torch.cat(preds).sigmoid()
    
    # save files
    with open("./preds_gru.pt", "wb") as handle:
        pickle.dump(preds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("./targets_gru.pt", "wb") as handle:
        pickle.dump(targets, handle, protocol=pickle.HIGHEST_PROTOCOL)

    score = roc_auc_score(targets, preds)    
    
    return score
    
    

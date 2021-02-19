from transformers import get_cosine_schedule_with_warmup
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

# initialize loss
loss_bce = BCEWithLogitsLoss()

def init_opt_sch(model, train_dataloader, lr, epochs, warmup_steps):
    """
    define optimzier and lr scheduler
    """
    # define optimizer - Adam
    optimizer = Adam(model.parameters(), lr=lr)
    # define lr scheduler
    num_train_steps = epochs * len(train_dataloader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_train_steps)
    
    return optimizer, scheduler



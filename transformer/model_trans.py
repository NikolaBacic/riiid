"""Transformers based model"""

import torch
from torch import nn

class SaintNikolaTransformer(nn.Module):
    """
    An encoder only model.
    """
    
    def __init__(self, device, num_quest, max_quest, head_dim, nhead, dim_feedforward, num_encoder_layers):
        super(SaintNikolaTransformer, self).__init__()
        
        self.device = device
        self.num_quest = num_quest
        self.max_quest = max_quest
        self.d_model = head_dim * nhead
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.num_encoder_layers = num_encoder_layers
        
        # same for each sample -> define once
        self.src_mask = (torch.tril(torch.ones((max_quest, max_quest), dtype=torch.bool)) == False).to(device)
        self.zero_tensor = torch.zeros(1, dtype=torch.long).to(device)
        
        # LAYERS
        self.quest_embedd = nn.Embedding(num_quest+1, self.d_model, padding_idx=num_quest) # +1 is for the padding token
        self.response_embedd = nn.Embedding(4, self.d_model)
        self.minute_quest_lag_embedd = nn.Embedding(1, self.d_model)
        self.minute_prior_elapsed_embedd = nn.Embedding(1, self.d_model)
                              
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
                                                        activation="relu", dropout=0.0)

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        
        self.clf = nn.Linear(self.d_model, 1)
        
    
    def init_weights(self):
        """
        initialize model's weights with xavier uniform
        """
        for par in self.parameters():
            if par.dim() > 1:
                nn.init.xavier_uniform_(par)
                
                
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        
    def forward(self, quest_ids, prior_elapsed, quest_lag, response, quest_padd_mask):
                
        # pull the embeddings
        quest_ids = self.quest_embedd(quest_ids).permute(1, 0, 2)
        response = self.response_embedd(response).permute(1, 0, 2)
        minute_prior_elapsed = self.minute_prior_elapsed_embedd(self.zero_tensor)
        minute_quest_lag = self.minute_quest_lag_embedd(self.zero_tensor)
        
        # apply ln(1+x) function on prior elapsed
        prior_elapsed = torch.log(prior_elapsed+1)
        # lag * minute calculation
        prior_elapsed = prior_elapsed.repeat(self.d_model, 1, 1).permute(2, 1, 0)
        prior_elapsed = torch.mul(prior_elapsed, minute_prior_elapsed)
        
        # apply ln(1+x) function on quest lags
        quest_lag = torch.log(quest_lag+1)
        # lag * minute calculation
        quest_lag = quest_lag.repeat(self.d_model, 1, 1).permute(2, 1, 0)
        quest_lag = torch.mul(quest_lag, minute_quest_lag)
        
        src = quest_ids + response + prior_elapsed + quest_lag
        
        output = self.encoder(src=src, mask=self.src_mask, src_key_padding_mask=quest_padd_mask).permute(1, 0, 2)
        
        output = self.clf(output)
                
        return output


"""Long short-term memory based model"""

import torch
from torch import nn

class SaintNikolaLSTM(nn.Module):
    """
    LSTM model
    """
    
    def __init__(self, device, num_quest, max_quest, input_size, hidden_size, num_layers):
        super(SaintNikolaLSTM, self).__init__()
        
        self.device = device
        self.num_quest = num_quest
        self.max_quest = max_quest
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # same for each sample -> define once
        self.zero_tensor = torch.zeros(1, dtype=torch.long).to(device)
        
        # LAYERS
        self.quest_embedd = nn.Embedding(num_quest+1, input_size, padding_idx=num_quest) # +1 is for the padding token
        self.response_embedd = nn.Embedding(4, input_size)
        self.minute_quest_lag_embedd = nn.Embedding(1, input_size)
        self.minute_prior_elapsed_embedd = nn.Embedding(1, input_size)
                              
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                         bias=True, batch_first=True, dropout=0.0, bidirectional=False)
        
        self.clf = nn.Linear(hidden_size, 1)
        
    
    def init_weights(self):
        """
        initialize model's weights with xavier uniform
        """
        for par in self.parameters():
            if par.dim() > 1:
                nn.init.xavier_uniform_(par)
                
                
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        
    def forward(self, quest_ids, prior_elapsed, quest_lag, response):
        
        # pull the embeddings
        quest_ids = self.quest_embedd(quest_ids)
        response = self.response_embedd(response)
        minute_prior_elapsed = self.minute_prior_elapsed_embedd(self.zero_tensor)
        minute_quest_lag = self.minute_quest_lag_embedd(self.zero_tensor)
        
        # apply ln function on prior elapsed
        prior_elapsed = torch.log(prior_elapsed+1)
        # lag * minute calculation
        prior_elapsed = prior_elapsed.repeat(self.input_size, 1, 1).permute(1, 2, 0)
        prior_elapsed = torch.mul(prior_elapsed, minute_prior_elapsed)
        
        # apply ln function on quest lags
        quest_lag = torch.log(quest_lag+1)
        # lag * minute calculation
        quest_lag = quest_lag.repeat(self.input_size, 1, 1).permute(1, 2, 0)
        quest_lag = torch.mul(quest_lag, minute_quest_lag)
        
        src = quest_ids + response + prior_elapsed + quest_lag
        
        output = self.lstm(src)[0]
                
        output = self.clf(output)
                
        return output


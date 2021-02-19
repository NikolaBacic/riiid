"""Data handling class"""

from tqdm import tqdm
from math import ceil

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler


class RiiidData:
    
    def __init__(self, questions_container, max_quest, num_quest, start_response_token, sequel_response_token, batch_size):

        self.questions_container = questions_container
        self.max_quest = max_quest
        self.step = max_quest // 2
        self.num_quest = num_quest
        self.start_response_token = start_response_token
        self.sequel_response_token = sequel_response_token
        self.batch_size = batch_size
        
        self.quest_ids_all = []
        self.prior_elapsed_all = []
        self.quest_lag_all = []
        self.target_all = []
        self.response_all = []
        self.quest_padd_mask_all = []
        self.loss_mask_all = []
        
    
    def _one_sample(self, quest_content_id, quest_timestamp, prior_elapsed, answered_correctly, num_rows):
        """
        sample a user when he has under max_quest answered questions
        """
        
        assert num_rows <= self.max_quest
        
        pad_num = self.max_quest - num_rows
        zeros_pad = torch.zeros(pad_num)
            
        # quest sample
        quest_pad = torch.full((pad_num,), self.num_quest)
        quest_sample = torch.cat((quest_content_id, quest_pad)).unsqueeze(0).type(torch.int64)
            
        # target sample
        target_sample = torch.cat((answered_correctly, zeros_pad)).unsqueeze(0).type(torch.float32)
            
        # prior elapsed sample
        prior_elapsed_sample = torch.div(prior_elapsed, 60000)
        prior_elapsed_sample = torch.cat((prior_elapsed_sample, zeros_pad)).unsqueeze(0).type(torch.float32)
        
        # quest lag sample
        quest_lag_sample = torch.div(quest_timestamp[1:] - quest_timestamp[:-1], 60000)
        quest_lag_sample = torch.cat((torch.zeros(1), quest_lag_sample))
        quest_lag_sample = torch.cat((quest_lag_sample, zeros_pad)).unsqueeze(0).type(torch.float32)
        
        # response sample
        response_sample = torch.cat((torch.tensor([self.start_response_token]), answered_correctly))[:-1]
        response_sample = torch.cat((response_sample, zeros_pad)).unsqueeze(0).type(torch.int64)
            
        # quest_padd_mask
        quest_padd_mask = (quest_sample == self.num_quest)

        # loss mask
        loss_mask = (quest_padd_mask == False)

        # append the sample
        self.quest_ids_all.append(quest_sample)
        self.prior_elapsed_all.append(prior_elapsed_sample)
        self.quest_lag_all.append(quest_lag_sample)
        self.target_all.append(target_sample)
        self.response_all.append(response_sample)
        self.quest_padd_mask_all.append(quest_padd_mask)
        self.loss_mask_all.append(loss_mask)
    
    
    def _mult_sample(self, quest_content_id, quest_timestamp, prior_elapsed, answered_correctly, num_rows):
        """
        sample a user when he has over max_quest answered questions
        """
        
        assert num_rows > self.max_quest
                
        if num_rows % self.step == 0:
            pad_num = 0
            num_samples = num_rows // self.step - 1
        else:
            pad_num = self.step - (num_rows % self.step)
            num_samples = (num_rows + pad_num) // self.step - 1
            
        zeros_pad = torch.zeros(pad_num)

        # quest tensor
        quest_pad = torch.full((pad_num,), self.num_quest)
        quest_content_id_t = torch.cat((quest_content_id, quest_pad)).unsqueeze(0).type(torch.int64)

        # target tensor
        target_t = torch.cat((answered_correctly, zeros_pad)).unsqueeze(0).type(torch.float32)

        # prior elapsed tensor
        prior_elapsed_t = torch.div(prior_elapsed, 60000)
        prior_elapsed_t = torch.cat((prior_elapsed_t, zeros_pad)).unsqueeze(0).type(torch.float32)

        # quest lag tensor    
        quest_lag_t = torch.div(quest_timestamp[1:] - quest_timestamp[:-1], 60000)
        quest_lag_t = torch.cat((torch.zeros(1), quest_lag_t))
        quest_lag_t = torch.cat((quest_lag_t, zeros_pad)).unsqueeze(0).type(torch.float32)

        # response tensor
        response_t = torch.cat((torch.tensor([self.start_response_token]), answered_correctly))[:-1]
        response_t = torch.cat((response_t, zeros_pad)).unsqueeze(0).type(torch.int64)
                
        # get the first sample
        quest_samples = quest_content_id_t[:, :self.max_quest]
        target_samples = target_t[:, :self.max_quest]
        prior_elapsed_samples = prior_elapsed_t[:, :self.max_quest]
        quest_lag_samples = quest_lag_t[:, :self.max_quest]
        response_samples = response_t[:, :self.max_quest]
        
        # get the rest
        for s in range(1, num_samples):
            quest_samples = torch.cat((quest_samples, quest_content_id_t[:, s*self.step:s*self.step+self.max_quest]))
            target_samples = torch.cat((target_samples, target_t[:, s*self.step:s*self.step+self.max_quest]))
            prior_elapsed_samples = torch.cat((prior_elapsed_samples, prior_elapsed_t[:, s*self.step:s*self.step+self.max_quest]))
            quest_lag_samples = torch.cat((quest_lag_samples, quest_lag_t[:, s*self.step:s*self.step+self.max_quest]))            
            response_samples = torch.cat((response_samples, response_t[:, s*self.step:s*self.step+self.max_quest]))
        response_samples[1:, 0] = self.sequel_response_token

        # quest_padd_mask
        quest_padd_mask = (quest_samples == self.num_quest)
        # loss mask
        loss_mask = (quest_padd_mask == False)
        loss_mask[1:, :self.step] = False        
        
        # append the samples
        self.quest_ids_all.append(quest_samples)
        self.prior_elapsed_all.append(prior_elapsed_samples)
        self.quest_lag_all.append(quest_lag_samples)
        self.target_all.append(target_samples)
        self.response_all.append(response_samples)
        self.quest_padd_mask_all.append(quest_padd_mask)
        self.loss_mask_all.append(loss_mask)
        
    
    def sampling_process(self):
        """
        iterate trough questions container and make samples from users
        """
        
        for user_id in self.questions_container.keys():
            # data retrieve
            quest_content_id = self.questions_container[user_id]["content_id"]
            quest_timestamp = self.questions_container[user_id]["timestamp"]
            prior_elapsed = self.questions_container[user_id]["prior_question_elapsed_time"]
            answered_correctly = self.questions_container[user_id]["answered_correctly"]
            
            num_rows = quest_content_id.shape[0]
            
            if num_rows <= self.max_quest:
                self._one_sample(quest_content_id, quest_timestamp, prior_elapsed, answered_correctly, num_rows)
            else:
                self._mult_sample(quest_content_id, quest_timestamp, prior_elapsed, answered_correctly, num_rows)
            
                
    def get_dataloaders(self, val_size=0.1):
        
        dataset = TensorDataset(torch.cat(self.quest_ids_all),
                                torch.cat(self.prior_elapsed_all),
                                torch.cat(self.quest_lag_all),
                                torch.cat(self.target_all),
                                torch.cat(self.response_all),
                                torch.cat(self.quest_padd_mask_all),
                                torch.cat(self.loss_mask_all),
        )
        
                
        train_data, val_data = train_test_split(dataset, test_size=val_size, shuffle=False)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=self.batch_size) 
        
        return train_dataloader, val_dataloader
    


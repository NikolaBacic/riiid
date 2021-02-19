"""Configuration parameters/hyperparameters file"""

# random seed
random_seed = 42

# maximum number of questions in the encoder's input
max_quest = 300

num_quest = 13523 # number of questions; equal to input padding index
no_elapsed = 301000 # "prior_question_elapsed_time" fillna value
start_response_token = 2
sequel_response_token = 3

# encoder's hyperparameters
nhead = 8
head_dim = 60
dim_feedforward = 2048
num_encoder_layers = 8

# optimization parameters
epochs = 6
batch_size = 64
lr = 0.00019809259513409007
warmup_steps = 150*5


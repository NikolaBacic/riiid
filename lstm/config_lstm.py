"""LSTM configuration parameters/hyperparameters file"""

# random seed
random_seed = 42

# maximum number of questions in the encoder's input
max_quest = 300

num_quest = 13523 # number of questions; equal to input padding index
no_elapsed = 301000 # "prior_question_elapsed_time" fillna value
start_response_token = 2
sequel_response_token = 3

# network's hyperparameters
input_size_lstm = 384
hidden_size_lstm = 768
num_layers_lstm = 4

# optimization parameters
epochs = 4
batch_size = 64
lr = 0.0007019926812886481
warmup_steps = 100


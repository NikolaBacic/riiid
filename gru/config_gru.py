"""GRU configuration parameters/hyperparameters file"""

# random seed
random_seed = 42

# maximum number of questions in the encoder's input
max_quest = 300

num_quest = 13523 # number of questions; equal to input padding index
no_elapsed = 301000 # "prior_question_elapsed_time" fillna value
start_response_token = 2
sequel_response_token = 3

# network's hyperparameters
input_size_gru = 320
hidden_size_gru = 512
num_layers_gru = 3

# optimization parameters
epochs = 4
batch_size = 64
lr = 0.0008419253431185227
warmup_steps = 80


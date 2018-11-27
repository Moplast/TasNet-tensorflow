sr = 8000
sample_duration = 0.005 # s
L = int(sr * sample_duration)   # [40]
N = 500 # num_basis
nspk = 2
batch_size = 128
epochs = 100
seq_duration = 0.5 # s
seq_len = int(seq_duration / sample_duration)

cuda = True
seed = 20181117
log_step = 100
lr = 3e-4

rnn_type = 'LSTM'
rnn_hidden_size = 500
num_layers = 4
bidirectional = True

display_freq = 10
val_save = 'model_181117.pt'
data_dir = '/home/grz/data/SSSR/wsj0/min/'
sum_dir = './tasnet/summary/'
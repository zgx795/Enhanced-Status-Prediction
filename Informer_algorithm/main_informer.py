# Informer-based Digital Twin Framework for Forced Draft Fans Status Forecasting in Coal-Fired Power Units
# Corresponding code for manuscript review and reproducibility

import argparse
import torch
from exp.exp_informer import Exp_Informer

# Argument parser initialization
parser = argparse.ArgumentParser(description='Informer Model for Forced Draft Fan Status Forecasting')

# Basic configurations
parser.add_argument('--model', type=str, default='informer', help='Model type: [informer, informerstack, informerlight(TBD)]')
parser.add_argument('--data', type=str, default='MSS', help='Dataset identifier')
parser.add_argument('--root_path', type=str, default='./data/', help='Root directory path for dataset')
parser.add_argument('--data_path', type=str, default='MSS.csv', help='Dataset filename')
parser.add_argument('--features', type=str, default='M', help='Prediction type: [M: Multivariate->Multivariate, S: Univariate->Univariate, MS: Multivariate->Univariate]')
parser.add_argument('--target', type=str, default='OT', help='Target feature column for univariate prediction')
parser.add_argument('--freq', type=str, default='10s', help='Sampling frequency for time features encoding')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Path to save model checkpoints')

# Informer model hyperparameters
parser.add_argument('--seq_len', type=int, default=48*16, help='Input sequence length for encoder')
parser.add_argument('--label_len', type=int, default=24*16, help='Length of start token for decoder')
parser.add_argument('--pred_len', type=int, default=12*16, help='Length of prediction output sequence')
parser.add_argument('--enc_in', type=int, default=11, help='Number of encoder input features')
parser.add_argument('--dec_in', type=int, default=6, help='Number of decoder input features')
parser.add_argument('--c_out', type=int, default=11, help='Number of output features')
parser.add_argument('--d_model', type=int, default=512, help='Model dimensionality')
parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--e_layers', type=int, default=2, help='Encoder layer count')
parser.add_argument('--d_layers', type=int, default=1, help='Decoder layer count')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='Stacked encoder layer configuration')
parser.add_argument('--d_ff', type=int, default=2048, help='Dimension of feed-forward network')
parser.add_argument('--factor', type=int, default=5, help='ProbSparse attention factor')
parser.add_argument('--distil', action='store_false', help='Use distillation in encoder (default: True)', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='Dropout rate')
parser.add_argument('--attn', type=str, default='prob', help='Type of attention: [prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='Time features encoding method: [timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='Activation function')

# Training and optimization parameters
parser.add_argument('--itr', type=int, default=2, help='Number of experiment iterations')
parser.add_argument('--train_epochs', type=int, default=6, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--loss', type=str, default='mse', help='Loss function')
parser.add_argument('--lradj', type=str, default='type1', help='Learning rate adjustment strategy')
parser.add_argument('--use_amp', action='store_true', help='Use mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='Inverse transform output data', default=True)

# GPU configuration
parser.add_argument('--use_gpu', type=bool, default=True, help='Use GPU for training')
parser.add_argument('--gpu', type=int, default=0, help='GPU device index')
parser.add_argument('--use_multi_gpu', action='store_true', help='Enable multi-GPU training', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='Multi-GPU device indices')

# Parse arguments
args = parser.parse_args()

# GPU availability check
args.use_gpu = torch.cuda.is_available() and args.use_gpu
if args.use_gpu and args.use_multi_gpu:
    device_ids = args.devices.replace(' ', '').split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# Dataset-specific configurations
DATA_CONFIGS = {
    'MSS': {'data': 'MSS.csv', 'T': 'PCCT', 'M': [11, 11, 11], 'S': [1, 1, 1], 'MS': [11, 11, 1]},
}

if args.data in DATA_CONFIGS:
    data_info = DATA_CONFIGS[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

# Prepare stacked encoder layer configuration
args.s_layers = [int(layer) for layer in args.s_layers.replace(' ', '').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

# Print experiment arguments
print('Experiment configuration:')
print(args)

# Run experiments
for ii in range(args.itr):
    # Experiment identifier for saving results
    setting = f'{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_' \
              f'dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_' \
              f'at{args.attn}_fc{args.factor}_eb{args.embed}_dt{args.distil}_mx{args.mix}_{args.des}_{ii}'

    # Initialize experiment
    exp = Exp_Informer(args)

    print(f'>>>>> Starting training: {setting}')
    exp.train(setting)

    print(f'>>>>> Testing model: {setting}')
    exp.test(setting)

    if args.do_predict:
        print(f'>>>>> Predicting unseen data: {setting}')
        exp.predict(setting, True)

    # Free GPU memory
    torch.cuda.empty_cache()

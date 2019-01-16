import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10, help="Epoch to train [5]")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate of adam [0.001]")
    parser.add_argument("--batch_size", type=int, default=50, help="The number of batch images [64]")
    parser.add_argument('--data_ratio', nargs='+', type=int, default=[8, 1, 1])
    parser.add_argument("--data_dir", default='./dataset', help="The directory of files")
    parser.add_argument("--log_dir", default="log", help="The directory for logging")
    parser.add_argument("--graph_dir", default="board", help="The directory for tensorboard logging")
    parser.add_argument("--checkpoint_dir", default="npz", help="Directory name to save the checkpoints [checkpoint]")
    parser.add_argument("--img_resolution", type=int, default=128, help="The resolution of images")
    parser.add_argument("--print_freq", type=int, default=1, help="The interval of batches to print")
    parser.add_argument("--save_interval", type=int, default=1, help="The interval of saving npz")
    parser.add_argument('--mode', default='train', help='mode [train, test]')

    # Arguments for loading 
    parser.add_argument("--load_time", default=None, help="load time")
    parser.add_argument("--max_acc", type=float, default=0.7578, help="used when load the pretrained model")
    parser.add_argument("--npz_to_load", default=None, help="used when load the pretrained model")

    args = parser.parse_args()
    return args

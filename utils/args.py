import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--class_img', type=int, default=-1, help='n. of images per class after augments')
    parser.add_argument('--k_splits', type=int, default=8, help='n. of images per class after augments')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'svhn', 'imagenet'], required=True, help='dataset name')
    parser.add_argument('--qm', type=str, default="remove", choices=['replace', 'remove'], required=False, help='Replace or remove question marks')
    parser.add_argument('--save_model', action='store_true', required=False, help='Save the model')

    # parser.add_argument('--test_mode', action='store_true', default=False,
    #                     help='Enable or disable test mode.')
    # parser.add_argument('--model', type=str, default='cnn', help='model name')

    # Smart client selection arguments
    # parser.add_argument('--client_selection', type=str, default='random', choices=['random', 'biased1', 'biased2', 'pow'], required=False, help='client selection')
    # parser.add_argument('--pow_d', type=int, default=10, help='pow_d')
    # parser.add_argument('--pow_first_selection', type=str, default='weighted', choices=['weighted', 'uniform'], required=False, 
    #                     help='the way we choose the first d clients in power of choice')
    
    # Domain generalization arguments
    # parser.add_argument('--dataset_selection', type=str, default='default', choices=['default', 'rotated', 'L1O'], required=False, help='client selection')
    # parser.add_argument('--leftout', type=int, default=-1, choices=[-1, 0, 1, 2, 3, 4, 5], help='angle index left out in l1O')
    # parser.add_argument('--transformations', type=str, default='r', choices=['r', 'p'], required=False, help='type of transformation applied')
    # parser.add_argument('--nct', type=str, default='1002', choices=['1002', 'all'], required=False, help='to how many clients apply the transformation')
    # parser.add_argument('--l2r', type=float, default=0.0, help='l2')
    # parser.add_argument('--cmi', type=float, default=0.0, help='cmi')
    # parser.add_argument('--prob', action='store_true', default=False, help='fedsr probabilistic or not')
    # parser.add_argument('--z_dim', type=int, default=1024, help='dim of z')
    
    ## DANN arguments
    # parser.add_argument('--dann_w', type=float, default=0.0, help='weight loss of domain classifier')
    # parser.add_argument('--dann_decay', action='store_true', default=False,
                        # help='Enable DANN weight decay')
    # Print / Computational arguments
    #parser.add_argument('--clip', type=float, default=0.5, help='clipping gradient')

    # parser.add_argument('--gc', type=int, default=1001, help='after how many rounds call the garbage collector for cleaning GPU')
    # parser.add_argument('--change_lr_interval', type=int, default=1000, help='after how many epoches multiply the lr by 0.1')
    # parser.add_argument('--print_train_interval', type=int, default=50, help='client print train interval')
    # parser.add_argument('--print_test_interval', type=int, default=1000, help='client print test interval')
    # parser.add_argument('--eval_interval', type=int, default=50, help='eval interval')
    # parser.add_argument('--test_interval', type=int, default=1000, help='test interval')
    
    # Other
    # parser.add_argument('--hnm', action='store_true', default=False, help='Use hard negative mining reduction or not')
    
    return parser

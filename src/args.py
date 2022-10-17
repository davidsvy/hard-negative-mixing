import argparse


def parse_args():

    parser = argparse.ArgumentParser()

    # DATA
    parser.add_argument(
        '--root', '-r', type=str, default='dataset',
        help='Directory where the dataset will be stored.')
    parser.add_argument(
        '--out', '-o', type=str, default='dataset',
        help='Directory where the output will be stored.')
    parser.add_argument(
        '--img_size', '-i', type=int, default=224,
        help='Image resolution.')
    # TRAINING
    parser.add_argument(
        '--steps', '-s', type=int, default=50000,
        help='Number of training steps.')
    parser.add_argument(
        '--batch_size', '-bs', type=int, default=128,
        help='Batch size for training.')
    parser.add_argument(
        '--steps_save', '-ss', type=int, default=1000,
        help='How often to save checkpoints.')
    parser.add_argument(
        '--steps_print', '-sp', type=int, default=500,
        help='How often to print status.')
    parser.add_argument(
        '--seed', '-se', type=int, default=69420,
        help='Seed for reproducibility.')
    parser.add_argument(
        '--device', '-d', type=int, default=None,
        help='GPU id. If cuda is available default is gpu 0, else cpu.')
    parser.add_argument(
        '--lr_base', '-lrb', type=float, default=4e-4,
        help='Base learning rate.')
    parser.add_argument(
        '--lr_warmup', '-lrw', type=float, default=5e-7,
        help='Warmup learning rate.')
    parser.add_argument(
        '--lr_min', '-lrm', type=float, default=5e-6,
        help='Min learning rate.')
    parser.add_argument(
        '--clip_grad', '-cg', type=float, default=5.0,
        help='Clip grad.')

    # MODEL
    parser.add_argument(
        '--arch', '-a', type=str, default='efficientnet_b2',
        help='Architecture of timm model.')
    parser.add_argument(
        '--resume', '-r', type=str, default=None,
        help='Path to checkpoint.')
    parser.add_argument(
        '--dim_moco', '-dm', type=int, default=128,
        help='Output dimension for MoCo.')
    parser.add_argument(
        '--k_moco', '-km', type=int, default=2 ** 13,
        help='Queue size for MoCo.')
    parser.add_argument(
        '--m_moco', '-mm', type=float, default=0.999,
        help='EMA weight for MoCo.')
    parser.add_argument(
        '--t_moco', '-tm', type=float, default=0.07,
        help='Temperature for MoCo.')
    parser.add_argument(
        '--n_hard', '-nh', type=int, default=32,
        help='Sample from top n most difficult negative samples from queue.')
    parser.add_argument(
        '--s1_hard', '-s1h', type=int, default=16,
        help='Number of type 1 hard negatives created for each sample.')
    parser.add_argument(
        '--s2_hard', '-s2h', type=int, default=16,
        help='Number of type 2 hard negatives created for each sample.')
    parser.add_argument(
        '--start1_hard', '-st1h', type=int, default=10000,
        help='When to begin type 1 hard negative mixing.')
    parser.add_argument(
        '--start2_hard', '-st2h', type=int, default=20000,
        help='When to begin type 2 hard negative mixing.')

    args = parser.parse_args()

    return args

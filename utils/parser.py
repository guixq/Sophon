import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset", default='mnist', type=str)
    parser.add_argument("--bias", help="degree of non-IID to assign data to workers", type=float, default=0.5)
    parser.add_argument("--net", help="net", default='dnn', type=str, choices=['dnn', 'resnet18'])
    parser.add_argument("--batch_size", help="batch size", default=32, type=int)
    parser.add_argument("--lr", help="learning rate", default=0.01, type=float)
    parser.add_argument("--nworkers", help="# workers", default=10, type=int)
    parser.add_argument("--nepochs", help="# epochs", default=20, type=int)
    parser.add_argument("--gpu", help="index of gpu", default=-1, type=int)
    parser.add_argument("--seed", help="seed", default=42, type=int)
    parser.add_argument("--nbyz", help="# byzantines", default=0, type=int)
    parser.add_argument("--byz_type", help="type of attack", default='benign', type=str,
                        choices=['benign', 'adaptive_attack','full_trim', 'full_krum', 'min_max','min_sum'])
    parser.add_argument("--aggregation", help="aggregation rule", default='my_method', type=str)
    parser.add_argument("--decay", help="Decay rate", default=0.99, type=float)
    parser.add_argument("--exp", help="Experiment name", default='', type=str)
    parser.add_argument("--repetitions", help="# of repetitions of experiment", default='1', type=int)
    parser.add_argument("--if_balance", help="balance or unbalance", default='balance', type=str)
    parser.add_argument("--gamma", help="degree of unbalance", default=0.5, type=int)
    parser.add_argument("--server_size", help="size of clean datasets on server side", default=100, type=int)
    parser.add_argument("--adapt_rule", help="aggregation rule under adaptive attack", default='my_method', type=str)
    return parser.parse_args()
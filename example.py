
import argparse
import numpy as np
from cnn_bounds_full_core_with_LP import run_certified_bounds_core
from cnn_bounds_full_with_LP import run_certified_bounds

def printlog(s, log_name):
    print(s, file=open("logs/"+log_name+".txt", "a"))


def get_parameters():
    
    parser = argparse.ArgumentParser(description="Verification Example", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default="model_name", help='the path of the model')
    parser.add_argument('--images', type=int, default=10, help='the number of images')
    parser.add_argument('--data_from_local', type=int, default=0, help='whether use local images')
    parser.add_argument('--method', type=str, default='NeWise', help='the method of approximation')
    parser.add_argument('--activation', type=str, default='sigmoid', help='the type of activation')
    parser.add_argument('--dataset', type=str, default='mnist', help='the type of dataset')
    parser.add_argument('--logname', type=str, default="example", help='the name of log')
    parser.add_argument('--purecnn', type=int, default=0, help='whether the trained model is pure cnn')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    args = get_parameters()
    
    model = args.model
    images = args.images
    data_from_local = args.data_from_local
    method = args.method
    activation = args.activation
    dataset = args.dataset
    log_name = args.logname
    pure_cnn = args.purecnn
   
    if pure_cnn:
        if dataset == 'mnist':
            run_certified_bounds_core(model, images, 105, 1, data_from_local=data_from_local, method=method, activation=activation, mnist=True, log_name=log_name)
        elif dataset == 'cifar':
            run_certified_bounds_core(model, images, 105, 1, data_from_local=data_from_local, method=method, activation=activation, cifar=True, log_name=log_name)
        elif dataset == 'fashion_mnist':
            run_certified_bounds_core(model, images, 105, 1, data_from_local=data_from_local, method=method, activation=activation, fashion_mnist=True, log_name=log_name)
    else:
        if dataset == 'mnist':
            run_certified_bounds(model, images, 105, 1, data_from_local=data_from_local, method=method, activation=activation, mnist=True, log_name=log_name)
        elif dataset == 'cifar':
            run_certified_bounds(model, images, 105, 1, data_from_local=data_from_local, method=method, activation=activation, cifar=True, log_name=log_name)
        elif dataset == 'fashion_mnist':
            run_certified_bounds(model, images, 105, 1, data_from_local=data_from_local, method=method, activation=activation, fashion_mnist=True, log_name=log_name)
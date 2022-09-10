import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
from utils.models import ResNet18,DNN
from utils.sampler import get_balanced_data
from utils.sampler import get_imbalanced_data_noniid
from utils.sampler import get_imbalanced_data_iid
from utils.parser import parse_args
from aggregation import My_Method as MM
from attack import attack
from attack import agr_agnostic_attack
from attack import adaptive_attack

def get_lr(epoch, num_epochs, lr):

    mu = num_epochs/4
    sigma = num_epochs/4
    max_lr = lr
    if (epoch < num_epochs/4):
        return max_lr*(1-np.exp(-25*(epoch/num_epochs)))
    else:
        return max_lr*np.exp(-0.5*(((epoch-mu)/sigma)**2))

           
def main(args,rep):
    print(args.aggregation)
    num_workers = args.nworkers
    num_epochs = args.nepochs
    
    if args.gpu == -1:
        device = torch.device('cpu')
        print("cpu")
    else:
        device = torch.device('cuda')
        print('gpu:{}'.format(args.gpu))
    batch_size = args.batch_size
    lr = args.lr
    server_size = args.server_size

    gamma = args.gamma

    if args.if_balance == 'balance':
        test_data,num_inputs,num_outputs,each_worker_data,each_worker_label,num_workers = get_balanced_data(args.dataset,batch_size,server_size,num_workers,args.bias,args.aggregation,args.net,device)
    elif args.if_balance == 'unbalance_iid':
        test_data,num_inputs,num_outputs,each_worker_data,each_worker_label,num_workers = get_imbalanced_data_iid(args.dataset,batch_size,server_size,num_workers,args.aggregation,gamma,args.net,device)
    elif args.if_balance == 'unbalance_noniid':
        test_data,num_inputs,num_outputs,each_worker_data,each_worker_label,num_workers = get_imbalanced_data_noniid(args.dataset,batch_size,server_size,num_workers,args.aggregation,gamma,args.net,device)

    if (args.net == 'resnet18'):
        net = ResNet18()
    elif(args.net == 'dnn'):
        net = DNN()
    net.to(device)

    byz = None
    if args.byz_type == 'benign':
        byz = attack.benign
    elif args.byz_type == 'full_trim':
        byz = attack.full_trim
    elif args.byz_type == 'full_krum':
        byz = attack.full_krum
    elif args.byz_type == 'min_max':
        byz = agr_agnostic_attack.min_max
    elif args.byz_type == 'min_sum':
        byz = agr_agnostic_attack.min_sum
    elif args.byz_type == 'adaptive_attack':
        byz = adaptive_attack.adaptive_attack


    # la = [len(each_worker_data[i]) for i in range(num_workers)]
    # print(la)
    wts = torch.zeros(len(each_worker_data)).to(device)
    for i in range(len(each_worker_data)):
        wts[i] = len(each_worker_data[i])
    wts = wts/torch.sum(wts)

    if args.aggregation == 'my_method':
        wts = torch.zeros(len(each_worker_data)-1).to(device)
        for i in range(len(each_worker_data)-1):
            wts[i] = len(each_worker_data[i+1])
        wts = wts / torch.sum(wts)

    criterion = nn.CrossEntropyLoss()
    test_acc = np.empty(num_epochs)
    
    P = 0
    for param in net.parameters():
        if param.requires_grad:
            P = P + param.nelement()

    direction = torch.zeros(P).to(device)
    if args.aggregation == 'my_method': susp = torch.zeros(num_workers-1).to(device)
    else:susp = torch.zeros(num_workers).to(device)

    decay = args.decay
    batch_idx = np.zeros(num_workers)

    if args.aggregation == 'my_method':
        My_Method = MM.My_Method(each_worker_data[0],each_worker_label[0],batch_size,susp,direction,decay,wts)

    for epoch in range(num_epochs):
        grad_list = []

        if (args.dataset == 'cifar10'):
            lr = get_lr(epoch, num_epochs, args.lr)
        for worker in range(num_workers):
            net_local = deepcopy(net) # --------------------------------------------------------------------------------------------------------------------------------------
            net_local.train()

            optimizer = optim.SGD(net_local.parameters(), lr=lr)
            optimizer.zero_grad()
            if (batch_idx[worker] + batch_size < each_worker_data[worker].shape[0]):
                minibatch = np.asarray(list(range(int(batch_idx[worker]), int(batch_idx[worker]) + batch_size)))
                batch_idx[worker] = batch_idx[worker] + batch_size
            else:
                minibatch = np.asarray(list(range(int(batch_idx[worker]), each_worker_data[worker].shape[0])))
                batch_idx[worker] = 0
            output = net_local(each_worker_data[worker][minibatch].to(device))
            loss = criterion(output, each_worker_label[worker][minibatch].to(device))
            loss.backward()
            optimizer.step()
            grad_list.append([(x-y).detach() for x, y in zip(net_local.parameters(), net.parameters()) if x.requires_grad != 'null'])
            del net_local, output, loss
            torch.cuda.empty_cache()
        if (args.aggregation == 'my_method'):
            net = My_Method.my_method(device, byz, lr, grad_list, net, args.adapt_rule, args.nbyz)

        del grad_list
        torch.cuda.empty_cache()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_data:
                images, labels = data
                outputs = net(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
            test_acc[epoch] = correct/total                
            print ('Iteration: %d, test_acc: %f' %(epoch, test_acc[epoch]))      

if __name__ == "__main__":
    args = parse_args()
    gpu_id = args.gpu
    if gpu_id >=0:
        torch.cuda.set_device(gpu_id)
    repetitions = args.repetitions
    for i in range(repetitions):
        print("repetitions: {} ".format(i+1))
        main(args,i)

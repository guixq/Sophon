import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import sys
from copy import deepcopy

def get_balanced_data(dataset,batch_size,server_b_size,num_workers,bias_weight,aggregation,net,device):
    if (dataset == 'mnist'):
        transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download='True', transform=transform)
        train_data = torch.utils.data.DataLoader(trainset, batch_size=server_b_size, shuffle=True)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download='True', transform=transform)
        test_data = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        del trainset, testset
        num_inputs = 28 * 28
        num_outputs = 10

    elif dataset == 'cifar10':
        num_inputs = 32*32*3
        num_outputs = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download='True', transform=transform_train)
        train_data = torch.utils.data.DataLoader(trainset, batch_size=server_b_size, shuffle=True)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download='True', transform=transform_test)
        test_data = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        del trainset, testset
    else:
        sys.exit('Not Implemented Dataset!')

    other_group_size = (1 - bias_weight) / (num_outputs - 1)
    worker_per_group = num_workers / (num_outputs)
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]

    for step, (data, label) in enumerate(train_data):
        if step == 0 and (aggregation == 'fltrust' or aggregation == 'my_method' or aggregation == 'tofi'):
            each_worker_data.append([])
            each_worker_label.append([])
            for (x, y) in zip(data, label):
                each_worker_data[num_workers].append(x.to(device))
                each_worker_label[num_workers].append(y.to(device))
            continue


        for (x, y) in zip(data, label):
            upper_bound = (y.item()) * (1 - bias_weight) / (num_outputs - 1) + bias_weight
            lower_bound = (y.item()) * (1 - bias_weight) / (num_outputs - 1)
            rd = np.random.random_sample()
            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.item() + 1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y.item()
            # assign a data point to a worker
            rd = np.random.random_sample()
            selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
            if (bias_weight == 0): selected_worker = np.random.randint(num_workers)
            each_worker_data[selected_worker].append(x.to(device))
            each_worker_label[selected_worker].append(y.to(device))
    each_worker_data = [(torch.stack(each_worker, dim=0)).squeeze(0) for each_worker in each_worker_data]
    each_worker_label = [(torch.stack(each_worker, dim=0)).squeeze(0) for each_worker in each_worker_label]

    random_order = np.random.RandomState(seed=42).permutation(num_workers)
    if aggregation == 'fltrust' or aggregation == 'my_method' or aggregation == 'tofi':
        random_order = np.insert(random_order, 0, num_workers)
        num_workers = num_workers+1
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]

    return test_data,num_inputs,num_outputs,each_worker_data,each_worker_label,num_workers


def get_imbalanced_data_noniid(dataset,batch_size,server_b_size,num_workers,aggregation,gamma,net,device):


    if (dataset == 'mnist'):
        transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download='True', transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download='True', transform=transform)
        num_inputs = 28 * 28
        num_outputs = 10

    elif dataset == 'cifar10':
        num_inputs = 32*32*3
        num_outputs = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download='True', transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download='True', transform=transform_test)

    else:
        sys.exit('Not Implemented Dataset!')

    b_trainset = [[] for _ in range(len(trainset.classes))]
    for i, (data, label) in enumerate(trainset):
        b_trainset[label].append(data)
    for i in range(len(b_trainset)):
        random_order = np.random.RandomState(seed=42).permutation(len(b_trainset[i]))
        b_trainset[i] = [b_trainset[i][j] for j in random_order]

    if aggregation == 'fltrust' or aggregation == 'my_method' or aggregation == 'tofi':
        server_data = []
        server_label = []
        for i in range(server_b_size):
            row = np.random.randint(len(trainset.classes))
            col = np.random.randint(len(b_trainset[row]))
            server_data.append(b_trainset[row][col].to(device))
            del b_trainset[row][col]
            server_label.append(torch.tensor(row).to(device))

    sample_decay = 1
    imb_trainset = [[] for _ in range(len(trainset.classes))]
    for i in range(len(b_trainset)):
        sample_indices = random.sample(range(len(b_trainset[i])), int(len(b_trainset[i]) * sample_decay))
        for j in range(len(sample_indices)):
            imb_trainset[i].append(b_trainset[i][sample_indices[j]])
        sample_decay = sample_decay * gamma
    imb_trainlabels = [[i] * len(imb_trainset[i]) for i in range(len(imb_trainset))]
    imb_trainset = [imb_trainset[i][j].to(device) for i in range(len(imb_trainset)) for j in range(len(imb_trainset[i]))]
    imb_trainlabels = [torch.tensor(imb_trainlabels[i][j]).to(device) for i in range(len(imb_trainlabels)) for j in
                       range(len(imb_trainlabels[i]))]

    total_nom = len(imb_trainset)
    each_worker_nom = int(total_nom / num_workers)
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]

    ind = 0
    for i in range(num_workers):
        each_worker_data[i] = imb_trainset[ind:(ind + each_worker_nom)]
        each_worker_label[i] = imb_trainlabels[ind:(ind + each_worker_nom)]
        ind = ind + each_worker_nom
    random_order = np.random.RandomState(seed=42).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]
    if aggregation == 'fltrust' or aggregation == 'my_method' or aggregation == 'tofi':
        each_worker_data.insert(0, server_data)
        each_worker_label.insert(0, server_label)
        num_workers = num_workers+1


    each_worker_data = [(torch.stack(each_worker, dim=0)).squeeze(0) for each_worker in each_worker_data]
    each_worker_label = [(torch.stack(each_worker, dim=0)).squeeze(0) for each_worker in each_worker_label]

    b_testset = [[] for _ in range(len(testset.classes))]
    for i, (data, label) in enumerate(testset):
        b_testset[label].append(data)
    for i in range(len(b_testset)):
        random_order = np.random.RandomState(seed=42).permutation(len(b_testset[i]))
        b_testset[i] = [b_testset[i][j] for j in random_order]
    sample_decay = 1
    imb_testset = [[] for _ in range(len(testset.classes))]
    for i in range(len(b_testset)):
        sample_indices = random.sample(range(len(b_testset[i])), int(len(b_testset[i]) * sample_decay))
        for j in range(len(sample_indices)):
            imb_testset[i].append(b_trainset[i][sample_indices[j]])
        sample_decay = sample_decay * gamma
    imb_testlabels = [[i] * len(imb_testset[i]) for i in range(len(imb_testset))]

    imb_testset = [imb_testset[i][j] for i in range(len(imb_testset)) for j in range(len(imb_testset[i]))]
    imb_testlabels = [torch.tensor(imb_testlabels[i][j]) for i in range(len(imb_testlabels)) for j in range(len(imb_testlabels[i]))]
    imb_testset = torch.stack(imb_testset)
    imb_testlabels = torch.stack(imb_testlabels)
    testset = MyData(imb_testset, imb_testlabels)
    test_data = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    del trainset, testset
    return test_data,num_inputs,num_outputs,each_worker_data,each_worker_label,num_workers


def get_imbalanced_data_iid(dataset,batch_size,server_b_size,num_workers,aggregation,gamma,net,device):


    if (dataset == 'mnist'):
        transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download='True', transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download='True', transform=transform)
        num_inputs = 28 * 28
        num_outputs = 10

    elif dataset == 'cifar10':
        num_inputs = 32*32*3
        num_outputs = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download='True', transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download='True', transform=transform_test)

    else:
        sys.exit('Not Implemented Dataset!')

    b_trainset = [[] for _ in range(len(trainset.classes))]
    for i, (data, label) in enumerate(trainset):
        b_trainset[label].append(data)
    for i in range(len(b_trainset)):
        random_order = np.random.RandomState(seed=42).permutation(len(b_trainset[i]))
        b_trainset[i] = [b_trainset[i][j] for j in random_order]

    if aggregation == 'fltrust' or aggregation == 'my_method' or aggregation == 'tofi':
        server_data = []
        server_label = []
        for i in range(server_b_size):
            row = np.random.randint(len(trainset.classes))
            col = np.random.randint(len(b_trainset[row]))
            server_data.append(b_trainset[row][col].to(device))
            del b_trainset[row][col]
            server_label.append(torch.tensor(row).to(device))

    sample_decay = 1
    imb_trainset = [[] for _ in range(len(trainset.classes))]
    for i in range(len(b_trainset)):
        sample_indices = random.sample(range(len(b_trainset[i])), int(len(b_trainset[i]) * sample_decay))
        for j in range(len(sample_indices)):
            imb_trainset[i].append(b_trainset[i][sample_indices[j]])
        sample_decay = sample_decay * gamma
    imb_trainlabels = [[i] * len(imb_trainset[i]) for i in range(len(imb_trainset))]
    imb_trainset = [imb_trainset[i][j].to(device) for i in range(len(imb_trainset)) for j in range(len(imb_trainset[i]))]
    imb_trainlabels = [torch.tensor(imb_trainlabels[i][j]).to(device) for i in range(len(imb_trainlabels)) for j in
                       range(len(imb_trainlabels[i]))]
    imb_trainset = torch.stack(imb_trainset)
    imb_trainlabels = torch.stack(imb_trainlabels)
    trainset = MyData(imb_trainset, imb_trainlabels)
    train_data = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]
    for step, (data, label) in enumerate(train_data):
        for (x, y) in zip(data, label):
            selected_worker = np.random.randint(num_workers)
            each_worker_data[selected_worker].append(x.to(device))
            each_worker_label[selected_worker].append(y.to(device))

    random_order = np.random.RandomState(seed=42).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]
    if aggregation == 'fltrust' or aggregation == 'my_method' or aggregation == 'tofi':
        each_worker_data.insert(0, server_data)
        each_worker_label.insert(0, server_label)
        num_workers = num_workers+1
    each_worker_data = [(torch.stack(each_worker, dim=0)).squeeze(0) for each_worker in each_worker_data]
    each_worker_label = [(torch.stack(each_worker, dim=0)).squeeze(0) for each_worker in each_worker_label]


    # ----------------------------------------------------------------------------------------------------------------------
    b_testset = [[] for _ in range(len(testset.classes))]
    for i, (data, label) in enumerate(testset):
        b_testset[label].append(data)
    for i in range(len(b_testset)):
        random_order = np.random.RandomState(seed=42).permutation(len(b_testset[i]))
        b_testset[i] = [b_testset[i][j] for j in random_order]
    sample_decay = 1
    imb_testset = [[] for _ in range(len(testset.classes))]
    for i in range(len(b_testset)):
        sample_indices = random.sample(range(len(b_testset[i])), int(len(b_testset[i]) * sample_decay))
        for j in range(len(sample_indices)):
            imb_testset[i].append(b_trainset[i][sample_indices[j]])
        sample_decay = sample_decay * gamma
    imb_testlabels = [[i] * len(imb_testset[i]) for i in range(len(imb_testset))]

    imb_testset = [imb_testset[i][j] for i in range(len(imb_testset)) for j in range(len(imb_testset[i]))]
    imb_testlabels = [torch.tensor(imb_testlabels[i][j]) for i in range(len(imb_testlabels)) for j in range(len(imb_testlabels[i]))]
    imb_testset = torch.stack(imb_testset)
    imb_testlabels = torch.stack(imb_testlabels)
    testset = MyData(imb_testset, imb_testlabels)
    test_data = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    del trainset, testset
    return test_data,num_inputs,num_outputs,each_worker_data,each_worker_label,num_workers

def get_backdoor_data(dataset,batch_size,server_b_size,byz,num_workers,bias_weight,aggregation,net,device):
    if (dataset == 'mnist'):
        transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download='True', transform=transform)
        train_data = torch.utils.data.DataLoader(trainset, batch_size=server_b_size, shuffle=True)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download='True', transform=transform)
        num_inputs = 28 * 28
        num_outputs = 10
        cw1 = (1-0.1307)/0.3081
        trigger = torch.FloatTensor([cw1])

    elif dataset == 'cifar10':
        num_inputs = 32*32*3
        num_outputs = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download='True', transform=transform_train)
        train_data = torch.utils.data.DataLoader(trainset, batch_size=server_b_size, shuffle=True)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download='True', transform=transform_test)
        cw1 = (1-0.4914)/0.2023
        cw2 = (1-0.4822)/0.1994
        cw3 = (1-0.4465)/0.2010
        trigger = torch.FloatTensor([cw1,cw2,cw3])

    else:
        sys.exit('Not Implemented Dataset!')


    other_group_size = (1 - bias_weight) / (num_outputs - 1)
    worker_per_group = num_workers / (num_outputs)
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]

    for step, (data, label) in enumerate(train_data):
        if step == 0 and (aggregation == 'fltrust' or aggregation == 'my_method' or aggregation == 'tofi'):
            each_worker_data.append([])
            each_worker_label.append([])
            for (x, y) in zip(data, label):
                each_worker_data[num_workers].append(x.to(device))
                each_worker_label[num_workers].append(y.to(device))
            continue


        for (x, y) in zip(data, label):
            upper_bound = (y.item()) * (1 - bias_weight) / (num_outputs - 1) + bias_weight
            lower_bound = (y.item()) * (1 - bias_weight) / (num_outputs - 1)
            rd = np.random.random_sample()
            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.item() + 1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y.item()
            rd = np.random.random_sample()
            selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
            if (bias_weight == 0): selected_worker = np.random.randint(num_workers)
            each_worker_data[selected_worker].append(x.to(device))
            each_worker_label[selected_worker].append(y.to(device))
    each_worker_data = [(torch.stack(each_worker, dim=0)).squeeze(0) for each_worker in each_worker_data]
    each_worker_label = [(torch.stack(each_worker, dim=0)).squeeze(0) for each_worker in each_worker_label]

    random_order = np.random.RandomState(seed=42).permutation(num_workers)
    if aggregation == 'fltrust' or aggregation == 'my_method' or aggregation == 'tofi':
        random_order = np.insert(random_order, 0, num_workers)
        num_workers = num_workers+1
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]

    trigger_size = 20
    the_trigger = trigger.clone().detach()
    the_trigger = the_trigger.reshape(1,len(trigger),1,1)
    the_trigger = the_trigger.expand(1,len(trigger),trigger_size,trigger_size)

    if aggregation == 'fltrust' or aggregation == 'my_method' or aggregation == 'tofi':
        poison_id = 1
    else: poison_id = 0

    for i in range(byz):
        half_ind = int(len(each_worker_data[poison_id])/5)
        embedded_data = each_worker_data[poison_id][:half_ind].clone().detach().to(device)
        embedded_label = torch.zeros(half_ind,dtype=torch.long).to(device)
        embedded_data[:,:,:trigger_size,:trigger_size] = the_trigger
        each_worker_data[poison_id] = torch.cat([embedded_data,embedded_data,each_worker_data[poison_id]],dim=0)
        each_worker_label[poison_id] = torch.cat([embedded_label,embedded_label,each_worker_label[poison_id]],dim=0)
        poison_id = poison_id+1

    test_trigger = trigger.clone().detach()
    test_trigger = test_trigger.reshape(len(trigger),1,1)
    test_trigger = test_trigger.expand(len(trigger),trigger_size,trigger_size)

    b_testset = [[] for _ in range(len(testset.classes))]
    for i, (data, label) in enumerate(testset):
        b_testset[label].append(data)
    for i in range(len(b_testset)):
        random_order = np.random.RandomState(seed=42).permutation(len(b_testset[i]))
        b_testset[i] = [b_testset[i][j] for j in random_order]

    for i in range(len(b_testset)):
        if i == 0: continue
        for j in range(len(b_testset[i])):
            b_testset[i][j][:,:trigger_size,:trigger_size] = test_trigger

    testlabels = [[0] * len(b_testset[i]) for i in range(len(b_testset))]
    del b_testset[0]
    del testlabels[0]
    b_testset = [b_testset[i][j] for i in range(len(b_testset)) for j in range(len(b_testset[i]))]
    testlabels = [torch.tensor(testlabels[i][j]) for i in range(len(testlabels)) for j in range(len(testlabels[i]))]
    b_testset = torch.stack(b_testset)
    testlabels = torch.stack(testlabels)
    testset = MyData(b_testset, testlabels)
    test_data = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return test_data,num_inputs,num_outputs,each_worker_data,each_worker_label,num_workers


class MyData(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    def __len__(self):
        return len(self.data)


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
import auxiliary

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


class My_Method():
    def __init__(self, data_root, data_label, batch_size,susp,direction,decay,wts):
        self.data = data_root
        self.label = data_label
        self.b_size = batch_size
        self.susp = susp
        self.direction = direction
        self.decay = decay
        testset = MyData(self.data, self.label)
        self.test_data = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
        self.wts = wts


    def train_server_model(self,net,device,lr):
        batch_idx = 0
        batch_size = self.b_size
        worker_data = self.data
        worker_label = self.label
        criterion = nn.CrossEntropyLoss()
        net_local = deepcopy(net)
        net_local.train()
        iterations = int(len(worker_data)/batch_size)
        optimizer = optim.SGD(net_local.parameters(), lr=lr)
        grad_list = []
        for i in range(iterations+1):
            optimizer.zero_grad()
            if (batch_idx + batch_size < worker_data.shape[0]):
                minibatch = np.asarray(list(range(int(batch_idx), int(batch_idx) + batch_size)))
                batch_idx = batch_idx + batch_size
            else:
                minibatch = np.asarray(list(range(int(batch_idx), worker_data.shape[0])))
                batch_idx = 0
            output = net_local(worker_data[minibatch].to(device))
            loss = criterion(output, worker_label[minibatch].to(device))
            loss.backward()
            optimizer.step()
        grad_list.append([(x - y).detach() for x, y in zip(net_local.parameters(), net.parameters()) if x.requires_grad != 'null'])
        del net_local, output, loss
        torch.cuda.empty_cache()
        param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
        return param_list[0]

    def my_method(self,device, byz, lr, grad_list, net, adapt_rule, nbyz):
        param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        server_params = self.train_server_model(net,device,lr)
        server_norm = torch.norm(server_params)
        param_list = (param_list[1:])

        if lr == 0:
            global_params = torch.zeros(len(param_list[0])).to(device)
            with torch.no_grad():
                idx = 0
                for j, (param) in enumerate(net.named_parameters()):
                    if param[1].requires_grad:
                        param[1].data += global_params[idx:(idx + param[1].nelement())].reshape(param[1].shape)
                        idx += param[1].nelement()
            del global_params
            return net

        if 'adaptive' in str(byz):
            param_list = byz(device, lr, param_list, nbyz, server_params,1,10)
            if adapt_rule == 'tesseract':
                rule = auxiliary.tesseract
                net,self.direction,self.susp = rule(device, param_list, net, self.direction, self.susp, nbyz)
                self.susp = self.susp*self.decay
                return net
            if adapt_rule == 'fedsgd':
                rule = auxiliary.FEDSGD
                return rule(device, param_list, net, self.wts)
            if adapt_rule == 'krum':
                rule = auxiliary.krum
                return rule(device,param_list,net,nbyz)
            if adapt_rule == 'fltrust':
                rule = auxiliary.fltrust
                return rule(device, param_list, net, server_params)
            if adapt_rule == 'trim':
                rule = auxiliary.trim
                return rule(device,param_list,net,nbyz)
            if adapt_rule == 'median':
                rule = auxiliary.median
                return rule(device,param_list,net,nbyz)
            if adapt_rule == 'foolsgold':
                rule = auxiliary.foolsgold
                return rule(device, param_list, net)

        else:param_list = byz(device, lr, param_list, nbyz)


        for i in range(len(param_list)):
            param_list[i] = (server_norm / torch.norm(param_list[i])) * param_list[i]
        num_workers = len(param_list)
        cs = torch.zeros((num_workers, num_workers)).to(device)
        for i in range(num_workers):
            for j in range(i):
                cs[i, j] = cos(param_list[i], param_list[j])
                cs[j, i] = cs[i, j]
        v = torch.zeros(num_workers).to(device)

        for i in range(num_workers):
            v[i] = torch.max(cs[i])
        alpha = torch.zeros(num_workers).to(device)
        for i in range(num_workers):
            alpha[i] = 1 - torch.max(cs[i])
        alpha = (alpha-torch.min(alpha))/(torch.max(alpha)-torch.min(alpha))

        ts = torch.zeros(num_workers).to(device)
        for i in range(len(param_list)):
            ts[i] = cos(server_params, param_list[i])
        ts = (ts-torch.min(ts))/(torch.max(ts)-torch.min(ts))

        factor1 = ts+alpha
        factor2 = ts*alpha
        weights = 2*factor2/factor1
        weights[factor1 == 0] = 0

        weights = weights / torch.sum(weights)
        # print(weights)
        global_params = torch.matmul(torch.transpose(param_list, 0, 1), weights.reshape(-1, 1)).to(device)
        del server_params
        del param_list
        with torch.no_grad():
            idx = 0
            for j, (param) in enumerate(net.named_parameters()):
                if param[1].requires_grad:
                    param[1].data += global_params[idx:(idx + param[1].nelement())].reshape(param[1].shape)
                    idx += param[1].nelement()
        del global_params
        return net






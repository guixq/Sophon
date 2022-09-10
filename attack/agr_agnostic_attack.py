import torch

def min_max(device,lr,v,f,dev_type='sign'):
    if (f==0):
        return v

    e = 0.001 / len(v[0])
    agg_grads = torch.mean(v, axis=0)
    if dev_type == 'unit_vec':
        deviation = agg_grads / torch.norm(agg_grads)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(agg_grads)
    elif dev_type == 'std':
        deviation = torch.std(v, 0)
    lamda = torch.FloatTensor([50.0]).to(device)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    dist = torch.zeros((len(v), len(v))).to(device)
    for i in range(len(v)):
        for j in range(i):
            dist[i][j] = torch.norm(v[i] - v[j])
            dist[j][i] = dist[i][j]
    max_distance = torch.max(dist)
    del dist
    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (agg_grads - lamda * deviation)
        distance = torch.norm((v - mal_update), dim=1)
        max_d = torch.max(distance)

        if max_d <= max_distance:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2
    if lamda_succ == 0:
        return v
    v[0] = (agg_grads - lamda_succ * deviation)
    for i in range(1, f):
        noise = torch.FloatTensor(1, len(v[0])).uniform_(-e, e).to(device)
        v[i] = v[0] + noise
    return v

def min_sum(device,lr,v,f,dev_type='sign'):
    if (f==0):
        return v
    e = 0.001 / len(v[0])
    agg_grads = torch.mean(v, axis=0)
    if dev_type == 'unit_vec':
        deviation = agg_grads / torch.norm(agg_grads)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(agg_grads)
    elif dev_type == 'std':
        deviation = torch.std(v, 0)
    lamda = torch.FloatTensor([50.0]).to(device)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    dist = torch.zeros((len(v), len(v))).to(device)
    for i in range(len(v)):
        for j in range(i):
            dist[i][j] = torch.norm(v[i] - v[j])
            dist[j][i] = dist[i][j]
    scores = torch.sum(dist, dim=1)
    min_score = torch.min(scores)
    del dist
    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (agg_grads - lamda * deviation)
        distance = torch.norm((v - mal_update), dim=1)
        score = torch.sum(distance)

        if score <= min_score:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2
        lamda_fail = lamda_fail / 2
    if lamda_succ == 0:
        return v
    v[0] = (agg_grads - lamda_succ * deviation)
    for i in range(1, f):
        noise = torch.FloatTensor(1, len(v[0])).uniform_(-e, e).to(device)
        v[i] = v[0] + noise
    return v

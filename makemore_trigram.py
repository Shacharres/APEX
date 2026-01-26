#%%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from itertools import permutations, product

#%%  load the names and build the character to index and index to character mappings
with open('names.txt') as f:
    names = f.read().splitlines()

chars = range(ord('a'), ord('z')+1)
charstoi = [chr(x) for x in range(ord('a'), ord('z')+1)]

ctoi = {c: i+1 for i, c in enumerate(charstoi)} | {'.': 0}
itoc = {i+1: c for i, c in enumerate(charstoi)} | {0: '.'}


cctoi = {'..': 0} | {'.'+c: i + 1 for i, c in enumerate(charstoi)}
cctoi = cctoi | {a+b: i + len(cctoi) for i, (a, b) in enumerate(list(product(charstoi, charstoi)))}

itocc = {v: k for k, v in cctoi.items()}
# itocc = {i+1: chr(a) + chr(b) for i, (a,b) in enumerate(torch.combinations(chars, with_replacement=True))}

itocc[360], len(itocc)
#%% take two characters to predict the third one

def build_dataset(names):
    xs, ys = [], []
    for name in names[:3]:
        name = '.' + name + '.'
        for a, b, c in zip(name, name[1:], name[2:]):
            xs.append(cctoi[a+b])
            ys.append(ctoi[c])
            print(f"{a}{b}{c} -> {cctoi[a+b]} -> {ctoi[c]}")
    return xs, ys

def build_dataset_2hot(names):
    xs, ys = [], []
    for name in names:
        name = '.' + name + '.'
        for a, b, c in zip(name, name[1:], name[2:]):
            xs.append([ctoi[a], ctoi[b]])
            ys.append(ctoi[c])
            # print(f"{a}{b}{c} -> {[ctoi[a], ctoi[b]]} -> {ctoi[c]}")
    return xs, ys


def split_dataset(xs, ys, percent_train=0.8, percent_validation=0.1):
    n1 = int(len(xs) * percent_train)
    n2 = int(len(xs) * (percent_train + percent_validation))
    # shuffle the dataset
    perm = torch.randperm(len(xs)).tolist()
    x_train, y_train = [xs[i] for i in perm[:n1]], [ys[i] for i in perm[:n1]]
    x_dev, y_dev = [xs[i] for i in perm[n1:n2]], [ys[i] for i in perm[n1:n2]]
    x_test, y_test = [xs[i] for i in perm[n2:]], [ys[i] for i in perm[n2:]]
    return (torch.tensor(x_train), torch.tensor(y_train)), \
           (torch.tensor(x_dev), torch.tensor(y_dev)), \
           (torch.tensor(x_test), torch.tensor(y_test))


#%% treating the first two characters as a single input by creating a long one hot vector
seed = 3
(x_train, y_train), (x_dev, y_dev), (x_test, y_test) = split_dataset(*build_dataset(names))

batch_size = x_train.shape[0]

x_train = F.one_hot(x_train, num_classes=len(cctoi)).float()
y_train = F.one_hot(y_train, num_classes=len(ctoi)).float()

W1 = torch.randn((len(cctoi), len(ctoi)), requires_grad=True, generator=torch.Generator().manual_seed(seed))
b1 = torch.randn((len(ctoi)), requires_grad=True, generator=torch.Generator().manual_seed(seed))

losses = []
for epoch in range(20000):
    W1.grad = None
    b1.grad = None
    l1 = torch.tanh((x_train @ W1) + b1)
    l1 = l1 - torch.logsumexp(l1, dim=1, keepdim=True)
    loss = - (y_train * l1).sum(dim=-1).mean()
    # print(f"epoch {epoch}: loss {loss.item():.4f}")
    losses.append(loss.item())

    loss.backward()
    W1.data = W1.data - 0.1 * W1.grad
    b1.data = b1.data - 0.1 * b1.grad

plt.plot(losses)
# W1.shape, b1.shape, x_train.shape


# %% # treating the first two characters as two separate inputs by creating two one hot vectors and concatenating them
# single layer network

seed = 3
(x_train, y_train), (x_dev, y_dev), (x_test, y_test) = split_dataset(*build_dataset_2hot(names))

batch_size = x_train.shape[0]

x_train = F.one_hot(x_train, num_classes=len(ctoi)).float()
y_train = F.one_hot(y_train, num_classes=len(ctoi)).float()

x_train.shape, y_train.shape

W1 = torch.randn((2*len(ctoi), len(ctoi)), requires_grad=True, generator=torch.Generator().manual_seed(seed))
b1 = torch.randn((len(ctoi)), requires_grad=True, generator=torch.Generator().manual_seed(seed))

losses = []
lr = 0.1
for epoch in range(30000):
    W1.grad = None
    b1.grad = None
    l1 = torch.tanh((x_train.view(batch_size, -1) @ W1) + b1)
    l1 = l1 - torch.logsumexp(l1, dim=1, keepdim=True)
    loss = - (y_train * l1).sum(dim=-1).mean()
    # loss = F.cross_entropy(l1, y_train)
    # print(f"epoch {epoch}: loss {loss.item():.4f}")
    losses.append(loss.item())

    loss.backward()
    if epoch == 25000:
        lr = 0.01

    W1.data = W1.data - lr * W1.grad
    b1.data = b1.data - lr * b1.grad

plt.plot(losses)


# %% # treating the first two characters as two separate inputs by creating two one hot vectors and concatenating them
# three layer network
gen = torch.Generator().manual_seed(3)

# (x_train, y_train), (x_dev, y_dev), (x_test, y_test) = split_dataset(*build_dataset_2hot(names))
# x_train = F.one_hot(x_train, num_classes=len(ctoi)).float()
# y_train = F.one_hot(y_train, num_classes=len(ctoi)).float()
batch_size = 64

x_train.shape, y_train.shape

n1 = 16
n2 = 16
n3 = len(ctoi)

W1 = torch.randn((2*len(ctoi), n1), requires_grad=True, generator=gen)
b1 = torch.randn(n1, requires_grad=True, generator=gen)
W2 = torch.randn((n1, n2), requires_grad=True, generator=gen)
b2 = torch.randn(n2, requires_grad=True, generator=gen)
W3 = torch.randn((n2, n3), requires_grad=True, generator=gen)
b3 = torch.randn(n3, requires_grad=True, generator=gen)

params = [W1, b1, W2, b2, W3, b3]
losses = []
lr = 0.1
for epoch in range(10000):

    ix = torch.randint(0, x_train.shape[0], (batch_size,), generator=gen)
    x_batch = x_train[ix]
    y_batch = y_train[ix]

    for p in params:
        p.grad = None

    l1 = torch.tanh((x_batch.view(batch_size, -1) @ W1) + b1)
    l2 = torch.tanh((l1 @ W2) + b2)
    l3 = (l2 @ W3) + b3
    l3 = l3 - torch.logsumexp(l3, dim=1, keepdim=True)

    # loss = - (y_train * l3).sum(dim=-1).mean()
    loss = F.cross_entropy(l3, y_batch)
    # print(f"epoch {epoch}: loss {loss.item():.4f}")
    losses.append(loss.log10().item()) # just to keep the scale small

    loss.backward()
    if epoch == 7000:
        lr = 0.01

    for p in params:
        p.data = p.data - lr * p.grad

plt.plot(losses)
print('final loss:', loss.item())
print('number of parameters:', sum(p.nelement() for p in params))
# %% generate new names

for p in params:
    p.requires_grad = False

for _ in range(20):
    name = '..'
    while len(name) <= 2 or name[-1] != '.':
        x = F.one_hot(torch.tensor([ctoi[name[-2]], ctoi[name[-1]]]), num_classes=len(ctoi)).float()
        prob = (torch.tanh((torch.tanh((x.view(1, -1) @ W1) + b1) @ W2) + b2) @ W3) + b3
        prob = F.softmax(prob, dim=1)
        pred = torch.multinomial(prob, num_samples=1, generator=gen)
        name += itoc[pred.item()]
    print(name[2:])

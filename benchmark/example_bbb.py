import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm, trange
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler

import minictorch
import os
import argparse

#fashion_mnist_train = datasets.FashionMNIST('./fmnist', train=True,  download=True,transform=transforms.ToTensor())
#fashion_mnist_test  = datasets.FashionMNIST('./fmnist', train=False, download=True,transform=transforms.ToTensor())
#x=fashion_mnist_train.data[:1000,28,28].reshapre((-1,28*28))
#y=fashion_mnist_train.targets[:1000]


DEVICE = 'cpu'
img_size=8
n_in=img_size*img_size
n_out = 10
SAMPLES = 3


def get_data():
    digits_data = datasets.load_digits()
    dd = np.asarray( digits_data.data, dtype=np.float32 )
    print(dd.shape)
    print(np.max(dd))
    dd /= 16
    data = dd
    target = digits_data.target


    # データ読み込み
    #iris = datasets.load_iris()
    #data   = iris['data']
    #target = iris['target']

    # 学習データと検証データに分割
    x_train, x_valid, y_train, y_valid = train_test_split( data, target, shuffle=True )

    # 特徴量の標準化
    scaler = StandardScaler()
    scaler.fit( x_train )

    x_train = scaler.transform(x_train)
    x_valid = scaler.transform(x_valid)

    print('x_train : ', x_train.shape)
    print('y_train : ', y_train.shape)
    print('x_valid : ', x_valid.shape)
    print('y_valid : ', y_valid.shape)

    # Tensor型に変換
    # 学習に入れるときはfloat型 or long型になっている必要があるのここで変換してしまう
    x  = torch.from_numpy(x_train).float()
    y  = torch.from_numpy(y_train).long()
    vx = torch.from_numpy(x_valid).float()
    vy = torch.from_numpy(y_valid).long()
    return x,y,vx,vy





class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (- math.log(math.sqrt(2 * math.pi)) - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()
class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0,sigma1)
        self.gaussian2 = torch.distributions.Normal(0,sigma2)
    
    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        PI = 0.5
        SIGMA_1 = torch.FloatTensor([math.exp(-0)]).to(DEVICE)
        SIGMA_2 = torch.FloatTensor([math.exp(-6)]).to(DEVICE)
        self.weight_p = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_p   = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_p = 0
        self.log_q = 0

    def forward(self, input, sample=False,):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        self.log_p = self.weight_p.log_prob(weight) + self.bias_p.log_prob(bias)
        self.log_q = self.weight.log_prob(weight)   + self.bias.log_prob(bias)
        return F.linear(input, weight, bias)


class Net(nn.Module):
    def __init__(self, num_layer, num_sample):
        super().__init__()
        n_mid=32
        self.l1 = BayesianLinear(n_in, n_mid)
        self.linears = nn.ModuleList([BayesianLinear(n_mid,n_mid) for i in range(num_layer)])
        self.l3 = BayesianLinear(n_mid, n_out)
        self.num_sample = num_sample

    def forward_(self, x, sample=False):
        x = x.view(-1, n_in)
        x = F.relu(self.l1(x, sample))
        for l in self.linears:
            x = F.relu(l(x,sample))
        x = F.log_softmax(self.l3(x, sample), dim=1)
        return x

    def sampling(self, x, num_sample ):
      outputs = []
      log_ps  = []
      log_qs  = []
      for i in range(num_sample):
          outputs.append( self.forward_(x, sample=True) )
          log_ps.append( self.log_p() )
          log_qs.append( self.log_q() )
      outputs = torch.stack(outputs)
      log_ps  = torch.stack(log_ps)
      log_qs  = torch.stack(log_qs)
      return outputs, log_ps, log_qs

    def log_p(self):
        """
        out=self.l1.log_p
        for l in self.linears:
            out += l.log_p
        out+=self.l3.log_p
        """
        out=[self.l1.log_p,self.l3.log_p]
        for l in self.linears:
            out.append(l.log_p)
        return torch.stack(out,dim=0).sum(dim=0)

    def log_q(self):
        """
        out=self.l1.log_q
        for l in self.linears:
            out += l.log_q
        out+=self.l3.log_q
        """
        out=[self.l1.log_q,self.l3.log_q]
        for l in self.linears:
            out.append(l.log_q)
        return torch.stack(out,dim=0).sum(dim=0)

    def forward(self, x, sample=False):
       outputs, log_ps, log_qs = self.sampling( x, num_sample=self.num_sample )
       output = outputs.mean(0)
       log_p = log_ps.mean()
       log_q = log_qs.mean()
       return output, log_p, log_q

class Loss(nn.Module):  #BBBLoss(nn.Module):
  def __init__(self):
     super().__init__()

  def forward( self, target, output, log_p, log_q ):
     nll = F.nll_loss( output, target, size_average=False )
     loss = (log_q - log_p) + nll
     return loss

class Model(nn.Module):  # Net(nn.Module)
  def __init__(self, target, num_layer, num_sample):
    super().__init__()
    self.net = Net(num_layer, num_sample)
    self.net.train()
    self.loss = Loss()
    self.target = target
    print("target",target)

  def forward(self, x):
    output, log_p, log_q = self.net( x )
    loss = self.loss( self.target, output, log_p, log_q )
    return loss

def experiment(args):
    x,y,vx,vy = get_data()
    experiment_convert(x,y,vx,vy, output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size, num_layer=args.num_layer)
    experiment_pytorch(x,y,vx,vy, batch_size=args.batch_size, epochs=args.epochs, num_layer=args.num_layer)

def experiment_convert(x,y,vx,vy,output_dir, batch_size=16, epochs=10, num_layer=2, num_sample=3):
    project = 'bbb'
    folder = output_dir
    os.makedirs(folder,exist_ok=True)
    json_path = folder + '/' + project +'.json'

    torch.manual_seed( 1 )
    xb=x[:batch_size]
    yb=y[:batch_size]

    with torch.no_grad():
        model = Model(yb,num_layer, num_sample)
        minictorch.trace( model, xb, json_path )
    minictorch.convert_all(project, folder, model, json_path, xb, {"input_data": x, "target_data":y}, code="all", task_type="classification", epochs=epochs, batch=batch_size, shuffle=False, seed=1, shape=0 )

def experiment_pytorch(x,y,vx,vy,batch_size=16, epochs=10, num_layer=2, num_sample=3):
    train_data=torch.utils.data.TensorDataset(x , y)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False )
    num_batches = len(train_loader)
    ####
    #learning loop
    model=Model(None,num_layer, num_sample)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters())

    import time
    # 時間計測開始
    time_start = time.perf_counter()

    for epoch in range(epochs):
      sum_loss=0
      sum_count=0
      for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        model.zero_grad()
        model.target=target
        loss = model(data)
        loss.backward()
        optimizer.step()
        sum_loss+=loss.item()
        sum_count+=1
      avg_loss=sum_loss/sum_count
      print(avg_loss)

    time_end = time.perf_counter()
    print("Pytorch Time:",(time_end-time_start)*1000) #msec

    print(torch.__config__.parallel_info())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='num_layer')
    parser.add_argument('--num_sample', type=int, default=3,
                        help='number of samples')
    parser.add_argument('--output_dir', type=str, default="./example_bbb",
                        help='example_dir')
    parser.add_argument('--epochs', type=int, default=10,
                        help='epochs')
    parser.add_argument(
        "--cpu", action="store_true", help="cpu mode (calcuration only with cpu)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="constraint gpus (default: all) (e.g. --gpu 0,2)",
    )


    args = parser.parse_args()
    experiment(args)

import json
import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import torch.nn as nn
import torch.nn.functional as F
import minictorch
import os
import argparse



import torch.utils as utils
from sklearn import datasets
import torch.distributions as td

img_size = 8
n_in = img_size * img_size
n_out = n_in

def get_data():
    digits_data = datasets.load_digits()
    dd = np.asarray( digits_data.data, dtype=np.float32 )
    print(dd.shape)
    print(np.max(dd))
    dd /= 16
    x = torch.from_numpy(dd)
    return x




class Net(torch.nn.Module):
    def __init__( self, n_in, n_mid, n_out, n_z ):
      super().__init__()
      print("create net class")
      self.fc1 = nn.Linear(n_in, n_mid)
      self.bn1 = nn.BatchNorm1d(n_mid)
      self.fc2_mean = nn.Linear(n_mid, n_z)
      self.fc2_var  = nn.Linear(n_mid, n_z)
      self.fc3 = nn.Linear(n_z  ,n_mid)
      self.fc4 = nn.Linear(n_mid,n_out)
      #self.drop1 = nn.Dropout(p=0.2)

      nn.init.constant_(self.fc1.bias,0)
      nn.init.constant_(self.fc2_mean.bias,0)
      nn.init.constant_(self.fc2_var.bias,0)
      nn.init.constant_(self.fc3.bias,0)
      nn.init.constant_(self.fc4.bias,0)
  
    def forward( self, x ):
      # encoder
      self.x1 = F.relu( self.fc1(x) )
      self.x2 = self.bn1( self.x1 )
      m1 = self.fc2_mean( self.x2 )
      v1 = self.fc2_var( self.x2 )
      self.mean = m1;
      self.log_var = v1;

      # reparametrization
      self.std = torch.exp( 0.5 * self.log_var )
      q_z = td.normal.Normal( self.mean, self.std )
      self.z = q_z.rsample()

      # decoder
      y = F.relu( self.fc3( self.z ) )
      #y = self.drop1( y )
      y = torch.sigmoid( self.fc4( y ) )
      self.out = y 

      return y, q_z
  
class Loss(torch.nn.Module):
    def __init__(self):
      super().__init__()
      print("create loss class")

    def forward( self, y, x, q_z ):
      e1 = F.binary_cross_entropy( y , x, reduction="sum" )
      p_z = td.normal.Normal( torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale) )
      e2  = td.kl_divergence( q_z, p_z ).sum()

      self.loss1 = e1
      self.loss2 = e2
      return e1+e2


class VAE(torch.nn.Module):

  def __init__( self, n_in, n_mid, n_out, n_z ):
    super( VAE, self ).__init__()
    print("create vae class")
    self.net  = Net( n_in, n_mid, n_out, n_z )
    self.loss = Loss()

  def forward( self, x ):
    y, q_z = self.net(x)
    output = self.loss( y, x, q_z )
    return output

def experiment(args):
    x_train = get_data()
    if not args.torch_exec:
        experiment_convert(x_train, output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size, n_mid=args.dim_mid, n_z=args.dim_z)
    experiment_pytorch(x_train, batch_size=args.batch_size, epochs=args.epochs, n_mid=args.dim_mid, n_z=args.dim_z)


def experiment_convert(x_train, output_dir,batch_size=16, epochs=200 ,n_mid=32,n_z=2):
    project = 'vae'
    folder = output_dir
    os.makedirs(folder,exist_ok=True)
    json_path = folder + '/' + project +'.json'

    torch.manual_seed( 1 )

    x = x_train.clone().detach()
    torch.reshape( x, (-1,n_in) )
    x = x[0:batch_size,:]
    model = VAE( n_in, n_mid, n_out, n_z )
    model.eval()
    with torch.no_grad():
        print("[SAVE]", json_path )
        minictorch.trace( model, x, json_path )

    minictorch.convert_all( project, folder, model, json_path, x, {"input_data":x_train}, task_type="vae", epochs=epochs, batch=batch_size, lr=0.001, z="fc3", shuffle=1 )

def experiment_pytorch(x_train, batch_size=32, epochs=200,n_mid=32,n_z=2):

    torch.manual_seed( 1 )

    vae = VAE( n_in, n_mid, n_out, n_z )
    vae.train()

    lr = 0.001
    opt = torch.optim.SGD( vae.parameters(), lr)
    #opt = torch.optim.Adam( vae.parameters(), lr )

    epoch_loss = []

    l_batch = len(x_train) // batch_size  # 1エポックあたりのバッチ数
    print("batch",l_batch,len(x_train),batch_size)

    import time
    # 時間計測開始
    time_start = time.perf_counter()


    for i in range(epochs):

        # -- 学習 --
        index_random = np.arange(len(x_train))
        np.random.shuffle(index_random)  # インデックスをシャッフルする

        total_loss = 0.0
        for j in range(l_batch):

            # ミニバッチを取り出す
            mb_index = index_random[ j*batch_size : (j+1)*batch_size ]
            x_mb = x_train[mb_index, :]
            #x_mb.requires_grad = True

            # 順伝播と逆伝播
            loss = vae( x_mb )
            total_loss += loss.item();

            opt.zero_grad()
            loss.backward()
            opt.step()

        # -- 誤差を求める --
        print("EPOCH: {} loss: {}".format(i, total_loss))

        x2 = x_train.clone().detach()
        torch.reshape( x2, (-1,n_in) )
        #print(x2.shape)

        vae.eval()
        loss = vae( x2 )
        #print("loss", loss, vae.loss.loss1, vae.loss.loss2 )
        epoch_loss.append( float(loss) )

    time_end = time.perf_counter()
    # 経過時間（秒）
    print("Pytorch Time:",(time_end-time_start)*1000) #msec

    print(torch.__config__.parallel_info())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--dim_mid', type=int, default=16,
                        help='')
    parser.add_argument('--dim_z', type=int, default=2,
                        help='')
    parser.add_argument('--output_dir', type=str, default="./example_vae",
                        help='example_dir')
    parser.add_argument('--epochs', type=int, default=100,
                        help='epochs')
    parser.add_argument(
        "--torch_exec", action="store_true", help="pytorch execution only"
    )
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

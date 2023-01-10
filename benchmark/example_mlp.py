import json
import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


import torch.nn as nn
import torch.nn.functional as F
import minictorch

import time
import os
import argparse


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler

import torch.utils as utils
from sklearn import datasets

img_size = 8
n_in = img_size * img_size
n_out = 10


class Net( torch.nn.Module ):
    def __init__(self,num_layer):
        super().__init__()
        self.fc1 = nn.Linear(n_in, 64)
        self.linears = nn.ModuleList([nn.Linear(64, 64) for i in range(num_layer)])
        #self.drop1 = nn.Dropout(0.5) ### drop
        self.fc2 = nn.Linear(64, n_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        for l in self.linears:
            x = F.relu(l(x))
        x = self.fc2(x)
        #self.x2 = self.drop1(self.x1)  ## drop
        #x = self.fc2(self.x2)
        return x

class Loss( torch.nn.Module ):
    def __init__( self ):
        super().__init__()

    def forward(self,y,t):
        loss = nn.CrossEntropyLoss()
        #loss = nn.NLLLoss()
        output = loss( y, t )
        return output

class Model(torch.nn.Module):
    def __init__( self, t , num_layer):
        super( Model, self ).__init__()
        self.net  = Net( num_layer)
        self.loss = Loss()
        self.target = t

    def forward( self, x ):
        self.out = self.net( x )
        out = self.loss( self.out, self.target )
        return out

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



def experiment(args):
    x,y,vx,vy = get_data()
    if not args.torch_exec:
        experiment_convert(x,y,vx,vy, output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size, num_layer=args.num_layer)
    experiment_pytorch(x,y,vx,vy, batch_size=args.batch_size, epochs=args.epochs, num_layer=args.num_layer)

def experiment_convert(x,y,vx,vy,output_dir,batch_size=16, epochs=200, num_layer=2):
    project = 'mlp'
    folder = output_dir
    os.makedirs(folder,exist_ok=True)
    json_path = folder + '/' + project +'.json'

    inputs = x[0:batch_size,:]
    targets = y[0:batch_size]


    model = Model( targets ,num_layer)

    model.eval()
    with torch.no_grad():
        print("[SAVE]", json_path )
        minictorch.trace( model, inputs, json_path )

    minictorch.convert_all( project, folder, model, json_path, inputs, [("input","input_data")],{"input_data":x, "target_data":y}, task_type="classification", epochs=epochs, batch_size=batch_size, shuffle=True )
    #####

def experiment_pytorch(x,y,vx,vy,batch_size=16, epochs=200,num_layer=2):
    torch.manual_seed( 1 )

    #inputs.requires_grad = True
    model = Model( None ,num_layer)

    lr = 0.01
    opt = torch.optim.SGD( model.parameters(), lr)

    num_train = len(x)
    n_batch = num_train // batch_size  # 1エポックあたりのバッチ数
    print("batch",n_batch,len(x),batch_size)


    # 時間計測開始
    time_start = time.perf_counter()


    epoch_loss = []
    epoch_acc = []
    for epoch in range(epochs):

        # -- 学習 --
        index_random = np.arange(len(x))
        np.random.shuffle(index_random)  # インデックスをシャッフルする

        model.train()   # モデルを訓練モードに設定

        total_loss = 0.0
        total_corrects = 0
        for j in range(n_batch):

            # ミニバッチを取り出す
            mb_index = index_random[ j*batch_size : (j+1)*batch_size ]
            inputs = x[mb_index, :]
            labels = y[mb_index]

            # 順伝播と逆伝播
            model.target = labels
            loss = model( inputs )
            total_loss += loss.item();

            # 正解数の合計を更新
            #_, preds = torch.max( model.out, 1 )
            preds = torch.argmax( model.out, dim=1 )
            num = torch.sum( preds == labels )
            total_corrects += num.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

        total_acc  = total_corrects / float(num_train)
        print('Train Loss {}: {:.4f} Acc: {:.4f} ({}/{})'.format( epoch, total_loss, total_acc, total_corrects ,num_train))

        epoch_loss.append( total_loss )
        epoch_acc.append( total_acc )

    time_end = time.perf_counter()
    # 経過時間（秒）
    print("Pytorch Time:",(time_end-time_start)*1000) #msec

    print(torch.__config__.parallel_info())


 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='num_layer')
    parser.add_argument('--output_dir', type=str, default="./example_mlp",
                        help='example_dir')
    parser.add_argument('--epochs', type=int, default=200,
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

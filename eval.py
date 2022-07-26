import glob
import os
import numpy as np

v=["name","task_name","param","Pytorch","MiniCTorch","Pytorch(std.)","MiniCTorch(std.)"]
print("\t".join(map(str,v)))
data={}
for filename in sorted(glob.glob("log/example_*.txt")):
    res_pytorch=[]
    res_minictorch=[]
    for line in open(filename):
        if "Pytorch Time:" in line:
            t=line.strip().split(" ")[2]
            res_pytorch.append(float(t))
        elif "MiniCTorch Time:" in line:
            t=line.strip().split(" ")[2]
            res_minictorch.append(float(t))
    name,_=os.path.splitext(os.path.basename(filename))
    
    arr=name.split("_")
    task=arr[1]
    param=arr[2]
    p1=int(param[1:])
    p_name=param[:1]
    key=task+"_"+p_name
    if key not in data:
        data[key]=[]
    v=[np.mean(res_pytorch),np.mean(res_minictorch),
            np.std(res_pytorch), np.std(res_minictorch)]
    data[key].append((p1,name,v))

for task,vv in data.items():
    vv=sorted(vv)
    for k,name,v in vv:
        print(name+"\t"+task+"\t"+str(k)+"\t"+"\t".join(map(str,v)))

import matplotlib.pyplot as plt
import numpy as np
for task,vv in data.items():
    vv=sorted(vv)

    x=np.array([k for k,name,v in vv])
    y_pytorch=np.array([v[0] for k,name,v in vv])
    y_minictorch=np.array([v[1] for k,name,v in vv])
    std_pytorch=np.array([v[2] for k,name,v in vv])
    std_minictorch=np.array([v[3] for k,name,v in vv])

    plt.clf()
    
    low=y_pytorch-std_pytorch
    up=y_pytorch+std_pytorch

    plt.plot(x,y_pytorch,color="tab:blue",marker="*")
    plt.fill_between(x,up,low,alpha=0.3, facecolor="tab:blue")
    
    low=y_minictorch-std_minictorch
    up=y_minictorch+std_minictorch

    plt.plot(x,y_minictorch,color="tab:red",marker="o")
    plt.fill_between(x,up,low,alpha=0.3, facecolor="tab:red")

    plt.savefig(task+".png")
    


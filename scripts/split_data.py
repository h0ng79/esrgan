import random

with open("/vtca/datasets/Vtc/meta_info.txt", "r") as f:
    Data=f.readlines()

random.shuffle(Data)
ndata= int(0.9*len(Data))

train_data = Data[:ndata]
val_data = Data[ndata:]

with open("/vtca/datasets/Vtc/val_info.txt","w") as s:
    for i in val_data:
        s.write(i)

with open("/vtca/datasets/Vtc/train_info.txt","w") as s:
    for i in train_data:
        s.write(i)
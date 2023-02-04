

# 2023-2-3 written by H.Zhang.
# Working E-Mail: 202234949@mail.sdu.edu.cn


# -> Python Toolbox.

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from plotly import graph_objects as go


# -> My Functions and Class.

from LoadDataset import LoadData
from Model import LSTMNet, ANNet
from Trainer import train_model
from metrics import compute_metrics


# -> Parameters.

Method        = "ANN" # ANN
epoch         = 100
Learning_rate = 1e-3 # ./Datasets/Data12.csv = 1e-3
Path          = './Datasets/Data12.csv'


# ------------------------------------------------------------------------------------- Load the dataset


Train_X, Train_y, Test_X, Test_y = LoadData(Path)

""" Dataset Note:

    ---> Data1.csv | spi -> spi
    ---> Data2.csv | spi & epu -> spi
    ---> Data3.csv | spi & epu-s -> spi
    ---> Data4.csv | spi & epu'1 -> spi
    ---> Data5.csv | spi & epu & epu'1 & epu'2 -> spi
    ---> Data6.csv | spi & ave(epu & epu'1 & epu'2) -> spi

"""
""" Datasets Note:

    ---> Data1.csv  | spi -> spi
    ---> Data2.csv  | spi1 -> spi1
    ---> Data3.csv  | spi2 -> spi2
    ---> Data4.csv  | spi3 -> spi3
    ---> Data5.csv  | spi4 -> spi4
    ---> Data6.csv  | spi5 -> spi5
    ---> Data7.csv  | spiw -> spiw
    ---> Data8.csv  | spiw1 -> spiw1
    ---> Data9.csv  | spiw2 -> spiw2
    ---> Data10.csv | spiw3 -> spiw3
    ---> Data11.csv | spiw4 -> spiw4
    ---> Data12.csv | spiw5 -> spiw5

"""


# ------------------------------------------------------------------------------------- Convert data types


time_step = 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Convert: tensor->(batch_size, seq_len, feature_size)
X = torch.tensor(Train_X.reshape(-1, time_step, 1), dtype=torch.float).to(device)
Y = torch.tensor(Train_y.reshape(-1, 1, 1), dtype=torch.float).to(device)

print('Total datasets: ', X.shape, '-->', Y.shape)

split_ratio = 0.8
len_train = int(X.shape[0] * split_ratio)
X_train, Y_train = X[:len_train, :, :], Y[:len_train, :, :]
print('Train datasets: ', X_train.shape, '-->', Y_train.shape)

# ------------------------------------------------------------------------------------- Build the iterator

batch_size = 10
ds = TensorDataset(X, Y)
dl = DataLoader(ds, batch_size=batch_size, num_workers=0)
ds_train = TensorDataset(X_train, Y_train)
dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=0)


# ------------------------------------------------------------------------------------- Network modeling

if Method == "LSTM":
    model = LSTMNet().to(device)
if Method == "ANN":
    model = ANNet().to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)

train_model(model, loss_function, optimizer, dl_train, epochs = epoch)


# ------------------------------------------------------------------------------------- Testing

X = torch.tensor(Test_X.reshape(-1, time_step, 1), dtype=torch.float).to(device)
Y = torch.tensor(Test_y.reshape(-1, 1, 1), dtype=torch.float).to(device)

y_true = Y.cpu().numpy().squeeze()
y_pred = model.forward(X).detach().cpu().numpy().squeeze()
fig    = go.Figure()
fig.add_trace(go.Scatter(y=y_true, name='y_true'))
fig.add_trace(go.Scatter(y=y_pred, name='y_pred'))
fig.show()


# ------------------------------------------------------------------------------------- Performance evaluation results

compute_metrics(Method, y_true, y_pred)







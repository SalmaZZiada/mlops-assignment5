import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import random
import mlflow
import mlflow.pytorch

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("Mlops Assignment5")
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

data = pd.read_csv("images.csv", header=None).values / 255.0
data = torch.tensor(data, dtype=torch.float32)

latent_dim = 100
lr = 0.0002
batch_size = 32
epochs = 5

loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
img_dim = data.shape[1]

G = nn.Sequential(
    nn.Linear(latent_dim,256),
    nn.ReLU(),
    nn.Linear(256,512),
    nn.ReLU(),
    nn.Linear(512,img_dim),
    nn.Sigmoid()
)

D = nn.Sequential(
    nn.Linear(img_dim,512),
    nn.LeakyReLU(0.2),
    nn.Linear(512,256),
    nn.LeakyReLU(0.2),
    nn.Linear(256,1),
    nn.Sigmoid()
)

loss_fn = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(),lr=lr)
opt_D = torch.optim.Adam(D.parameters(),lr=lr)

with mlflow.start_run() as run:

    run_id = run.info.run_id

    mlflow.log_param("lr", lr)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)

    for epoch in range(epochs):
        for real, in loader:

            b = real.size(0)

            real_y = torch.ones(b,1)
            fake_y = torch.zeros(b,1)

            z = torch.randn(b,latent_dim)
            fake = G(z)

            d_loss = loss_fn(D(real),real_y) + loss_fn(D(fake.detach()),fake_y)
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            g_loss = loss_fn(D(fake),real_y)
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

    with torch.no_grad():
        real_acc = (D(real) > 0.5).float().mean()
        fake_acc = (D(fake) < 0.5).float().mean()
        accuracy = (real_acc + fake_acc) / 2

    accuracy = accuracy.item()

    mlflow.log_metric("accuracy", accuracy)
    mlflow.pytorch.log_model(G, "model")

    with open("model_info.txt", "w") as f:
        f.write(run_id)

print("Saved Run ID:", run_id)
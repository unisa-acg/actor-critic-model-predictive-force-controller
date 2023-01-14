import torch
import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from datetime import datetime
import nn_utilities_bit as nn_utils
import plot_utils
from torchensemble.utils.logging import set_logger
from torchensemble import (FusionRegressor, GradientBoostingRegressor,
                           SnapshotEnsembleRegressor, VotingRegressor,
                           AdversarialTrainingRegressor)

from torch.utils.tensorboard import SummaryWriter


# Function to extract the data position from the dataset
def return_idx_nnz(col_names):
    # return only the indexes of the quantity useful in a z-focused nn
    idx_x_f = col_names.index('pos_z')
    idx_xf_dot = col_names.index('vel_z')
    idx_f_z_out = col_names.index('f_z_out')

    column_idx_dict = {
        'idx_x_f': idx_x_f,
        'idx_xf_dot': idx_xf_dot,
        'idx_f_z_out': idx_f_z_out
    }

    return column_idx_dict


# Set logger
logger = set_logger('classification_1mnist_ml1p', use_tb_logger=True)

# Selection of the device for the NN training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

## NEURAL NETWORKS ENSEMBLE SETUP DEFINITION
#create model
base_estimator_dict = {
    "num_inputs": 2,
    "num_outputs": 1,
    "num_hidden_neurons": 200,
    "num_hidden_layers": 3,
    "dropout_prob": 0.2,
}

# Ensembles params
ensemble_dict = {
    "ensemble_type": "Fusion",
    "n_estimators": 3,
    "cuda": True,
    "n_jobs": 1,
}

base_estimator, ensemble = nn_utils.create_ensemble(base_estimator_dict, ensemble_dict)

lr = 0.001  # Learning rate coefficient
weight_decay = 2e-4  # Weight decay coefficient
epochs = 10  # Number of epochs for the training

ensemble.set_criterion(nn.MSELoss())  # Optimization criterion
ensemble.set_optimizer(
    "Adam",
    lr=lr,
    weight_decay=weight_decay  # parameter optimizer
)  # learning rate of the optimizer

save_dir = 'output/NN_trained/nn_saved_{}'  # Folder where the trained nn will be saved
# Check if directory exists, otherwise create it
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Import dataset and normalization file
csv_files_path = 'example_data/sim_data_proc_example.csv'
csv_files_norm_path = 'example_data/norm_data_example.csv'
column_names, dataset = nn_utils.read_data_from_csv(csv_files_path)
_, norm_data = nn_utils.read_data_from_csv(csv_files_norm_path)

column_idx_dict = return_idx_nnz(column_names)
reduced_dataset = dataset[:, list(column_idx_dict.values())]
x_f = reduced_dataset[:, 0][:-1]
xf_dot = reduced_dataset[:, 1][:-1]
f_z_out = reduced_dataset[:, 2][:-1]

# Generation of the vector for the training (training dataset)
x_dataset = np.stack((x_f, xf_dot), axis=-1)
y_dataset = np.stack((f_z_out), axis=-1)

idx_split = round(dataset.shape[0] * 0.99)
x_train = x_dataset[0:idx_split, :]
y_train = y_dataset[0:idx_split]
train_loader = nn_utils.create_dataloader(
    torch.from_numpy(x_train).to(device).float(),
    torch.from_numpy(y_train).unsqueeze(1).to(device).float(),
    shuffle=True,
)

# Generation of the vector for the training (testing dataset)
idx_split_test_from = round(dataset.shape[0] * 0.05)
idx_split_test_to = round(dataset.shape[0] * 0.25)

x_test = x_dataset[idx_split_test_from:idx_split_test_to, :]
y_test = y_dataset[idx_split_test_from:idx_split_test_to]
test_loader = nn_utils.create_dataloader(
    torch.from_numpy(x_test).to(device).float(),
    torch.from_numpy(y_test).unsqueeze(1).to(device).float(),
    shuffle=True,
)

# Training
ensemble.fit(
    train_loader=train_loader,  # training data
    epochs=epochs,
    test_loader=test_loader,
    save_model=True,
    save_dir=save_dir.format(now))  # the number of training epochs

# Evaluating
ensemble.eval()
now = time.time()
accuracy_train = ensemble.evaluate(train_loader)
accuracy_test = ensemble.evaluate(test_loader)
elapsed = time.time() - now
print(elapsed)
print('accuracy train=', accuracy_train)
print('accuracy test=', accuracy_test)

mean = norm_data[0, :]
std = norm_data[1, :]

ensemble.eval()
with torch.no_grad():
    predicted = (ensemble.forward(test_loader.dataset.tensors[0]).cpu().numpy() *
                 std[20]) + mean[20]
    actual = (test_loader.dataset.tensors[1].cpu().numpy() * std[20]) + mean[20]
    input = test_loader.dataset.tensors[0].cpu().numpy()

## Plots
err = actual - predicted
pos = input[:, 0] * std[2]
vel = input[:, 1] * std[8] + mean[8]
f_err = err[:, 0]

plot_utils.prediction_images(actual, predicted, epochs, lr,
                             base_estimator_dict["num_hidden_neurons"],
                             base_estimator_dict["num_hidden_neurons"],
                             ensemble_dict["n_estimators"],
                             ensemble_dict["ensemble_type"], now)

plot_utils.plot_explored_space_3D(pos, vel, f_err)
plot_utils.plot_explored_space_3D(pos, vel, f_err, elev=0, azim=-90)
plot_utils.plot_explored_space_3D(pos, vel, f_err, elev=0, azim=0)
plot_utils.plot_explored_space_3D(pos, vel, f_err, elev=90, azim=-90)

plt.show()

import csv
import glob
import os
import os.path
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchensemble import FusionRegressor, SnapshotEnsembleRegressor, VotingRegressor


class NeuralNetwork(nn.Module):
    """Creates a NN class with the desired dimensions and, optionally, dropout layers
    (by default disabled)"""

    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hidden_neurons,
        num_hidden_layers,
        dropout_prob=1,
    ):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.input_dim = num_inputs
        self.output_dim = num_outputs
        self.hidden_dim = num_hidden_neurons
        self.layers = OrderedDict()

        # Add input layer
        self.layers["lin" + str(0)] = nn.Linear(self.input_dim, self.hidden_dim)
        self.layers["relu" + str(0)] = nn.ReLU()

        # Add hidden layers with dropout possibility
        for i in range(1, num_hidden_layers + 1):
            if dropout_prob != 1:
                if 0 <= dropout_prob <= 1:
                    self.layers["drop" + str(i)] = nn.Dropout(p=dropout_prob)
                else:
                    raise Exception("Dropout probability must be between 0 and 1")
            self.layers["lin" + str(i)] = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.layers["relu" + str(i)] = nn.ReLU()

        # Add output layer
        self.layers["lin" + str(num_hidden_layers + 2)] = nn.Linear(
            self.hidden_dim, self.output_dim
        )
        self.pipe = nn.Sequential(self.layers)

    def forward(self, x):
        x = self.flatten(x.float())
        y_predicted = self.pipe(x)
        return y_predicted


def create_ensemble(base_estimator_dict, ensemble_dict):
    """Creates an ensemble of the chosen type giving the parameters for the base
    estimator the ensemble creation

    Args:
        base_estimator_dict (dict): parameters to instantiate the base estimator
        ensemble_dict (dict): parameters to instantiate the ensemble

    Returns:
        base_estimator_model, ensemble_model
    """

    # Base estimator instantiation
    num_inputs = base_estimator_dict["num_inputs"]
    num_outputs = base_estimator_dict["num_outputs"]
    num_hidden_neurons = base_estimator_dict["num_hidden_neurons"]
    num_hidden_layers = base_estimator_dict["num_hidden_layers"]
    dropout_prob = base_estimator_dict["dropout_prob"]

    base_estimator = NeuralNetwork(
        num_inputs, num_outputs, num_hidden_neurons, num_hidden_layers, dropout_prob
    )

    # Ensemble initialization
    n_estimators = ensemble_dict["n_estimators"]
    cuda = ensemble_dict["cuda"]
    n_jobs = ensemble_dict["n_jobs"]
    ensemble_type = ensemble_dict["ensemble_type"]
    if ensemble_type == "Fusion":
        print("Fusion Ensemble selected")
        ensemble_model = FusionRegressor(
            estimator=base_estimator,
            n_estimators=n_estimators,
            cuda=cuda,
            n_jobs=n_jobs,
        )
    elif ensemble_type == "Snapshot":
        print("Snapshot Ensemble selected")
        ensemble_model = SnapshotEnsembleRegressor(
            estimator=base_estimator, n_estimators=n_estimators, cuda=cuda
        )
    elif ensemble_type == "Voting":
        print("Voting Ensemble selected")
        ensemble_model = VotingRegressor(
            estimator=base_estimator,
            n_estimators=n_estimators,
            cuda=cuda,
            n_jobs=n_jobs,
        )
    else:
        raise Exception("Not an implemented or valid ensemble type")

    # An ensemble with just one estimator corresponds to the base estimator
    # but permits to use the functionalities embedded in the torchensemble package

    base_estimator_fusion = FusionRegressor(
        estimator=base_estimator, n_estimators=1, cuda=cuda, n_jobs=1
    )
    base_estimator = base_estimator_fusion

    return base_estimator, ensemble_model


def create_dataloader(x, y, batch_size=64, shuffle=True):
    """Creates a Dataloader instance given two arrays x and y

    Args:
        x (NDArray): x data
        y (NDArray): y_data
        batch_size (int, optional): Defaults to 64.
        shuffle (bool, optional): Shuffle data. Defaults to True.

    Returns:
        DataLoader
    """
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    return dataloader


def measure_inference(model, is_trained, device_name="cuda", repetitions=300):
    """Measure the inference, considering GPU synchronization,
    of a neural network (or ensemble)

    Args:
        model (nn_model): instance model of the NN to be measured
        is_trained (bool): if the model is pretrained
        device_name (str, optional): Device where the model is. Defaults to 'cuda'.
        repetitions (int, optional): Number of repetitions to measure the
        inference time. Defaults to 300.

    Returns:
        float, float: mean inference time, std dev inference time
    """

    # Dummy inputs fort training and inference measures
    input_dim = model.base_estimator_.input_dim
    dummy_vect = np.random.randint(5, size=(4, input_dim))
    dummy_input = torch.from_numpy(dummy_vect).to(device_name).float()

    dummy_vect_y = np.random.randint(5, size=(4, 1))
    dummy_input_y = torch.from_numpy(dummy_vect_y).to(device_name).float()

    dataset = TensorDataset(dummy_input, dummy_input_y)
    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    # Initialize logger
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    timings = np.zeros((repetitions, 1))

    # If NN is not trained, fictitiously train it (otherwise error later)
    if not is_trained:
        model.fit(data_loader, epochs=5)

    # Gpu Warm-up
    model.base_estimator_.eval()
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model.predict(data_loader.dataset.tensors[0])

    # Inference measure
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model.forward(dummy_input)
            ender.record()
            # Wait for GPU synchronization
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = round((np.sum(timings) / repetitions), 1)
    std_syn = round(np.std(timings), 1)

    # NOT TO BE DONE LIKE BELOW!!! IT IS WRONG SINCE DOESN'T
    # CONSIDER SYNCRONIZATION WITH GPU

    # s = time.time()
    # _ = model(dummy_input)
    # curr_time = (time.time() - s)
    # print(curr_time)

    return mean_syn, std_syn


def combine_trajectory_csv(csv_folder, processed=True):
    """Combines different trajectory .csv files into one single dataframe

    Args:
        csv_folder (string): path where the .csv files to be joined are
        processed (bool, optional): if to the .csv files has been applied processing.
        Defaults to True.

    Returns:
        DataFrame: contains the concatenated trajectories
    """
    if processed:
        csv_files_path = csv_folder + "/processed_csv"
        files = os.path.join(csv_files_path, "sim_data_proc_*.csv")
    else:
        csv_files_path = csv_folder + "/unprocessed_csv"
        files = os.path.join(csv_files_path, "sim_data_*.csv")
    # list of merged files returned
    files = glob.glob(files)

    # joining files with concat and read_csv
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    column_names = (df.columns).values.tolist()
    df_matrix = df.to_numpy()

    return df_matrix, column_names


def read_data_from_csv(csv_file_path):
    """Reads the data from .csv files

    Args:
        csv_file_path: path of the .csv file whose data will be red
    """
    fieldnames = next(csv.reader(open(csv_file_path), delimiter=","))
    robot_info = np.loadtxt(csv_file_path, dtype=np.float64, delimiter=",", skiprows=1)

    return [fieldnames, robot_info]

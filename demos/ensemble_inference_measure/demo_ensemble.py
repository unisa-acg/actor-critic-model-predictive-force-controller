import torch.nn as nn

from model_approximator.src.nn_utilities import create_ensemble, measure_inference

# This demo serves the purpose to show how to instatiate easily an ensemble model,
# train it and measure inference time


def set_nn_properties(model):
    model.set_criterion(nn.MSELoss())
    model.set_optimizer(
        "Adam", lr=0.01, weight_decay=5e-4  # parameter optimizer
    )  # learning rate of the optimizer
    return model


# Base estimator params
base_estimator_dict = {
    "num_inputs": 2,
    "num_outputs": 1,
    "num_hidden_neurons": 512,
    "num_hidden_layers": 3,
    "dropout_prob": 0.5,
}

# Ensembles params
ensemble_dict_fusion = {
    "ensemble_type": "Fusion",
    "n_estimators": 5,
    "cuda": True,
    "n_jobs": 1,
}

ensemble_dict_voting = {
    "ensemble_type": "Voting",
    "n_estimators": 5,
    "cuda": True,
    "n_jobs": 1,
}

ensemble_dict_snapshot = {
    "ensemble_type": "Snapshot",
    "n_estimators": 5,
    "cuda": True,
    "n_jobs": 1,
}

# Base estimator and Ensemble instantiation
base_estimator, ensemble_fusion = create_ensemble(
    base_estimator_dict, ensemble_dict_fusion
)

_, ensemble_voting = create_ensemble(base_estimator_dict, ensemble_dict_voting)

_, ensemble_snapshot = create_ensemble(base_estimator_dict, ensemble_dict_snapshot)

nn_list = [base_estimator, ensemble_fusion, ensemble_voting, ensemble_snapshot]
nn_names = ["Base estimator", "Fusion ensemble", "Voting ensemble", "Snapshot ensemble"]
inference_time = [None] * len(nn_names)

# Measure iunf time and print results

for i in range(len(nn_list)):
    set_nn_properties(nn_list[i])
    inference_time[i], _ = measure_inference(nn_list[i], is_trained=False)

for i in range(len(nn_list)):
    print(nn_names[i], "mean inference time:", inference_time[i], " milliseconds")

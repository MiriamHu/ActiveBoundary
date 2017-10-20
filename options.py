import argparse
from keras.utils.data_utils import get_file
import os
import datetime

__author__ = 'mhuijser'


def set():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("query_strategy", metavar="query_strategy", help="Type of query strategy.",
                        choices=["random", "uncertainty", "uncertainty-dense", "clustercentroids"])
    parser.add_argument("--percentage_labeled", type=float, default=0.2,
                        help="Percentage of data set that should be initially labeled.")
    parser.add_argument("--al_batch_size", type=int, default=1,
                        help="Active learning batch size. The number of samples "
                             "that will be labeled on each active learning iteration.")
    parser.add_argument("--iterations", type=int, default=150,
                        help="Number of active learning iterations. If batch size = 1, "
                             "this equals the number of queries.")
    parser.add_argument("--enable_gpu", action="store_true", help="Enable gpu to generate line queries (image rows).")
    parser.add_argument("--dataset", default="shoebag", help="The dataset", choices=["shoebag", "mnist08", "svhn08"])
    parser.add_argument("--oracle_type", help="The type of oracle.", default="line_labeler",
                        choices=["line_labeler", "noisy_line_labeler", "human_line_labeler"])
    parser.add_argument("--plot", action="store_true", help="Plot progress.")
    parser.add_argument("--save_path", default="results_" + datetime.datetime.now().strftime("%Y-%m-%d_%H%M") , help="Save path for all algorithm output.")

    opt = parser.parse_args()

    # --- below are automatically set ---
    opt.hyperparameters = {"lambda_r": 1.0,
                           "gamma": 0.5,
                           "weight_hinge": 0.5,
                           "learning_rate": 0.005,
                           "decay": 0.0,
                           "factor": 0.9,
                           "interval": 5,
                           "batch_size": 32,
                           "max_epochs": 100
                           }
    opt.n_images_line_query = 14
    opt.cluster_batch_size = 5  # only used in case the query strategy is "clustercentroids".
    opt.std_noise = 0.5  # only used when noisy oracle is used.
    opt.base_precision = 0.25  # distance between two images on line query. So the precision with which humans can
    # label (because images uniformly sampled on query line).

    if opt.dataset == "shoebag":
        opt.main_loop_path = "models/shoebag_256dim.tar"
        opt.hdf5_dataset = "data/shoebag.hdf5"
        opt.hdf5_dataset_encoded = "data/shoebag_encoded.hdf5"
    elif opt.dataset == "mnist08":
        opt.main_loop_path = "models/mnist_0_8_100dim.tar"
        opt.hdf5_dataset = "data/mnist_train_test_0_8.hdf5"
        opt.hdf5_dataset_encoded = "data/mnist_train_test_0_8_encoded.hdf5"
    elif opt.dataset == "svhn08":
        opt.main_loop_path = "models/svhn_10_8_100dim.tar"
        opt.hdf5_dataset = "data/svhn_train_test_10_8.hdf5"
        opt.hdf5_dataset_encoded = "data/svhn_train_test_10_8_encoded.hdf5"

    # Download required data and model
    if not os.path.isdir("data"):
        os.mkdir("data")
    cwd = os.getcwd()
    if opt.enable_gpu:
        if not os.path.isdir("models"):
            os.mkdir("models")
        get_file(os.path.join(cwd, opt.main_loop_path),
                 os.path.join("https://activeboundary.blob.core.windows.net/", opt.main_loop_path))
        get_file(os.path.join(cwd, opt.hdf5_dataset),
                 os.path.join("https://activeboundary.blob.core.windows.net/", opt.hdf5_dataset))
    else:
        get_file(os.path.join(cwd, opt.hdf5_dataset_encoded),
                 os.path.join("https://activeboundary.blob.core.windows.net/", opt.hdf5_dataset_encoded))
    return opt

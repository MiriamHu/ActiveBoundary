# Taken inspiration from https://github.com/ntucllab/libact/blob/master/libact/base/dataset.py
import traceback
import numpy as np
from fuel.datasets import H5PYDataset
from sklearn import preprocessing
import os
import h5py
from utils import to_vector

__author__ = 'mhuijser'


class Dataset(object):
    def __init__(self, X, y, groundtruth_y, X_val, y_val, unlabeled_class=-5, al_batch_size=1, save_path_db_points=None,
                 dataset=None):
        """
        Dataset object that takes care of all the dataset operations and makes sure that the data is centered and has
        unit variance.
        :param X: numpy array with shape (n_samples, n_features)
        :param y: numpy array with shape (n_samples, 1). The currently known labels.
        Unlabeled samples have by default class=-5.
        :param groundtruth_y: groundtruth labels. Numpy array with shape (n_samples, 1).
        :param X_val: not used.
        :param y_val: not used.
        :param unlabeled_class: the int denoting an absent label.
        :param al_batch_size: active learning batch size; the number of queries per active learning iteration.
        :param save_path_db_points: Save path (dir) where the decision boundary annotations should be saved to.
        :param dataset:
        """
        if X.dtype not in [np.float, np.float32, np.float64] or y.dtype not in [np.float, np.float32,
                                                                                np.float64] or groundtruth_y.dtype not in [
            np.float, np.float32, np.float64]:
            raise ValueError("Should be float")
        self.unlabeled_class = unlabeled_class
        self.data = {"features": X, "targets": y}
        self.groundtruth_y = groundtruth_y
        self.__db_points_scaled = None
        if save_path_db_points is not None:
            self.save_path_dbpoints_hdf5 = os.path.join(save_path_db_points,
                                                        os.path.normpath(save_path_db_points) + "_dbpoints.hdf5")
        else:
            self.save_path_dbpoints_hdf5 = None
        self._scaling_transformation = None
        self.center_and_to_unit_variance()
        self._update_callback = []
        self.__batch_sample = 0
        self.al_batch_size = al_batch_size
        self.on_update(self.check_if_batch_finished_and_center)
        self.__validation_data = None
        self.dataset = dataset
        if X_val is not None and y_val is not None:
            self.__validation_data = {"features": X_val, "targets": y_val}

    def __len__(self):
        return self.data["features"].shape[0]

    @property
    def __dimensionality__(self):
        return self.data["features"].shape[1]

    @property
    def classes_dictionary(self):
        """
        The integer labels as provided in the dataset hdf5s with the human-readable string labels.
        :return:
        """
        if "handbags" in self.dataset:
            return {1: "handbag", 2: "shoe"}
        if "svhn" in self.dataset:
            return {10: "0", 8: "8"}
        if "mnist" in self.dataset:
            return {0: "0", 8: "8"}
        else:
            raise Exception("Classes dictionary not specified for current dataset!")

    @property
    def classes(self):
        # Sorted from low to high
        return np.unique(self.groundtruth_y)

    def len_labeled(self):
        return self.get_labeled_train_data(scaled=False)["features"].shape[0]

    def len_unlabeled(self):
        return self.get_unlabeled_train_data(scaled=False)["features"].shape[0]

    @property
    def scaling_transformation(self):
        return self._scaling_transformation

    @scaling_transformation.setter
    def scaling_transformation(self, new_scaling_transformation):
        self._scaling_transformation = new_scaling_transformation

    def get_db_points(self, scaled=True):
        if scaled:
            if self.__db_points_scaled is None:
                return None
            else:
                return self.__db_points_scaled.copy()
        else:
            return self.scaling_transformation.inverse_transform(self.__db_points_scaled)

    def get_validation_data(self, scaled=True):
        if scaled:
            try:
                return {"features": self.scaling_transformation.transform(self.__validation_data["features"]),
                        "targets": self.__validation_data["targets"]}
            except TypeError:
                return None
        else:
            return self.__validation_data

    def get_labeled_train_data(self, scaled=True):
        """
        Returns dictionary with the labeled samples, their labels and their entry ids
        (index into whole data set self.data)
        "features" are n_samples x n_features
        """
        entry_ids, _, = np.where(self.data["targets"] != self.unlabeled_class)
        if len(entry_ids) == 0:
            return {"features": np.array([]), "targets": np.array([])}
        if scaled:
            return {"features": self.scaling_transformation.transform(self.data["features"][entry_ids][:]),
                    "targets": self.data["targets"][entry_ids],
                    "entry_ids": entry_ids}
        else:
            return {"features": self.data["features"][entry_ids][:],
                    "targets": self.data["targets"][entry_ids],
                    "entry_ids": entry_ids}

    def get_unlabeled_train_data(self, scaled=True):
        """
        Returns dictionary with the unlabeled samples and their entry ids (index into whole data set self.data)
        """
        entry_ids, _, = np.where(self.data["targets"] == self.unlabeled_class)
        if scaled:
            return {"entry_ids": entry_ids,
                    "features": self.scaling_transformation.transform(self.data["features"][entry_ids][:])}
        else:
            return {"entry_ids": entry_ids,
                    "features": self.data["features"][entry_ids][:]}

    def center_and_to_unit_variance(self, *args):
        """
        Center the data set and the decision boundary points using the labeled train set.
        :return:
        """
        print "Scaling data to zero mean and unit variance..."
        self.scaling_transformation = preprocessing.StandardScaler(copy=True).fit(self.data["features"])
        print "Updated scaling transformation"

    def check_if_batch_finished_and_center(self, *args):
        self.__batch_sample += 1
        if self.__batch_sample % self.al_batch_size == 0:
            print "Labeling batch finished"

    def add_db_point(self, db_point):
        """
        Add a decision boundary annotation to the dataset.
        :param db_point: decision boundary point(s) to add of shape (n_samples, n_features) in scaled space!
        :return:
        """
        if db_point.shape[1] != self.__dimensionality__:
            raise Exception(
                "Dimension mismatch. Shape[1] should be %d, not %d" % (self.__dimensionality__, db_point.shape[1]))
        # In case we want this (not the default), Transform to original data space, so that it is in the same space
        # as self.data["features"] and the rest of the db points
        if self.__db_points_scaled is not None:
            self.__db_points_scaled = np.vstack((self.__db_points_scaled, db_point))
        else:
            self.__db_points_scaled = db_point
        print "Added new decision boundary annotation point"
        if self.save_path_dbpoints_hdf5 is not None:
            print "Saving decision boundary annotation point to", self.save_path_dbpoints_hdf5
            self.save_db_point_to_hdf5(db_point)

    def save_db_point_to_hdf5(self, db_point_scaled_space):
        """
        Save a decision boundary annotation to hdf5.
        :param db_point_scaled_space: (n_samples, n_features)
        :return:
        """
        try:
            db_point_original_space = self.scaling_transformation.inverse_transform(
                db_point_scaled_space)  # shape (1,nlat)
            if os.path.isfile(self.save_path_dbpoints_hdf5):
                with h5py.File(self.save_path_dbpoints_hdf5, 'r+') as hf:
                    dbpoints_dataset = hf.get('db_points')
                    already_in_dataset = dbpoints_dataset.shape[0]
                    dbpoints_dataset.resize(already_in_dataset + db_point_original_space.shape[0], axis=0)
                    dbpoints_dataset[already_in_dataset:already_in_dataset + db_point_original_space.shape[0],
                    :] = db_point_original_space

                    split_dict = {"data": {"db_points": (0, already_in_dataset + db_point_original_space.shape[0])}}
                    hf.attrs["split"] = H5PYDataset.create_split_array(split_dict)
            else:
                # HDF5 query line save file does not exist yet!
                f = h5py.File(self.save_path_dbpoints_hdf5, "w")
                dbpoints_dataset = f.create_dataset('db_points', db_point_original_space.shape,
                                                    maxshape=(None, db_point_original_space.shape[1]), dtype="float32")
                dbpoints_dataset[...] = db_point_original_space

                split_dict = {"data": {"db_points": (0, db_point_original_space.shape[0])}}
                f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
                f.flush()
                f.close()
        except Exception:
            traceback.print_exc()

    def update(self, entry_id, new_label, sample=None):
        """
        Updates an entry with entry_id with the given label.
        :param entry_id: entry id of the sample to update.
        :param label: Label of the sample to be update.
        """
        if isinstance(new_label, int):
            new_label = np.array(new_label).reshape(1, 1)
        if entry_id is None and sample is not None:
            self.data["features"] = np.concatenate((self.data["features"], to_vector(sample).T), axis=0)
            self.data["targets"] = np.concatenate((self.data["targets"], new_label), axis=0)
        else:
            self.data["targets"][entry_id] = new_label
        for callback in self._update_callback:
            callback(entry_id, new_label)

    def on_update(self, callback):
        self._update_callback.append(callback)

    def format_sklearn(self):
        """
        Returns dataset in (X, y) format for use in scikit-learn.
        Unlabeled entries are ignored.
        Returns
        -------
        X : numpy array, shape = (n_samples, n_features)
            Sample feature set.
        y : numpy array, shape = (n_samples)
            Sample labels.
        """
        labeled_train = self.get_labeled_train_data()
        X, y = labeled_train["features"], labeled_train["targets"]
        return X, y[:, 0]

    def format_keras(self):
        scaled_data = self.get_labeled_train_data()
        return scaled_data["features"], scaled_data["targets"]

    def validation_data_format_keras(self):
        scaled_val = self.get_validation_data()
        if scaled_val is None:
            return None, None
        else:
            return scaled_val["features"], scaled_val["targets"]

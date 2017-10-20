import time
from sklearn.svm import LinearSVC
import numpy as np
from joint_optimisation import do_optimization
import traceback
from utils import to_vector
import h5py
from fuel.datasets import H5PYDataset
import os

__author__ = 'mhuijser'

class Model(object):
    def __init__(self):
        pass

    @property
    def w(self):
        raise NotImplementedError

    @property
    def b(self):
        raise NotImplementedError

    def train(self, dataset):
        raise NotImplementedError

    def predict(self, feature):
        raise NotImplementedError

    def score(self, testing_dataset):
        raise NotImplementedError

    def predict_real(self, feature):
        raise NotImplementedError

class SimpleSVM(Model):
    def __init__(self):
        super(SimpleSVM, self).__init__()
        self.model = LinearSVC()

    @property
    def w(self):
        return self.model.coef_.T

    @property
    def b(self):
        return self.model.intercept_

    def train(self, dataset, *args, **kwargs):
        X, y = dataset.format_sklearn()
        start_time = time.time()
        fit = self.model.fit(X, y)
        print "Trained model on %d samples in %.2f seconds" % (len(y), time.time()-start_time)
        return fit

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)

    def predict_real(self, feature, *args, **kwargs):
        dvalue = self.model.decision_function(feature)
        if len(np.shape(dvalue)) == 1:  # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            return dvalue

class JointOptimisationSVM(Model):
    def __init__(self, initial_model=None, classes=None, hyperparameters=None, save_path_boundaries=None,
                 supervised=False, no_hyp=False):
        super(JointOptimisationSVM, self).__init__()
        self.save_path_boundaries = None
        if save_path_boundaries is not None:
            self.save_path_boundaries = os.path.join(save_path_boundaries,
                                                     os.path.normpath(save_path_boundaries) + "_ip_boundaries.hdf5")
            if supervised:
                self.save_path_boundaries = os.path.normpath(save_path_boundaries) + "_gt_boundaries.hdf5"
        self.__w = None
        self.__b = None
        self.__initial_model = None
        self.initial_model = initial_model
        self.decision_function = lambda x: self.w.T.dot(x) + self.b
        self.classes = classes # np array
        self.no_class = None
        self.trained = self.w is not None
        if not no_hyp:
            self.lambda_r = hyperparameters["lambda_r"]
            self.gamma = hyperparameters["gamma"]
            self.weight_hinge = hyperparameters["weight_hinge"]
            self.learning_rate = hyperparameters["learning_rate"]
            self.decay = hyperparameters["decay"]
            self.interval = hyperparameters["interval"]
            self.factor = hyperparameters["factor"]
            self.max_epochs = hyperparameters["max_epochs"]
            self.batch_size = hyperparameters["batch_size"]
        if supervised:
            self.save_decision_boundary(self.w, self.b)

    @property
    def initial_model(self):
        return self.__initial_model

    @initial_model.setter
    def initial_model(self, init_model):
        if init_model is not None:
            self.__w = init_model["w"]
            self.__b = init_model["b0"]
            self.__initial_model = init_model

    @property
    def model(self):
        return {"w": self.w, "b0":self.b}

    @property
    def w(self):
        return self.__w

    @property
    def b(self):
        if self.__b is None:
            return None
        return float(self.__b)

    def train(self, dataset, first_time=False, *args, **kwargs):
        X, labels = dataset.format_keras() # (n_samples, n_features)
        self.classes = np.unique(labels)
        self.no_class = dataset.unlabeled_class
        if first_time and self.initial_model is not None:
            print "No need for training initial model, already set!"
            return

        # Change labels to +1/-1. The biggest classlabel is the +1 class, the smallest class label is the -1 class.
        # So, labels 0,8 would become -1,1. Labels 7,1 would become 1,-1.
        X_val, labels_val = dataset.validation_data_format_keras()
        if self.no_class in labels:
            raise Exception("Model can only be trained on labeled data. Unlabeled data encountered!")
        y = np.copy(labels)
        y[y==min(self.classes)] = -1
        y[y==max(self.classes)] = 1

        y_val = None
        if labels_val is not None:
            y_val = np.copy(labels_val)
            y_val[y_val==min(self.classes)] = -1
            y_val[y_val==max(self.classes)] = 1
        # Get decision boundary points and label them with zeros
        Xi = dataset.get_db_points()
        if Xi is not None:
            print "Decision boundary points:", Xi.shape
        try:
            yi = np.zeros((Xi.shape[0],1))
        except AttributeError:
            yi = None

        # Joint optimisation
        try:
            history, (w, b) = do_optimization(X, y, Xi, yi, lambda_r=self.lambda_r, gamma=self.gamma, weight_hinge=self.weight_hinge,
                                              learning_rate=self.learning_rate, decay=self.decay,
                                              interval=self.interval, factor=self.factor, max_epochs=self.max_epochs,
                                              batch_size=self.batch_size, Xj_val=X_val, yj_val=y_val)
            del y, yi, X, Xi
            self.__w = w
            self.__b = float(b)

            self.trained = True
            if first_time:
                self.initial_model = self.model
            if self.save_path_boundaries is not None:
                self.save_decision_boundary(self.w, self.b)
        except Exception as e:
            print ">>> Exception occurred: ", traceback.format_exc()
            quit(0)

    def save_decision_boundary(self, w, b):
        """
        :return:
        """
        w = to_vector(w).T
        if os.path.isfile(self.save_path_boundaries):
            with h5py.File(self.save_path_boundaries, 'r+') as hf:
                w_dataset = hf.get('w')
                already_in_w_ds = w_dataset.shape[0]
                w_dataset.resize(already_in_w_ds + w.shape[0], axis=0)
                w_dataset[already_in_w_ds:already_in_w_ds + w.shape[0],:] = w

                b_dataset = hf.get('b')
                already_in_b_ds = b_dataset.len()
                b_dataset.resize(already_in_b_ds + 1, axis=0)
                b_dataset[already_in_b_ds:already_in_b_ds + 1] = b

                split_dict = {"data": {"w": (0, already_in_w_ds + w.shape[0]),
                                       "b": (0, already_in_b_ds + 1)}}
                hf.attrs["split"] = H5PYDataset.create_split_array(split_dict)
        else:
            # HDF5 query line save file does not exist yet!
            f = h5py.File(self.save_path_boundaries, "w")

            w_dataset = f.create_dataset('w', w.shape,
                                              maxshape=(None, w.shape[1]), dtype="float32")
            w_dataset[...] = w

            b_dataset = f.create_dataset('b', (1,), maxshape=(None,), dtype="float32")
            b_dataset[...] = b

            split_dict = {"data": {
                "w": (0, w.shape[0]),
                "b": (0, 1)}
            }
            f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
            f.flush()
            f.close()

    def predict(self, sample, *args, **kwargs):
        """

        :param sample: n_samples x n_features
        :return:
        """
        if not self.trained:
            raise Exception("Cannot predict with an untrained model.")
        scores = self.decision_function(sample.T).T
        scores[scores==0] = self.no_class
        scores[scores>0] = max(self.classes)
        scores[scores<0] = min(self.classes)
        labels = scores
        return labels

    def score(self, data, *args, **kwargs):
        """
        Classify test data and compute accuracy.
        X: shape = (n_samples, n_features)
        y: shape = (n_samples, 1)
        :param args:
        :param kwargs:
        :return:
        """
        X, y_true = data
        if not self.trained:
            raise Exception("Cannot test with an untrained model.")
        n_samples = X.shape[0]
        y_pred = self.predict(X)
        n_correct = np.sum(y_pred == y_true)
        return float(n_correct)/n_samples

    def predict_real(self, sample, vstack=True, *args, **kwargs):
        """
        Computes per-class confidence scores for sample(s).
        The confidence score for a sample is the signed distance of that sample to the hyperplane.
        Returns: array-like, shape (n_samples, n_classes)
            Each entry is the confidence scores per (sample, class)
            combination.
        :param sample: n_samples x n_features
        :param args:
        :param kwargs:
        :return:
        """
        if not self.trained:
            raise Exception("Cannot predict with an untrained model.")
        dvalue = self.decision_function(sample.T)
        if vstack:
            return np.vstack((-dvalue, dvalue)).T
        else:
            return dvalue

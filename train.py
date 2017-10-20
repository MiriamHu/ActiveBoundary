import options
import h5py
import matplotlib
matplotlib.use('Agg')
import time
from fuel.datasets import H5PYDataset
from dataset import Dataset
from query_strategy import UncertaintySamplingLine, UncertaintyDenseSamplingLine, ClusterCentroidsLine, \
    RandomSamplingLine
from model import JointOptimisationSVM
from labeler import LineLabeler, NoisyLineLabeler, HumanLineLabeler
from plot_utils import *
from generative_model import GenerativeALIModel
import os
import traceback
import math

__author__ = 'mhuijser'

"""
load in trained ALI model
load in train set.
Split up train set into labeled images and unlabeled images
Encode labeled and unlabeled images
Center all the encoded images
Train a base classifier on labeled encoded images. This gives initial w and b
Enter active learning loop with encoded labeled images, encoded unlabeled train images, base classifier, encoded test images
"""
class ActiveBoundary(object):
    def __init__(self, opt, groundtruth_model=None, indices_unlabeled=None, initial_model=None):
        """
        :param opt: parameters that can be set in options.py.
        :param groundtruth_model: {"w": (dim, 1), "b": float}. Use as ground truth model for labeling query samples.
        If None, ground truth model is trained from all train samples treating all as labeled.
        :param indices_unlabeled: array of indices of data samples that will be set as unlabeled.
        If None, the unlabeled samples will be randomly selected.
        :param initial_model: {"w": (dim, 1), "b": float}. Start active learning with this base model.
        If None, initial model is trained using initially labeled samples.
        """
        self.groundtruth_model = groundtruth_model
        self.indices_unlabeled = indices_unlabeled
        self.initial_model = initial_model

        if opt.query_strategy == "clustercentroids":
            opt.al_batch_size = opt.cluster_batch_size
            opt.iterations = int(opt.iterations / float(opt.al_batch_size))
        self.human_experiment = opt.oracle_type == "human_line_labeler"
        if not self.human_experiment:
            opt.base_precision = None

        os.mkdir(opt.save_path)
        try:
            self.dataset, self.model, self.query_strategy, self.labeler = self.init(*self.load_ali(opt))
        except Exception as e:
            print ">>> Exception occurred while initializing linebased: ", traceback.format_exc()

    def save_indices_unlabeled(self, opt, unlabeled_indices):
        path_indices_hdf5 = os.path.join(opt.save_path, os.path.normpath(opt.save_path) + "_unlabeled_indices.hdf5")
        if os.path.isfile(path_indices_hdf5):
            print "This file already exists %s" % path_indices_hdf5
            quit(0)
        f = h5py.File(path_indices_hdf5, "w")

        unlabeled_indices_dataset = f.create_dataset('unlabeled_indices', unlabeled_indices.shape, maxshape=(None,))
        unlabeled_indices_dataset[...] = unlabeled_indices

        split_dict = {"data": {"unlabeled_indices": (0, len(unlabeled_indices))}}
        f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        f.flush()
        f.close()

    def load_ali(self, opt, unlabeled_class=-5):
        y_groundtruth = None
        X_train_embedded = None
        y_val = None
        X_val_embedded = None
        self.ali = None
        if opt.enable_gpu:
            if not os.path.isfile(opt.main_loop_path):
                print "Exiting...main loop path does not exist"
                return
            print 'Loading ALI model from main loop path', opt.main_loop_path
            self.ali = GenerativeALIModel(opt.main_loop_path)
            print "Loading training data from", opt.hdf5_dataset
            X_train, y_groundtruth = self.load_data(opt.hdf5_dataset, set='train', sources=['features', 'targets'])
            X_train_embedded = self.ali.encode(X_train).squeeze()
            del X_train
            print "Finished encoding train data. Current shape: ", X_train_embedded.shape
        else:
            print "Loading training data from", opt.hdf5_dataset_encoded
            X_train_embedded, y_groundtruth = self.load_data(opt.hdf5_dataset_encoded, set="train",
                                                             sources=['features', 'targets'])

        y = y_groundtruth.copy()
        if self.indices_unlabeled is None:
            n_samples = X_train_embedded.shape[0]
            n_unlabeled = int(n_samples * (1.0 - opt.percentage_labeled))
            self.indices_unlabeled = np.random.choice(range(n_samples), n_unlabeled, replace=False)
        y[self.indices_unlabeled] = unlabeled_class
        self.save_indices_unlabeled(opt, self.indices_unlabeled)

        # Adjust number of active learning iterations to the number of iterations possible given the number of unlabeled
        # samples in the data set and the batch size.
        al_iterations_possible = int(math.ceil(len(self.indices_unlabeled) / float(opt.al_batch_size)))
        opt.iterations = min(al_iterations_possible, opt.iterations)

        return opt, X_train_embedded, y, y_groundtruth, X_val_embedded, y_val, unlabeled_class, self.ali

    def init(self, opt, X, y, y_groundtruth, X_val, y_val, unlabeled_class=-5, ali_model=None):
        dataset = Dataset(X, y, y_groundtruth, X_val, y_val, unlabeled_class=unlabeled_class,
                          al_batch_size=opt.al_batch_size, save_path_db_points=opt.save_path,
                          dataset=opt.hdf5_dataset_encoded)
        print "All samples: ", len(dataset)
        print "Labeled samples: ", dataset.len_labeled()
        print "Unlabeled samples: ", dataset.len_unlabeled()

        print "Initializing model"
        model = JointOptimisationSVM(initial_model=self.initial_model,
                                     hyperparameters=opt.hyperparameters,
                                     save_path_boundaries=opt.save_path)  # declare model instance
        print "Done declaring model"
        print "Initializing query strategy", opt.query_strategy
        if opt.query_strategy == "uncertainty":
            query_strategy = UncertaintySamplingLine(dataset, model=model, generative_model=ali_model,
                                                     save_path_queries=opt.save_path,
                                                     human_experiment=self.human_experiment,
                                                     base_precision=opt.base_precision)  # declare a QueryStrategy instance
        elif opt.query_strategy == "uncertainty-dense":
            query_strategy = UncertaintyDenseSamplingLine(dataset, model=model, generative_model=ali_model,
                                                          save_path_queries=opt.save_path,
                                                          human_experiment=self.human_experiment,
                                                          base_precision=opt.base_precision)
        elif opt.query_strategy == "clustercentroids":
            query_strategy = ClusterCentroidsLine(dataset, model=model, generative_model=ali_model,
                                                  batch_size=opt.al_batch_size, save_path_queries=opt.save_path,
                                                  human_experiment=self.human_experiment,
                                                  base_precision=opt.base_precision)
        elif opt.query_strategy == "random":
            query_strategy = RandomSamplingLine(dataset, model=model, generative_model=ali_model,
                                                save_path_queries=opt.save_path,
                                                human_experiment=self.human_experiment,
                                                base_precision=opt.base_precision)
        else:
            raise Exception("Please specify a query strategy")
        print "Done declaring query strategy", opt.query_strategy

        if opt.oracle_type == "noisy_line_labeler":
            labeler = NoisyLineLabeler(dataset, opt.std_noise, pretrained_groundtruth=self.groundtruth_model,
                                       hyperparameters=opt.hyperparameters)
            print "Done declaring NoisyLineLabeler"
        elif opt.oracle_type == "human_line_labeler":
            labeler = HumanLineLabeler(dataset, ali_model, hyperparameters=opt.hyperparameters)
        else:
            labeler = LineLabeler(dataset, pretrained_groundtruth=self.groundtruth_model,
                                  hyperparameters=opt.hyperparameters)  # declare Labeler instance
            print "Done declaring LineLabeler"
        print "Done initializing"
        return dataset, model, query_strategy, labeler

    def run(self, opt):
        test_scores_per_iteration = []
        begin_system_time = time.time()
        no_more_samples = False

        if os.path.isfile(opt.hdf5_dataset_encoded):
            X_test_embedded, y_test = self.load_data(opt.hdf5_dataset_encoded, set="test",
                                                     sources=["features", "targets"])
        else:
            raise Exception("Test data could not be loaded")
        X_test_embedded = self.dataset.scaling_transformation.transform(X_test_embedded)

        test_scores_per_iteration.append(self.query_strategy.model.score((X_test_embedded, y_test)))
        print "Test accuracy initial model: ", test_scores_per_iteration[-1]
        del X_test_embedded, y_test

        for i in xrange(opt.iterations):  # loop through the number of queries
            try:
                print "\nIteration", i + 1
                if opt.plot:
                    plot_base(self.dataset, self.labeler, self.model, i + 1)
                start_time = time.time()
                for b in range(opt.al_batch_size):
                    try:
                        query_id, line, line_segment, query_image, intersection_point_cdb = self.query_strategy.make_query(
                            n_images=opt.n_images_line_query)  # let the specified QueryStrategy suggest a data point to query
                    except IndexError as e:
                        print(e)
                        no_more_samples = True
                        break
                    if opt.plot:
                        plot_after_query(line_segment, self.dataset.data["features"][query_id], self.dataset,
                                         self.labeler, self.model, i + 1)
                    if opt.query_strategy == "clustercentroids":
                        lbl, db_point = self.labeler.label(query_image, line, line_segment, sample_already_scaled=True,
                                                           intersection_point_cdb=intersection_point_cdb)
                    else:
                        lbl, db_point = self.labeler.label(self.dataset.data["features"][query_id], line, line_segment,
                                                           sample_already_scaled=False,
                                                           intersection_point_cdb=intersection_point_cdb)  # query the label of the example at query_id
                    if db_point is not None:
                        self.dataset.add_db_point(db_point.T)
                    try:
                        self.dataset.update(query_id, lbl,
                                            sample=query_image)  # update the dataset with newly-labeled example
                    except Exception as e:
                        print "EXCEPTION", traceback.format_exc()
                    if opt.plot:
                        plot_after_label(line_segment, self.dataset.data["features"][query_id], self.dataset,
                                         self.labeler, self.model, i + 1)
                self.model.train(self.dataset)  # train model with newly-updated Dataset

                # Test here on embedded test data and append to test_scores_per_iteration
                if opt.hdf5_dataset is not None:
                    print "Testing on embedded test data"
                    print "Loading test data"
                    if os.path.isfile(opt.hdf5_dataset_encoded):
                        X_test_embedded, y_test = self.load_data(opt.hdf5_dataset_encoded, set="test",
                                                                 sources=["features", "targets"])
                    elif opt.enable_gpu:
                        X_test, y_test = self.load_data(opt.hdf5_dataset, set='test', sources=['features', 'targets'])
                        print "Encoding test data"
                        X_test_embedded = self.ali.encode(X_test).squeeze()
                        del X_test
                    else:
                        raise Exception("Test data could not be loaded")
                    print "Centering test data"
                    X_test_embedded = self.dataset.scaling_transformation.transform(X_test_embedded)
                    try:
                        test_scores_per_iteration.append(self.model.score((X_test_embedded, y_test)))
                        print "Test accuracy: ", test_scores_per_iteration[-1]
                    except Exception:
                        print "EXCEPTION", traceback.format_exc()
                    if 0 in test_scores_per_iteration:
                        print "Found 0 in test scores, line-based experiment failed. Trying again."
                        return test_scores_per_iteration
                    del X_test_embedded, y_test
                print "One iteration took %.2f seconds" % (time.time() - start_time)
                if no_more_samples:
                    break
            except KeyboardInterrupt:
                print "Iterating %d times took %.2f seconds" % (
                    len(test_scores_per_iteration), time.time() - begin_system_time)
                return test_scores_per_iteration
        print "Iterating %d times took %.2f seconds" % (len(test_scores_per_iteration), time.time() - begin_system_time)
        if opt.plot:
            plot_base(self.dataset, self.labeler, self.model, opt.iterations)
        return test_scores_per_iteration

    @staticmethod
    def load_data(hdf5_file, set, sources):
        data = H5PYDataset(hdf5_file,
                           which_sets=(set,),
                           sources=sources,
                           load_in_memory=True)
        X, y = data.data_sources
        return X.astype(np.float32), y.astype(np.float32)

    @staticmethod
    def save_encoded_test_data(save_file, X_test_encoded, y_test):
        import h5py
        f = h5py.File(save_file, mode="w")
        features = f.create_dataset("features", X_test_encoded.shape, dtype="float32")
        targets = f.create_dataset("targets", y_test.shape, dtype="uint8")
        features[...] = X_test_encoded
        targets[...] = y_test
        split_dict = {"test": {"features": (0, X_test_encoded.shape[0]), "targets": (0, X_test_encoded.shape[0])}}
        f.attrs["split"] = H5PYDataset.create_split_array(split_dict)
        f.flush()
        f.close()


if __name__ == "__main__":
    print("Setting configurations...")
    opt = options.set()
    system = ActiveBoundary(opt)
    test_scores_per_iteration = system.run(opt)

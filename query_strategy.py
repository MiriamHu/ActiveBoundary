import h5py
import matplotlib
from fuel.datasets import H5PYDataset
import math
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
import time
from six import with_metaclass
from abc import ABCMeta, abstractmethod
import numpy as np
from utils import project_point_on_decision_boundary, make_line_segment, to_vector, compute_radius_sphere
import os
import traceback
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

__author__ = 'mhuijser'


class QueryStrategy(with_metaclass(ABCMeta, object)):
    """Pool-based query strategy
    A QueryStrategy advices on which unlabeled data to be queried next given
    a pool of labeled and unlabeled data.
    """

    def __init__(self, dataset, **kwargs):
        super(QueryStrategy, self).__init__()
        self._dataset = dataset
        dataset.on_update(self.update)

    def save_query_to_hdf5_point(self, save_path_queries_hdf5, entry_id, sample_original_space):
        """
        :return:
        """
        sample_original_space = to_vector(sample_original_space).T
        if os.path.isfile(save_path_queries_hdf5):
            with h5py.File(save_path_queries_hdf5, 'r+') as hf:
                points_dataset = hf.get('point_queries')
                already_in_points_ds = points_dataset.shape[0]
                points_dataset.resize(already_in_points_ds + sample_original_space.shape[0], axis=0)
                points_dataset[already_in_points_ds:already_in_points_ds + sample_original_space.shape[0],
                :] = sample_original_space

                entryids_dataset = hf.get('entry_ids')
                already_in_entryids_ds = entryids_dataset.len()
                entryids_dataset.resize(already_in_entryids_ds + 1, axis=0)
                entryids_dataset[already_in_entryids_ds:already_in_entryids_ds + 1] = entry_id

                split_dict = {"data": {"point_queries": (0, already_in_points_ds + sample_original_space.shape[0]),
                                       "entry_ids": (0, already_in_entryids_ds + 1)}}
                hf.attrs["split"] = H5PYDataset.create_split_array(split_dict)
        else:
            # HDF5 query line save file does not exist yet!
            f = h5py.File(save_path_queries_hdf5, "w")

            points_dataset = f.create_dataset('point_queries', sample_original_space.shape,
                                              maxshape=(None, sample_original_space.shape[1]), dtype="float32")
            points_dataset[...] = sample_original_space
            entryids_dataset = f.create_dataset('entry_ids', (1,), maxshape=(None,), dtype=int)
            entryids_dataset[...] = entry_id

            split_dict = {"data": {
                "point_queries": (0, sample_original_space.shape[0]),
                "entry_ids": (0, 1)}
            }
            f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
            f.flush()
            f.close()

    @property
    def dataset(self):
        """The Dataset object that is associated with this QueryStrategy."""
        return self._dataset

    def update(self, entry_id, label):
        """Update the internal states of the QueryStrategy after each queried
        sample being labeled.
        Parameters
        ----------
        entry_id : int
            The index of the newly labeled sample.
        label : float
            The label of the queried sample.
        """
        pass

    @abstractmethod
    def make_query(self):
        """Return the index of the sample to be queried and labeled. Read-only.
        No modification to the internal states.
        Returns
        -------
        ask_id : int
            The index of the next unlabeled sample to be queried and labeled.
        """
        pass


class LineQueryStrategy(QueryStrategy):
    def __init__(self, dataset, model, generative_model, human_experiment, **kwargs):
        super(LineQueryStrategy, self).__init__(dataset, **kwargs)
        self.model = model
        self.generative_model = generative_model
        self.human_experiment = human_experiment

    @abstractmethod
    def make_line(self, query_id, *args):
        pass


class UncertaintySampling(QueryStrategy):
    """
    Uncertainty Sampling
    """

    def __init__(self, dataset, save_path_queries=None, **kwargs):
        super(UncertaintySampling, self).__init__(dataset, **kwargs)

        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )
        self.model.train(self.dataset, first_time=True)
        self.save_path_queries = save_path_queries
        self.save_path_queries_hdf5 = os.path.join(self.save_path_queries,
                                                   os.path.normpath(self.save_path_queries) + ".hdf5")
        if os.path.isfile(self.save_path_queries_hdf5):
            print "This file already exists %s" % self.save_path_queries_hdf5
            quit(0)

    def make_query(self):
        try:
            unlabeled_train_data = self.dataset.get_unlabeled_train_data()
        except ValueError:
            raise IndexError("No more unlabeled train samples")
        unlabeled_entry_ids, X_pool = unlabeled_train_data["entry_ids"], unlabeled_train_data["features"]
        del unlabeled_train_data
        if len(X_pool) > 0:
            # least confident
            start_time = time.time()
            ask_id = np.argmin(
                np.max(self.model.predict_real(X_pool), axis=1)
            )
            print "Found new query amongst %d unlabeled samples in %.2f seconds" % (
                X_pool.shape[0], time.time() - start_time)

            if self.save_path_queries is not None:
                try:
                    self.save_query_to_hdf5_point(self.save_path_queries_hdf5, unlabeled_entry_ids[ask_id],
                                                  self.dataset.data["features"][unlabeled_entry_ids[ask_id]])
                except:
                    traceback.print_exc()
            return unlabeled_entry_ids[ask_id], None
        else:
            raise IndexError("No more unlabeled train samples")


class UncertaintyDenseSampling(QueryStrategy):
    def __init__(self, dataset, save_path_queries=None, **kwargs):
        super(UncertaintyDenseSampling, self).__init__(dataset, **kwargs)

        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )
        self.save_path_queries = save_path_queries
        self.save_path_queries_hdf5 = os.path.join(self.save_path_queries,
                                                   os.path.normpath(self.save_path_queries) + ".hdf5")
        if os.path.isfile(self.save_path_queries_hdf5):
            print "This file already exists %s" % self.save_path_queries_hdf5
            quit(0)
        self.model.train(self.dataset, first_time=True)
        unlabeled_train = self.dataset.get_unlabeled_train_data()["features"]
        print "Computing cosine similarities of", unlabeled_train.shape, "by", unlabeled_train.shape
        self.similarity_matrix = cosine_similarity(unlabeled_train, unlabeled_train)

    def make_query(self):
        try:
            unlabeled_train_data = self.dataset.get_unlabeled_train_data()
        except ValueError:
            raise IndexError("No more unlabeled train samples")
        unlabeled_entry_ids, X_pool = unlabeled_train_data["entry_ids"], unlabeled_train_data["features"]
        del unlabeled_train_data
        if len(X_pool) > 0:
            start_time = time.time()
            uncertainties = np.max(self.model.predict_real(X_pool), axis=1)
            ask_id = self.get_most_uncertainty_dense(uncertainties, self.similarity_matrix, beta=1)
            self.delete_index_similarity_matrix(ask_id)
            print "Found new query amongst %d unlabeled samples in %.2f seconds" % (
                X_pool.shape[0], time.time() - start_time)
            if self.save_path_queries is not None:
                self.save_query_to_hdf5_point(self.save_path_queries_hdf5, unlabeled_entry_ids[ask_id],
                                              self.dataset.data["features"][unlabeled_entry_ids[ask_id]])

            return unlabeled_entry_ids[ask_id], None
        else:
            raise IndexError("No more unlabeled train samples")

    def delete_index_similarity_matrix(self, entry_index):
        self.similarity_matrix = np.delete(np.delete(self.similarity_matrix, entry_index, axis=0), entry_index, axis=1)

    def get_most_uncertainty_dense(self, uncertainties, similarities, beta):
        times = uncertainties * similarities.mean(axis=0) ** beta
        return np.argmax(times)


class RandomSampling(QueryStrategy):
    def __init__(self, dataset, save_path_queries=None, **kwargs):
        super(RandomSampling, self).__init__(dataset, **kwargs)

        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )
        self.save_path_queries = save_path_queries
        self.save_path_queries_hdf5 = os.path.join(self.save_path_queries,
                                                   os.path.normpath(self.save_path_queries) + ".hdf5")
        if os.path.isfile(self.save_path_queries_hdf5):
            print "This file already exists %s" % self.save_path_queries_hdf5
            quit(0)
        self.model.train(self.dataset, first_time=True)

    def make_query(self):
        try:
            unlabeled_train_data = self.dataset.get_unlabeled_train_data()
        except ValueError:
            raise IndexError("No more unlabeled train samples")
        unlabeled_entry_ids, X_pool = unlabeled_train_data["entry_ids"], unlabeled_train_data["features"]
        del unlabeled_train_data
        if len(X_pool) > 0:
            start_time = time.time()
            ask_id = np.random.randint(0, len(unlabeled_entry_ids))
            print "Found new query amongst %d unlabeled samples in %.2f seconds" % (
                X_pool.shape[0], time.time() - start_time)
            if self.save_path_queries is not None:
                self.save_query_to_hdf5_point(self.save_path_queries_hdf5, unlabeled_entry_ids[ask_id],
                                              self.dataset.data["features"][unlabeled_entry_ids[ask_id]])

            return unlabeled_entry_ids[ask_id], None
        else:
            raise IndexError("No more unlabeled train samples")


class ClusterCentroids(QueryStrategy):
    def __init__(self, dataset, batch_size, save_path_queries=None, **kwargs):
        super(ClusterCentroids, self).__init__(dataset, **kwargs)
        self.batch_size = batch_size
        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )
        self.save_path_queries = save_path_queries
        self.save_path_queries_hdf5 = os.path.join(self.save_path_queries,
                                                   os.path.normpath(self.save_path_queries) + ".hdf5")
        if os.path.isfile(self.save_path_queries_hdf5):
            print "This file already exists %s" % self.save_path_queries_hdf5
            quit(0)
        self.model.train(self.dataset, first_time=True)
        self.n_queries = 0
        self.clustered = None

    def make_query(self):
        try:
            unlabeled_train_data = self.dataset.get_unlabeled_train_data()
        except ValueError:
            raise IndexError("No more unlabeled train samples")
        unlabeled_entry_ids, X_pool = unlabeled_train_data["entry_ids"], unlabeled_train_data["features"]
        del unlabeled_train_data
        if len(X_pool) > 0:
            start_time = time.time()
            # Cluster centroids!
            if self.n_queries % self.batch_size == 0:
                # Cluster again!
                self.clustered = KMeans(n_clusters=self.batch_size).fit(X_pool)
            query_image = self.clustered.cluster_centers_[self.n_queries % self.batch_size]
            query_image_original_space = self.dataset._scaling_transformation.inverse_transform(
                to_vector(query_image).T).T  # (n_features, 1)
            print "Found new query using %d unlabeled clustered samples in %.2f seconds" % (
                X_pool.shape[0], time.time() - start_time)
            self.n_queries += 1
            if self.save_path_queries is not None:
                self.save_query_to_hdf5_point(self.save_path_queries_hdf5, -1, query_image_original_space.T)

            return None, query_image
        else:
            raise IndexError("No more unlabeled train samples")


class UncertaintySamplingLine(LineQueryStrategy):
    def __init__(self, dataset, model, generative_model, save_path_queries=None, human_experiment=False,
                 base_precision=None, **kwargs):
        super(UncertaintySamplingLine, self).__init__(dataset, model, generative_model,
                                                      human_experiment=human_experiment, **kwargs)
        self.model.train(self.dataset, first_time=True)
        self.save_path_queries = save_path_queries
        self.n_queries = 0
        self.base_precision = base_precision
        self.r = None
        self.save_path_queries_hdf5 = os.path.join(self.save_path_queries,
                                                   os.path.normpath(self.save_path_queries) + ".hdf5")
        if os.path.isfile(self.save_path_queries_hdf5):
            print "This file already exists %s" % self.save_path_queries_hdf5
            quit(0)

    @property
    def longest_line_possible(self):
        try:
            return 2. * self.r
        except TypeError:
            return None

    def make_query(self, n_images=50):
        try:
            unlabeled_train_data = self.dataset.get_unlabeled_train_data()
        except ValueError:
            raise IndexError("No more unlabeled train samples")
        unlabeled_entry_ids, X_pool = unlabeled_train_data["entry_ids"], unlabeled_train_data["features"]
        del unlabeled_train_data
        if len(X_pool) > 0:
            # least confident
            start_time = time.time()
            print self.model.predict_real(X_pool).shape
            ask_id = np.argmin(
                np.max(self.model.predict_real(X_pool), axis=1)
            )
            print "Found new query amongst %d unlabeled samples in %.2f seconds" % (
                X_pool.shape[0], time.time() - start_time)
            # Make line. Project query_image on current decision boundary
            start_time = time.time()
            line, line_segment, intersection_point = self.make_line(X_pool[ask_id], n_images)
            print "Made line from found query in %.2f seconds" % (time.time() - start_time)
            if not self.human_experiment and self.generative_model is not None:
                # Change 1 to higher number for faster algorithm (less generating and plotting)
                if self.n_queries % 1 == 0:
                    start_time = time.time()
                    self.generate_images_line_save(line_segment, unlabeled_entry_ids[ask_id])
                    print "Plotted query line in %.2f seconds" % (time.time() - start_time)
            else:
                self.save_query_to_hdf5(
                    to_vector(self.dataset.data["features"][unlabeled_entry_ids[ask_id]]).T,
                    # Already in original space
                    unlabeled_entry_ids[ask_id],
                    self.dataset.scaling_transformation.inverse_transform(line_segment),  # Transform to original space,
                    self.dataset.scaling_transformation.inverse_transform(intersection_point.T)
                )
            self.n_queries += 1
            return unlabeled_entry_ids[ask_id], line, line_segment, None, intersection_point
        else:
            raise IndexError("No more unlabeled train samples")

    def make_line(self, x, n_images=50):
        """
        Returns tuple of lambda function for query line and line segment (which has shape (n_images, n_features))
        :param x: query point in scaled space
        :param n_images: how many images the line segment will consist of.
        :return:
        """
        x = to_vector(x)  # in scaled space
        x_p = project_point_on_decision_boundary(self.model.w, self.model.b, x)  # in scaled space
        line = lambda t: x_p + (x - x_p) * t  # in scaled space

        all_train_data_scaled = self.dataset.scaling_transformation.transform(
            self.dataset.data["features"])  # Get all data in scaled space
        all_data_center = to_vector(all_train_data_scaled.mean(axis=0))  # Mean in scaled space (probably, near zero)
        if self.r is None:  # Compute r only the first time
            self.r = compute_radius_sphere(scaled_data=all_train_data_scaled)
            print "Radius", self.r
        del all_train_data_scaled

        if self.base_precision is not None:
            n_images = int(math.ceil(self.longest_line_possible / float(self.base_precision)) + 1)

        line_segment = make_line_segment(radius=self.r, mu=all_data_center, a=to_vector(x_p), b=to_vector(x),
                                         n_points_line=n_images)
        return line, line_segment, to_vector(x_p)

    def generate_images_line_save(self, line_segment, query_id, image_original_space=None):
        """
        ID of query point from which query line was generated is
        added to the filename of the saved line query.
        :param line_segment:
        :param query_id:
        :return:
        """
        try:
            if image_original_space is not None:
                x = self.generative_model.decode(image_original_space.T)
            else:
                x = self.generative_model.decode(to_vector(self.dataset.data["features"][
                                                               query_id]).T)  # comes from dataset.data["features"], so is already in original space in which ALI operates.
            save_path = os.path.join(self.save_path_queries, "pointquery_%d_%d.png" % (self.n_queries + 1, query_id))
            if x.shape[1] == 1:
                plt.imsave(save_path, x[0, 0, :, :], cmap=cm.Greys)
            else:
                plt.imsave(save_path, x[0, :, :, :].transpose(1, 2, 0), cmap=cm.Greys_r)

            decoded_images = self.generative_model.decode(self.dataset.scaling_transformation.inverse_transform(
                line_segment))  # Transform to original space, in which ALI operates.
            figure = plt.figure()
            grid = ImageGrid(figure, 111, (1, decoded_images.shape[0]), axes_pad=0.1)
            for image, axis in zip(decoded_images, grid):
                if image.shape[0] == 1:
                    axis.imshow(image[0, :, :].squeeze(),
                                cmap=cm.Greys, interpolation='nearest')
                else:
                    axis.imshow(image.transpose(1, 2, 0).squeeze(),
                                cmap=cm.Greys_r, interpolation='nearest')
                axis.set_yticklabels(['' for _ in range(image.shape[1])])
                axis.set_xticklabels(['' for _ in range(image.shape[2])])
                axis.axis('off')
            save_path = os.path.join(self.save_path_queries, "linequery_%d_%d.pdf" % (self.n_queries + 1, query_id))
            plt.savefig(save_path, transparent=True, bbox_inches='tight')
        except Exception as e:
            print "EXCEPTION:", traceback.format_exc()
            raise e

    def save_query_to_hdf5(self, sample_original_space, entry_id, line_segment_original_space,
                           intersection_point_original_space):
        """

        :param sample_original_space: point sample (n_samples=1, n_dimensionality_latent_space)
        :param line_segment_original_space: (n_images_line=14, n_dimensionality_latent_space)
        :param intersection_point_original_space: intersection point line query with current decision boundary
        (1, n_dimensionality_latent_space)
        :return:
        """
        shape_line = line_segment_original_space.shape
        line_segment_original_space = line_segment_original_space.reshape(1, shape_line[0], shape_line[1])
        shape_line = line_segment_original_space.shape
        if os.path.isfile(self.save_path_queries_hdf5):
            with h5py.File(self.save_path_queries_hdf5, 'r+') as hf:
                lines_dataset = hf.get('line_queries')
                already_in_dataset = lines_dataset.shape[0]
                lines_dataset.resize(already_in_dataset + shape_line[0], axis=0)
                lines_dataset[already_in_dataset:already_in_dataset + shape_line[0], :, :] = line_segment_original_space

                points_dataset = hf.get('point_queries')
                already_in_points_ds = points_dataset.shape[0]
                points_dataset.resize(already_in_points_ds + sample_original_space.shape[0], axis=0)
                points_dataset[already_in_points_ds:already_in_points_ds + sample_original_space.shape[0],
                :] = sample_original_space

                intersection_dataset = hf.get('intersection_points')
                already_in_intersection_ds = intersection_dataset.shape[0]
                intersection_dataset.resize(already_in_intersection_ds + intersection_point_original_space.shape[0],
                                            axis=0)
                intersection_dataset[
                already_in_intersection_ds:already_in_intersection_ds + intersection_point_original_space.shape[0],
                :] = intersection_point_original_space

                entryids_dataset = hf.get('entry_ids')
                already_in_entryids_ds = entryids_dataset.len()
                entryids_dataset.resize(already_in_entryids_ds + 1, axis=0)
                entryids_dataset[already_in_entryids_ds:already_in_entryids_ds + 1] = entry_id

                split_dict = {"data": {"line_queries": (0, already_in_dataset + shape_line[0]),
                                       "point_queries": (0, already_in_points_ds + sample_original_space.shape[0]),
                                       "intersection_points": (
                                           0, already_in_intersection_ds + intersection_point_original_space.shape[0]),
                                       "entry_ids": (0, already_in_entryids_ds + 1)}}
                hf.attrs["split"] = H5PYDataset.create_split_array(split_dict)
        else:
            # HDF5 query line save file does not exist yet!
            f = h5py.File(self.save_path_queries_hdf5, "w")
            lines_dataset = f.create_dataset('line_queries', shape_line, maxshape=(None, shape_line[1], shape_line[2]),
                                             dtype="float32")
            lines_dataset[...] = line_segment_original_space
            points_dataset = f.create_dataset('point_queries', sample_original_space.shape,
                                              maxshape=(None, sample_original_space.shape[1]), dtype="float32")
            points_dataset[...] = sample_original_space

            intersection_dataset = f.create_dataset('intersection_points', intersection_point_original_space.shape,
                                                    maxshape=(None, intersection_point_original_space.shape[1]),
                                                    dtype='float32')
            intersection_dataset[...] = intersection_point_original_space

            entryids_dataset = f.create_dataset('entry_ids', (1,), maxshape=(None,), dtype=int)
            entryids_dataset[...] = entry_id

            split_dict = {"data": {
                "line_queries": (0, shape_line[0]),
                "point_queries": (0, sample_original_space.shape[0]),
                "intersection_points": (0, intersection_point_original_space.shape[0]),
                "entry_ids": (0, 1)}
            }
            f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
            f.flush()
            f.close()


class RandomSamplingLine(UncertaintySamplingLine):
    def __init__(self, dataset, model, generative_model, save_path_queries=None, human_experiment=False,
                 base_precision=None, **kwargs):
        super(RandomSamplingLine, self).__init__(dataset, model, generative_model,
                                                 save_path_queries=save_path_queries,
                                                 human_experiment=human_experiment,
                                                 base_precision=base_precision, **kwargs)
        self.model.train(self.dataset, first_time=True)
        self.save_path_queries = save_path_queries
        self.n_queries = 0
        self.r = None

    def make_query(self, n_images=50):
        try:
            unlabeled_train_data = self.dataset.get_unlabeled_train_data()
        except ValueError:
            raise IndexError("No more unlabeled train samples")
        unlabeled_entry_ids, X_pool = unlabeled_train_data["entry_ids"], unlabeled_train_data["features"]
        del unlabeled_train_data
        if len(X_pool) > 0:
            # least confident
            start_time = time.time()
            ask_id = np.random.randint(0, len(unlabeled_entry_ids))
            print "Found new query amongst %d unlabeled samples in %.2f seconds" % (
                X_pool.shape[0], time.time() - start_time)
            # Make line. Project query_image on current decision boundary
            start_time = time.time()
            line, line_segment, intersection_point = self.make_line(X_pool[ask_id], n_images)
            print "Made line from found query in %.2f seconds" % (time.time() - start_time)
            if not self.human_experiment and self.generative_model is not None:
                # Change 1 to higher number for faster algorithm (less generating and plotting)
                if self.n_queries % 1 == 0:
                    start_time = time.time()
                    self.generate_images_line_save(line_segment, unlabeled_entry_ids[ask_id])
                    print "Plotted query line in %.2f seconds" % (time.time() - start_time)
            else:
                self.save_query_to_hdf5(
                    to_vector(self.dataset.data["features"][unlabeled_entry_ids[ask_id]]).T,
                    # Already in original space
                    unlabeled_entry_ids[ask_id],
                    self.dataset.scaling_transformation.inverse_transform(line_segment),  # Transform to original space,
                    self.dataset.scaling_transformation.inverse_transform(intersection_point.T)
                )
            self.n_queries += 1
            return unlabeled_entry_ids[ask_id], line, line_segment, None, intersection_point
        else:
            raise IndexError("No more unlabeled train samples")


class UncertaintyDenseSamplingLine(UncertaintySamplingLine):
    def __init__(self, dataset, model, generative_model, save_path_queries=None, human_experiment=False,
                 base_precision=None, **kwargs):
        super(UncertaintyDenseSamplingLine, self).__init__(dataset, model, generative_model,
                                                           save_path_queries=save_path_queries,
                                                           human_experiment=human_experiment,
                                                           base_precision=base_precision, **kwargs)
        self.model.train(self.dataset, first_time=True)
        self.save_path_queries = save_path_queries
        self.n_queries = 0
        self.r = None
        unlabeled_train = self.dataset.get_unlabeled_train_data()["features"]
        print "Computing cosine similarities of", unlabeled_train.shape, "by", unlabeled_train.shape
        self.similarity_matrix = cosine_similarity(unlabeled_train, unlabeled_train)

    def make_query(self, n_images=50):
        try:
            unlabeled_train_data = self.dataset.get_unlabeled_train_data()
        except ValueError:
            raise IndexError("No more unlabeled train samples")
        unlabeled_entry_ids, X_pool = unlabeled_train_data["entry_ids"], unlabeled_train_data["features"]
        del unlabeled_train_data
        if len(X_pool) > 0:
            # least confident and most representative of data
            start_time = time.time()
            uncertainties = np.max(self.model.predict_real(X_pool), axis=1)
            ask_id = self.get_most_uncertainty_dense(uncertainties, self.similarity_matrix, beta=1)
            self.delete_index_similarity_matrix(ask_id)
            print "Found new query amongst %d unlabeled samples in %.2f seconds" % (
                X_pool.shape[0], time.time() - start_time)
            # Make line. Project query_image on current decision boundary
            start_time = time.time()
            line, line_segment, intersection_point = self.make_line(X_pool[ask_id], n_images)
            print "Made line from found query in %.2f seconds" % (time.time() - start_time)
            if not self.human_experiment and self.generative_model is not None:
                # Change 1 to higher number for faster algorithm (less generating and plotting)
                if self.n_queries % 1 == 0:
                    start_time = time.time()
                    self.generate_images_line_save(line_segment, unlabeled_entry_ids[ask_id])
                    print "Plotted query line in %.2f seconds" % (time.time() - start_time)
            else:
                self.save_query_to_hdf5(
                    to_vector(self.dataset.data["features"][unlabeled_entry_ids[ask_id]]).T,
                    unlabeled_entry_ids[ask_id],
                    self.dataset.scaling_transformation.inverse_transform(line_segment),
                    self.dataset.scaling_transformation.inverse_transform(intersection_point.T)
                )
            self.n_queries += 1
            return unlabeled_entry_ids[ask_id], line, line_segment, None, intersection_point
        else:
            raise IndexError("No more unlabeled train samples")

    def delete_index_similarity_matrix(self, entry_index):
        self.similarity_matrix = np.delete(np.delete(self.similarity_matrix, entry_index, axis=0), entry_index, axis=1)

    def get_most_uncertainty_dense(self, uncertainties, similarities, beta):
        times = uncertainties * similarities.mean(axis=0) ** beta
        return np.argmax(times)


class ClusterCentroidsLine(UncertaintySamplingLine):
    def __init__(self, dataset, model, generative_model, batch_size, save_path_queries=None, human_experiment=False,
                 base_precision=None, **kwargs):
        super(ClusterCentroidsLine, self).__init__(dataset, model, generative_model,
                                                   save_path_queries=save_path_queries,
                                                   human_experiment=human_experiment, base_precision=base_precision,
                                                   **kwargs)
        self.model.train(self.dataset, first_time=True)
        self.save_path_queries = save_path_queries
        self.n_queries = 0
        self.r = None
        self.batch_size = batch_size
        self.clustered = None

    def make_query(self, n_images=50):
        try:
            unlabeled_train_data = self.dataset.get_unlabeled_train_data()
        except ValueError:
            raise IndexError("No more unlabeled train samples")
        unlabeled_entry_ids, X_pool = unlabeled_train_data["entry_ids"], unlabeled_train_data["features"]
        del unlabeled_train_data
        if len(X_pool) > 0:
            start_time = time.time()
            # Cluster centroids!
            if self.n_queries % self.batch_size == 0:
                # Cluster again!
                self.clustered = KMeans(n_clusters=self.batch_size).fit(X_pool)
            query_image = self.clustered.cluster_centers_[self.n_queries % self.batch_size]
            query_image_original_space = self.dataset._scaling_transformation.inverse_transform(
                to_vector(query_image).T).T  # (n_features, 1)
            print "Found new query using %d unlabeled clustered samples in %.2f seconds" % (
                X_pool.shape[0], time.time() - start_time)
            # Make line. Project query_image on current decision boundary
            start_time = time.time()
            line, line_segment, intersection_point = self.make_line(query_image, n_images)  # scaled space
            print "Made line from found query in %.2f seconds" % (time.time() - start_time)
            if not self.human_experiment and self.generative_model is not None:
                # Change 1 to higher number for faster algorithm (less generating and plotting)
                if self.n_queries % 1 == 0:
                    start_time = time.time()
                    self.generate_images_line_save(line_segment, None, image_original_space=query_image_original_space)
                    print "Plotted query line in %.2f seconds" % (time.time() - start_time)
            else:
                self.save_query_to_hdf5(
                    query_image_original_space.T,  # original space
                    -1,
                    self.dataset.scaling_transformation.inverse_transform(line_segment),
                    self.dataset.scaling_transformation.inverse_transform(intersection_point.T)
                )
            self.n_queries += 1
            return None, line, line_segment, query_image, intersection_point
        else:
            raise IndexError("No more unlabeled train samples")

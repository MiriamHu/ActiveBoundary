import numpy as np
from joint_optimisation import do_optimization
from utils import compute_intersection_line_decision_boundary, to_vector
import traceback
from interface import LabelInterface, PointLabelInterface

__author__ = 'mhuijser'


class Labeler(object):
    def __init__(self):
        pass

    def label(self, *args):
        raise NotImplementedError


class IdealLabeler(Labeler):
    """
    Provide the errorless/noiseless label to any feature vectors being queried.
    Parameters
    ----------
    dataset: Dataset object
        Dataset object with the ground-truth label for each sample.
    """

    def __init__(self, dataset, **kwargs):
        super(IdealLabeler, self).__init__()
        y = dataset.groundtruth_y  # Dataset in original data space
        # make sure the input dataset is fully labeled
        assert (np.array(y) != np.array(dataset.unlabeled_class)).all()
        self.y = y  # Ground truth labels
        self.dataset = dataset

    def label(self, feature, **kwargs):
        """
        Label feature (in original space)
        :param feature:
        :return:
        """
        try:
            found_id = np.where([np.array_equal(x, feature) for x in self.dataset.data["features"]])[0][0]
            label = self.y[found_id]
        except Exception as e:
            traceback.print_exc()
            raise e
        return label


class PointLabeler(Labeler):
    def __init__(self, dataset, pretrained_groundtruth, hyperparameters, **kwargs):
        super(PointLabeler, self).__init__()
        self.dataset = dataset
        self.lambda_r = hyperparameters["lambda_r"]
        self.gamma = hyperparameters["gamma"]
        self.weight_hinge = hyperparameters["weight_hinge"]
        self.learning_rate = hyperparameters["learning_rate"]
        self.decay = hyperparameters["decay"]
        self.interval = hyperparameters["interval"]
        self.factor = hyperparameters["factor"]
        self.max_epochs = hyperparameters["max_epochs"]
        self.batch_size = hyperparameters["batch_size"]
        self.no_class = self.dataset.unlabeled_class
        self.classes = np.unique(self.dataset.groundtruth_y)
        if pretrained_groundtruth is not None:
            self.w, self.b0 = pretrained_groundtruth["w"], pretrained_groundtruth["b0"]
        else:
            self.w, self.b0 = self.__train_groundtruth_model()
        self.decision_function = lambda x: self.w.T.dot(x) + self.b0
        self.trained = self.w is not None

    @property
    def groundtruth_model(self):
        return {"w": self.w, "b0": self.b0}

    def __train_groundtruth_model(self):
        print "Training ground truth model for labeler"
        X_original_space, labels = self.dataset.data["features"], self.dataset.groundtruth_y
        if self.no_class in labels:
            raise Exception("Model can only be trained on labeled data. Unlabeled data encountered!")
        X_scaled = self.dataset.scaling_transformation.transform(X_original_space)
        X_val_scaled, labels_val = self.dataset.validation_data_format_keras()
        y = np.copy(labels)
        y[y == min(self.classes)] = -1
        y[y == max(self.classes)] = 1

        y_val = None
        if labels_val is not None:
            y_val = np.copy(labels_val)
            y_val[y_val == min(self.classes)] = -1
            y_val[y_val == max(self.classes)] = 1

        history, (w, b) = do_optimization(X_scaled, y, None, None, Xj_val=X_val_scaled, yj_val=y_val,
                                          lambda_r=self.lambda_r,
                                          gamma=self.gamma, weight_hinge=self.weight_hinge,
                                          learning_rate=self.learning_rate, decay=self.decay,
                                          interval=self.interval, factor=self.factor, max_epochs=self.max_epochs,
                                          batch_size=self.batch_size)
        del X_scaled, y
        return w, float(b)

    def predict(self, sample):
        """

        :param sample: n_samples x n_features
        :return:
        """
        if not self.trained:
            raise Exception("Cannot predict with an untrained model.")
        scores = self.decision_function(sample.T).T
        scores[scores == 0] = self.no_class
        scores[scores > 0] = max(self.classes)
        scores[scores < 0] = min(self.classes)
        labels = scores
        # return labels.flatten()
        return labels

    def label(self, sample, sample_already_scaled=False, *args):
        # Transform from original data space to ground truth data space = scaled space.
        sample = to_vector(sample).T
        if not sample_already_scaled:
            sample = self.dataset.scaling_transformation.transform(sample)
        label = self.predict(sample)
        return label


class HumanPointLabeler(PointLabeler):
    def __init__(self, dataset, ALI_model, pretrained_groundtruth=None, hyperparameters=None, **kwargs):
        """
        Human oracle interface to label point queries (traditional active learning).
        GPU required because of image generation.
        :param dataset:
        :param ALI_model:
        :param pretrained_groundtruth:
        :param hyperparameters:
        :param kwargs:
        """
        super(HumanPointLabeler, self).__init__(dataset, pretrained_groundtruth=pretrained_groundtruth,
                                                hyperparameters=hyperparameters, **kwargs)
        self.ali = ALI_model

    def label(self, sample, sample_already_scaled=False, *args):
        # Transform from original data space to ground truth data space = scaled space.
        sample = to_vector(sample)
        if sample_already_scaled:
            sample = self.dataset.scaling_transformation.inverse_transform(sample.T).T
        point_query_image = self.ali.decode(sample.T)
        oracle = PointLabelInterface(point_query_image, list(self.dataset.classes),
                                     classes_dictionary=self.dataset.classes_dictionary)
        return oracle.label_point_query


class LineLabeler(PointLabeler):
    def __init__(self, dataset, pretrained_groundtruth=None, hyperparameters=None,
                 use_groundtruth_labels=False, **kwargs):
        """
        Automatic line labeler, uses a ground truth model to label.
        :param dataset: Dataset object with desired dataset loaded in.
        :param pretrained_groundtruth: optional, pass pretrained ground truth model to serve as labeler.
        If not provided, a ground truth model is trained with all ground truth labels.
        :param hyperparameters: hyperparameters (e.g. for optimization).
        :param use_groundtruth_labels: use ground truth labels to label query samples,
        instead of using ground truth model to classify.
        :param kwargs:
        """
        super(LineLabeler, self).__init__(dataset, pretrained_groundtruth, hyperparameters)
        self.use_groundtruth_labels = use_groundtruth_labels
        self.ideal_labeler = None
        if self.use_groundtruth_labels:
            self.ideal_labeler = IdealLabeler(dataset)

    def label(self, sample, line=None, line_segment=None, sample_already_scaled=False, intersection_point_cdb=None):
        """
        This function labels a query line with its decision boundary point (= intersection line with decision boundary)
        and the label of the query point from which the query line was created.
        :param sample: query point: (n_features, 1) in original dataspace
        :param line: query line in scaled data space
        :param line_segment: can be used to convert to human understandable line query. In scaled data space
        :return:
        """
        sample = to_vector(sample).T
        if self.ideal_labeler is None:
            if not sample_already_scaled:
                # Transform from original data space to ground truth data space = scaled space.
                sample = self.dataset.scaling_transformation.transform(sample)
            label = self.predict(sample)
        else:
            label = self.ideal_labeler.label(sample.squeeze())
        A, B = line(0), line(1)  # a + (b-a)*0 = a and a+(b-a)*1 = b
        db_point = compute_intersection_line_decision_boundary(A, B, self.w, self.b0)
        assert db_point.shape == A.shape
        return label, db_point


class HumanLineLabeler(LineLabeler):
    def __init__(self, dataset, ALI_model, hyperparameters=None, **kwargs):
        """
        Human oracle interface to label line queries (traditional active learning).
        GPU required because of image generation.
        :param dataset:
        :param ALI_model:
        :param hyperparameters:
        :param kwargs:
        """
        super(HumanLineLabeler, self).__init__(dataset, hyperparameters=hyperparameters, **kwargs)
        self.ali = ALI_model

    def label(self, sample, line=None, line_segment=None, sample_already_scaled=False, intersection_point_cdb=None):
        """
        NB ONLY HANDLES LINES FROM UNCERTAINTY STRATEGY (for clustercentroids, check if everything is in correct space!)
        :param sample: for uncertainty strategy in original ALI space
        :param line: lambda function in scaled space
        :param line_segment: scaled space
        :param sample_already_scaled:
        :return:
        """
        sample = to_vector(sample)  # original space if uncertainty strategy
        if sample_already_scaled:
            sample = self.dataset.scaling_transformation.inverse_transform(sample.T).T
        line_segment_original_space = self.dataset.scaling_transformation.inverse_transform(line_segment)
        line_images = self.generate_images(line_segment_original_space)
        point_query_image = self.generate_images(sample.T)
        intersection_point_cdb_original_space = self.dataset.scaling_transformation.inverse_transform(
            intersection_point_cdb.T).T
        oracle = LabelInterface(line_segment_original_space, line_images, point_query=sample,
                                point_query_image=point_query_image,
                                intersection_point_cdb=intersection_point_cdb_original_space,
                                classes=list(self.dataset.classes),
                                classes_dictionary=self.dataset.classes_dictionary)
        db_point_original_space = oracle.chosen_point
        if db_point_original_space is not None:
            db_point_scaled_space = self.dataset.scaling_transformation.transform(db_point_original_space.T).T
        else:
            db_point_scaled_space = None
        label_point_query = oracle.label_point_query
        print "Chosen label", label_point_query
        return label_point_query, db_point_scaled_space

    def generate_images(self, line_segment_original_space):
        """
        :param line_segment_original_space: in original space
        :return:
        """
        try:
            decoded_images = self.ali.decode(
                line_segment_original_space)  # Transform to original space, in which ALI operates.
        except Exception as e:
            print "EXCEPTION:", traceback.format_exc()
            raise e
        return decoded_images


class NoisyLineLabeler(LineLabeler):
    def __init__(self, dataset, std_noise, pretrained_groundtruth=None, hyperparameters=None,
                 use_groundtruth_labels=False, **kwargs):
        """
        :param dataset:
        :param std_noise:
        :param pretrained_groundtruth:
        :param hyperparameters:
        :param use_groundtruth_labels: use ground truth labels to label query samples,
        instead of using ground truth model to classify.
        :param kwargs:
        """
        super(NoisyLineLabeler, self).__init__(dataset, pretrained_groundtruth=pretrained_groundtruth,
                                               hyperparameters=hyperparameters,
                                               use_groundtruth_labels=use_groundtruth_labels, **kwargs)
        self.std_noise = std_noise

    def label(self, sample, line=None, line_segment=None, sample_already_scaled=False, intersection_point_cdb=None):
        label, db_point = super(NoisyLineLabeler, self).label(sample, line, line_segment,
                                                              sample_already_scaled=sample_already_scaled)
        a, b = line(0), line(1)  # a + (b-a)*0 = a and a+(b-a)*1 = b
        if self.std_noise == 0:
            return label, db_point
        noise = np.random.normal(0, self.std_noise)
        noisy_db_point = db_point + noise * (
            (b - a) / np.linalg.norm(b - a))
        return label, noisy_db_point

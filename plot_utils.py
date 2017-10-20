import numpy as np
import matplotlib.pyplot as plt

__author__ = 'mhuijser'

centered_in_old = True

"""
Some helper functions for visualizing the algorithm's workings in a 2D latent space. Useful for toy example.
"""

def plot_base(dataset, labeler, model, iteration, show=True):
    fig = plt.figure()
    fig.suptitle("Iteration %d" % iteration)
    ax = plt.subplot(111)
    plot_data_points(dataset.data["features"], dataset.data["targets"], dataset.mu, dataset.unlabeled_class)
    plot_intersection_points(dataset.db_points, dataset.mu)
    plot_groundtruth_decision_boundary(labeler.w, labeler.b0, dataset.mu)
    plot_estimated_decision_boundary(model.w, model.b, dataset.mu)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), numpoints=1)
    if show:
        plt.show()


def plot_after_query(line_segment, point_query, dataset, labeler, model, iteration):
    plot_base(dataset, labeler, model, iteration, show=False)
    plot_line_query_segment(line_segment, point_query, dataset.mu)
    plt.show()


def plot_after_label(line_segment, point_query, dataset, labeler, model, iteration):
    fig = plt.figure()
    fig.suptitle("Iteration %d" % iteration)
    ax = plt.subplot(111)
    plot_data_points(dataset.data["features"], dataset.data["targets"], dataset.mu, dataset.unlabeled_class)
    plot_intersection_points(dataset.db_points, dataset.mu)
    plot_groundtruth_decision_boundary(labeler.w, labeler.b0, dataset.mu)
    plot_estimated_decision_boundary(model.w, model.b, dataset.mu, old_mu=dataset._old_mu)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), numpoints=1)
    updated_line_segment = line_segment + dataset._old_mu - dataset.mu
    plot_line_query_segment(updated_line_segment, point_query, dataset.mu)
    plt.show()


def plot_data_points(X, y, mu, unlabeled_class, labeled_colour="g", unlabeled_marker="o", markers=("s", "^")):
    all_points = X
    if centered_in_old:
        all_points = X + mu
    plt.ylim(-10, 10)
    # plt.ylim(0,5)
    # plt.xlim([min(all_points[:,0]) - 1, max(all_points[:,0]) + 1])
    plt.xlim(-10, 10)

    classes = np.unique(y[y != unlabeled_class])
    for i in range(len(classes)):
        points = all_points[(y == classes[i]).flatten(), :]
        plt.plot(points[:, 0], points[:, 1], markers[i] + labeled_colour, label="Class " + str(int(classes[i])))
    unlabeled_points = all_points[(y == unlabeled_class).flatten(), :]
    plt.plot(unlabeled_points[:, 0], unlabeled_points[:, 1], unlabeled_marker + labeled_colour, alpha=0.5,
             label="Unlabeled")


def plot_estimated_decision_boundary(w, b, mu, colour="g", old_mu=None):
    a = (-w[0, 0] / w[1, 0])
    y = lambda x: a * x - b / w[1, 0]
    x = np.linspace(-10, 10)
    decision_boundary = np.array([x, y(x)]).T
    if old_mu is not None:
        decision_boundary = decision_boundary + old_mu - mu
    if centered_in_old:
        decision_boundary = decision_boundary + mu
    plt.plot(decision_boundary[:, 0], decision_boundary[:, 1], "-" + colour, label="Estimated decision boundary")


def plot_groundtruth_decision_boundary(w, b, mu, colour="k"):
    a = (-w[0, 0] / w[1, 0])
    y = lambda x: a * x - b / w[1, 0]
    x = np.linspace(-10, 10)
    decision_boundary = np.array([x, y(x)]).T
    if not centered_in_old:
        decision_boundary = decision_boundary - mu
    plt.plot(decision_boundary[:, 0], decision_boundary[:, 1], colour + '-', label="Ground truth decision boundary")


def plot_intersection_points(points, mu, colour="g", marker="x"):
    if points is not None:
        p = points
        if centered_in_old:
            p = points + mu
        plt.plot(p[:, 0], p[:, 1], colour + marker, label="Intersection point")


def plot_line_query_segment(line_segment, point_query, mu, colour="g"):
    """

    :param line_segment: shape (n_points, n_features)
    :return:
    """
    point = point_query
    line = line_segment
    if centered_in_old:
        # add mu to line_segment and point_query
        point = point_query + mu
        line = line_segment + mu
    plt.plot(point[0], point[1], 'o' + colour)
    plt.plot(line[:, 0], line[:, 1], '--' + colour)

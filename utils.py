import numpy as np
from fuel.datasets import H5PYDataset

__author__ = 'mhuijser'

def load_decision_boundary_from_hdf5(hdf5_file, index):
    data = H5PYDataset(hdf5_file,
                   which_sets=("data",),
                   sources=("w", "b"),
                   load_in_memory=True)
    w, b = data.data_sources
    return {"w":w[index,:].reshape((w.shape[1],1)), "b0":b[index]}

def load_unlabeled_indices_from_hdf5(hdf5_file):
    data = H5PYDataset(hdf5_file,
                   which_sets=("data",),
                   sources=("unlabeled_indices",),
                   load_in_memory=True)
    unlabeled_indices, = data.data_sources
    return unlabeled_indices.astype("int")

def compute_intersection_line_decision_boundary(A, B, w, b0):
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    w = w.astype(np.float32)
    b0 = np.float32(b0)
    t_intersection = np.float32((float(1)/w.T.dot(B-A))*(-b0-w.T.dot(A)))
    return A + (B-A)*t_intersection

def project_point_on_decision_boundary(w, b0, x):
    w = w.astype(np.float32)
    x = x.astype(np.float32)
    b0 = np.float32(b0)
    x_p = x - (((w.T.dot(x) + b0)[0,0] * w) / (w.T.dot(w))[0,0])
    print "Distance query point from boundary ", np.linalg.norm((((w.T.dot(x) + b0)[0,0] * w) / (w.T.dot(w))[0,0]))
    return x_p

def project_point_on_line(A, B, p):
    """
    Project point p on line = A + (B-A)t
    :param A:
    :param B:
    :param p: point which to project on line = A + (B-A)t
    :return:
    """
    ap = p.astype(np.float32) - A.astype(np.float32)
    ab = B.astype(np.float32) - A.astype(np.float32)
    return A.astype(np.float32) + ((ap.T.dot(ab) / ab.T.dot(ab)) * ab)

def make_line_segment(radius, mu, a, b, n_points_line):
    """
    This function makes a line segment using the radius of the sphere in which
    all the data fits.
    :param radius:
    :param mu:
    :param line:
    :return:
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mu = mu.astype(np.float32)
    minus_b = -2*(a.T.dot(b-a) + mu.T.dot(b-a))
    discriminant = ((2*(a.T.dot(b-a) + mu.T.dot(b-a)))**2 - 4*(b-a).T.dot(b-a)*(-1*radius**2 + a.T.dot(a) - 2*a.T.dot(mu) + mu.T.dot(mu)))**0.5
    denominator  = 2*(b-a).T.dot(b-a)
    minus_b = minus_b[0,0]
    discriminant = discriminant[0,0]
    denominator = denominator[0,0]
    t1 = (minus_b - discriminant)/denominator
    t2 = (minus_b + discriminant)/denominator
    line = lambda t: a + t*(b-a)
    return line(np.linspace(t1, t2, num=n_points_line).astype(np.float32)).T

def compute_radius_sphere(scaled_data):
    r = np.max(np.linalg.norm(scaled_data, axis=1))
    return r

def to_vector(x):
    try:
        if x.shape[1] == 1:
            return x
        elif x.shape[0] == 1 and x.shape[1] > 1:
                return x.reshape((x.shape[1], 1))
        else:
            raise TypeError("This is not a vector!")
    except IndexError:
        return x.reshape((len(x),1))
from blocks.serialization import load
from ali.bricks import ALI
import theano
from theano import tensor
import numpy as np

__author__ = 'mhuijser'


class GenerativeALIModel(object):
    def __init__(self, main_loop_path):
        self.ali = self._load_model(main_loop_path)
        assert (isinstance(self.ali, ALI))

    def _load_model(self, main_loop_path):
        with open(main_loop_path, 'rb') as src:
            main_loop = load(src, use_cpickle=True)
        ali, = main_loop.model.top_bricks
        return ali

    def encode(self, x):
        """

        :param x: (n_samples, n_channels, height, width)
        :return:
        """
        xs = tensor.tensor4('features')
        encode = theano.function([xs], self.ali.encoder.apply(xs))
        return encode(x)

    def decode(self, z):
        """

        :param z: of shape (n_dim,) or (n_dim, 1) or (n_samples, n_dim)
        :return:
        """
        try:
            z.shape[1]
        except IndexError:
            z = z.reshape(1, z.shape[0])
        if z.shape[1] == 1 and z.shape[0] > 1:
            z = np.transpose(z)
        z = z.reshape(z.shape[0], z.shape[1], 1, 1)
        zs = tensor.tensor4('z')
        decode = theano.function([zs], self.ali.decoder.apply(zs))
        return decode(z)

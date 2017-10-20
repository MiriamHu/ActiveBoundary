from keras import backend as K
from keras.objectives import mean_squared_error
__author__ = 'mhuijser'

def my_accuracy(y_true, y_pred):
    y_pred = K.sign(y_pred)
    return K.mean(K.equal(y_true, y_pred))

def gmse_factory(gamma):
    def gamma_mse(y_true, y_pred):
        return gamma * mean_squared_error(y_true, y_pred)
    return gamma_mse
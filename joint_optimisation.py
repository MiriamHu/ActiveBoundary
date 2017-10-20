import time
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from callbacks import LR_scheduler_factory
import numpy as np
import warnings
from objectives import my_accuracy

__author__ = 'mhuijser'


class WeightSharing(Callback):
    def __init__(self, bp, Xii, yii, history_bp, verbose=0):
        self.bp = bp  # boundary points model
        self.Xii = Xii
        self.yii = yii
        self.history_bp = history_bp
        self.epoch_loss = []
        self.verbose = verbose
        super(WeightSharing, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        self.bp.set_weights(self.model.get_weights())
        history = self.bp.fit(self.Xii, self.yii, batch_size=min(2, min(self.Xii.shape[0], 32)), nb_epoch=2, verbose=0)
        self.model.set_weights(self.bp.get_weights())
        mean_mse = history.history["loss"]
        if self.verbose:
            print "\n\nMSE on epoch end", mean_mse
            print ""
        self.history_bp["gamma_mse"].append(history.history["loss"])


class WeightsEarlyStopping(Callback):
    def __init__(self, monitor, patience=0, threshold_value=1e-3, verbose=0):
        super(WeightsEarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.threshold_value = threshold_value
        self.verbose = verbose
        self.monitor_op = np.less

    def on_train_begin(self, logs={}):
        self.previous_weights = self.model.get_weights()
        self.wait = 0  # Allow instances to be re-used

    def on_epoch_end(self, epoch, logs={}):
        current = self.monitor(self.previous_weights, self.model.get_weights())
        self.previous_weights = self.model.get_weights()
        if current is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if np.less(current, self.threshold_value):
            if current == 0:
                self.model.stop_training = True
                if self.verbose > 0:
                    print('Epoch %05d: early stopping: ratio weights = 0' % (epoch))
            elif self.wait >= self.patience:
                if self.verbose > 0:
                    print('Epoch %05d: early stopping: ratio weights below %.4f' % (epoch, self.threshold_value))
                self.model.stop_training = True
            self.wait += 1
        else:
            self.wait = 0

def create_joint_model(input_dim, init_w, init_b, gamma, weight_hinge, learning_rate, decay, regulariser=None):
    image_input = Input(shape=(input_dim,), dtype='float32', name='image_input')
    db_input = Input(shape=(input_dim,), dtype='float32', name="db_input")
    shared_layer = Dense(1, input_dim=input_dim, W_regularizer=regulariser,
                         init='uniform', activation="linear", bias=True, name='shared_layer')
    _ = shared_layer(image_input)
    _ = shared_layer(db_input)
    model = Model(input=[image_input, db_input], output=[shared_layer.get_output_at(0), shared_layer.get_output_at(1)])
    adam = Adam(lr=learning_rate)  # SGD should also work because convex loss function, but Adam converges faster.
    model.compile(optimizer=adam, loss=['hinge', 'mse'], loss_weights=[weight_hinge, gamma],
                  metrics=[my_accuracy, 'mse'])
    return model

def create_single_model(input_dim, init_w, init_b, learning_rate, decay, regulariser=None):
    svm_model = Sequential()
    svm_model.add(
        Dense(1, input_dim=input_dim, W_regularizer=regulariser, init='uniform', activation="linear", bias=True))
    if init_w is not None and init_b is not None:
        svm_model.set_weights([init_w, np.array([init_b])])
    adam = Adam(lr=learning_rate)  # SGD also option.
    svm_model.compile(loss="hinge", optimizer=adam, metrics=["hinge", my_accuracy])
    return svm_model

def make_Xi_bigger(new_size, Xi):
    add_number = new_size - Xi.shape[0]
    return np.pad(Xi, ((add_number, 0), (0, 0)), mode='mean'), np.zeros((new_size, 1))

def make_Xj_bigger(new_size, Xj, yj):
    add_number = new_size - Xj.shape[0]
    return np.pad(Xj, ((add_number, 0), (0, 0)), mode='wrap'), np.pad(yj, ((add_number, 0), (0, 0)), mode='wrap')

def do_optimization(Xj, yj, Xi, yi, lambda_r, gamma, weight_hinge, learning_rate, decay, interval, factor, max_epochs,
                    batch_size,
                    init_w=None, init_b=None, Xj_val=None, yj_val=None, regulariser='l2', verbose=True):
    input_dim = Xj.shape[1]

    lr = learning_rate
    joint_optimisation = True
    quantity_to_monitor = "loss"
    if Xi is None or yi is None:
        joint_optimisation = False
    # Xj and Xi need to be the same size
    elif Xj.shape[0] > Xi.shape[0]:
        Xi, yi = make_Xi_bigger(Xj.shape[0], Xi)
    elif Xj.shape[0] < Xi.shape[0]:
        Xj, yj = make_Xj_bigger(Xi.shape[0], Xj, yj)
    if regulariser == 'l2':
        regulariser = l2(lambda_r)

    batch_size = Xj.shape[0]

    if verbose:
        verbose = 1
    elif not verbose:
        verbose = 0
    if joint_optimisation:
        print "Joint optimisation..."
        svm_model = create_joint_model(input_dim, init_w, init_b, gamma, weight_hinge, lr, decay,
                                       regulariser=regulariser)

        start_time = time.time()
        history = svm_model.fit([Xj, Xi], [yj, yi], callbacks=[
            EarlyStopping(monitor=quantity_to_monitor, patience=10, verbose=verbose, mode="min"),
            ReduceLROnPlateau(monitor=quantity_to_monitor, factor=0.2, patience=3, cooldown=1, min_lr=0.0000001,
                              mode='min'),
            LR_scheduler_factory(lr, interval=interval, factor=factor)],
                                batch_size=batch_size, nb_epoch=max_epochs, verbose=verbose)
        print "Time taken", time.time() - start_time
        all_history = history.history
        return all_history, svm_model.get_weights()
    else:
        print "Single optimisation..."
        svm_model = create_single_model(input_dim, init_w, init_b, lr, decay)
        start_time = time.time()
        history = svm_model.fit(Xj, yj,
                                callbacks=[EarlyStopping(monitor=quantity_to_monitor, patience=10, verbose=verbose,
                                                         mode="min"),
                                           ReduceLROnPlateau(monitor=quantity_to_monitor, factor=0.2, patience=3,
                                                             cooldown=1, min_lr=0.0000001, mode="min"),
                                           LR_scheduler_factory(lr, interval=interval, factor=factor)],
                                batch_size=batch_size, nb_epoch=max_epochs, verbose=verbose)
        print "Time taken", time.time() - start_time
        all_history = history.history
        return all_history, svm_model.get_weights()

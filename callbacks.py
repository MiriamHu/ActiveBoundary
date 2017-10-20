import keras

__author__ = 'mhuijser'

def LR_scheduler_factory(initial_lr, interval=2, factor=0.75):
    """
    :param initial_lr: Initial learning rate
    :param interval: change learning rate every n epochs
    :param factor: change lr by multiplying with this factor
    :return:
    """

    # half learning rate every 10 epochs
    def scheduler(epochs):
        decay_times = int(epochs / interval)
        lr = initial_lr * (factor ** decay_times)
        print("Learning Rate is now %0.15f" % lr)
        return lr

    return keras.callbacks.LearningRateScheduler(schedule=scheduler)
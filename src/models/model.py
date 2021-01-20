from abc import ABCMeta
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import SGD
import h5py
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_WINDOW_SIZE = 8
DEFAULT_EMBEDDING_SIZE = 300
DEFAULT_NUM_EPOCHS = 250
DEFAULT_STEPS_PER_EPOCH = 10000

CUST_EMBEDDINGS_LAYER_NAME = 'cust_embeddings'


class Cust2VecModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, window_size, num_prods, num_custs,
                 embedding_size=DEFAULT_EMBEDDING_SIZE,
                 cust_embeddings_layer_name=CUST_EMBEDDINGS_LAYER_NAME):
        self._window_size = window_size
        self._num_prods = num_prods
        self._num_custs = num_custs

        self._embedding_size = embedding_size
        self._cust_embeddings_layer_name = cust_embeddings_layer_name

        self._model = None

    @property
    def cust_embeddings(self):
        return _cust_embeddings_from_model(self._model)

    def build(self):
        raise NotImplementedError()

    def compile(self, optimizer=None):
        if not optimizer:
            optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)

        self._model.compile(optimizer=optimizer,
                            loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'])

    def train(self, generator,
              steps_per_epoch=DEFAULT_STEPS_PER_EPOCH,
              epochs=DEFAULT_NUM_EPOCHS,
              early_stopping_patience=None,
              save_path=None, save_period=None,
              save_cust_embeddings_path=None, save_cust_embeddings_period=None):

        callbacks = []
        if early_stopping_patience:
            callbacks.append(EarlyStopping(monitor='loss',
                                           patience=early_stopping_patience))
        if save_path and save_period:
            callbacks.append(ModelCheckpoint(save_path,
                                             period=save_period))
        if save_cust_embeddings_path and save_cust_embeddings_period:
            callbacks.append(_SaveCustEmbeddings(save_cust_embeddings_path,
                                                 save_cust_embeddings_period))

        history = self._model.fit_generator(
            generator,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs)

        return history

    def save(self, path):
        logger.info('Saving model to %s', path)
        self._model.save(path)

    def save_cust_embeddings(self, path):
        _write_cust_embeddings(self.cust_embeddings, path)

    def load(self, path):
        logger.info('Loading model from %s', path)
        self._model = load_model(path)


class _SaveCustEmbeddings(Callback):

    def __init__(self, path, period):
        self.path = path
        self.period = period

    def on_epoch_end(self, epoch):
        if epoch % self.period != 0:
            return

        path = self.path.format(epoch=epoch)
        embeddings = _cust_embeddings_from_model(self.model)
        _write_cust_embeddings(embeddings, path)


def _cust_embeddings_from_model(keras_model):
    for layer in keras_model.layers:
        if layer.get_config()['name'] == CUST_EMBEDDINGS_LAYER_NAME:
            return layer.get_weights()[0]


def _write_cust_embeddings(cust_embeddings, path):
    logger.info('Saving cust embeddings to %s', path)
    with h5py.File(path, 'w') as f:
        f.create_dataset(CUST_EMBEDDINGS_LAYER_NAME, data=cust_embeddings)

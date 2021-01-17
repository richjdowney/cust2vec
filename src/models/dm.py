from keras.layers import Average, Concatenate, Dense, Embedding, Input, Lambda
from keras.models import Model
from src.models.model import Cust2VecModel
from src.models import lambdas


class DM(Cust2VecModel):
    def build(self):
        sequence_input = Input(shape=(self._window_size,))
        cust_input = Input(shape=(1,))

        embedded_sequence = Embedding(
            input_dim=self._num_prods,
            output_dim=self._embedding_size,
            input_length=self._window_size,
            name=self._cust_embeddings_layer_name,
        )(sequence_input)
        embedded_cust = Embedding(
            input_dim=self._num_custs, output_dim=self._embedding_size, input_length=1
        )(cust_input)

        embedded = Concatenate(axis=1)([embedded_cust, embedded_sequence])
        split = Lambda(lambdas.split(self._window_size))(embedded)
        averaged = Average()(split)
        squeezed = Lambda(lambdas.squeeze(axis=1))(averaged)
        softmax = Dense(self._num_prods, activation="softmax")(squeezed)

        self._model = Model(inputs=[cust_input, sequence_input], outputs=softmax)


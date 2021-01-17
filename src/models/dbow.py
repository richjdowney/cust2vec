from keras.layers import Dense, Embedding, Input, Lambda
from keras.models import Model

from src.models.model import Cust2VecModel
from src.models import lambdas


class DBOW(Cust2VecModel):

    def build(self):
        cust_input = Input(shape=(1,))

        embedded_cust = Embedding(input_dim=self._num_custs,
                                  output_dim=self._embedding_size,
                                  input_length=1,
                                  name=self._cust_embeddings_layer_name)(cust_input)

        embedded_cust = Lambda(lambdas.squeeze(axis=1))(embedded_cust)

        stack = Lambda(lambdas.stack(self._window_size))(embedded_cust)

        softmax = Dense(self._num_prods, activation='softmax')(stack)

        self._model = Model(inputs=cust_input, outputs=softmax)

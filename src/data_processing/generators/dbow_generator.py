import itertools
import random

from keras.utils import to_categorical
import numpy as np


def data_generator(
    cust_index_list: list, all_cust_data: list, window_size: int, num_prods: int
) -> tuple:
    """Function to generate data for modelling (customer index, context products)

        Parameters
        ----------
        cust_index_list : list
            List of customer indices
        all_cust_data : list
            Lists of items purchased by each customer
        window_size : int
            The size of the window to use for the context
        num_prods : int
            The number of products being modelled

        """

    for cust in itertools.cycle(cust_index_list):
        prods = all_cust_data[cust]
        cust_num_prods = len(prods)

        if cust_num_prods <= window_size:
            continue

        target_idx = random.randint(0, (cust_num_prods - window_size) - 1)

        context_window = prods[target_idx: target_idx + window_size]

        yield (cust, to_categorical(context_window, num_classes=num_prods))


def batch(data, batch_size=32):
    while True:
        batch = itertools.islice(data, batch_size)

        x = []
        y = []

        for item in batch:
            cust, context_window = item

            x.append(cust)
            y.append(context_window)

        yield np.array(x), np.array(y)

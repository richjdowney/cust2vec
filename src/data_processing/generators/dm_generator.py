import random
import itertools
import numpy as np
from keras.utils import to_categorical


def data_generator(
    cust_index_list: list, all_cust_data: list, window_size: int, num_prods: int
) -> tuple:
    """Function to generate data for modelling (customer index, context products, target product)

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

    assert window_size % 2 == 0, "window_size must be even"

    offset = window_size // 2

    for cust in itertools.cycle(cust_index_list):
        prods = all_cust_data[cust]
        cust_num_prods = len(prods)

        if cust_num_prods <= window_size:
            continue

        # Select a random product as the target
        target_idx = random.randint(offset, (cust_num_prods - offset) - 1)
        target_id = prods[target_idx]

        # Get the context (all products within the window around the target product)
        context_window = (
            prods[target_idx - offset : target_idx]
            + prods[target_idx + 1 : target_idx + offset + 1]
        )

        yield (cust, context_window, to_categorical(target_id, num_classes=num_prods))


def batch(data, batch_size=32) -> tuple:
    """Generator to produce batches for modelling

        Parameters
        ----------
        data : function
            Function to generate the tuple
        batch_size : int
            Batch size to use for modelling

        """
    while True:
        batch = itertools.islice(data, batch_size)

        x_1 = []
        x_2 = []
        y = []

        for item in batch:
            cust, context_window, target_ids = item

            x_1.append(cust)
            x_2.append(context_window)
            y.append(target_ids)

        yield [np.array(x_1), np.array(x_2)], np.array(y)


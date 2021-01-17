import sys

sys.path.insert(1, "/home/ubuntu/cust2vec")

import boto3
import pickle
from utils.logging_framework import log
from pyspark.sql import SparkSession
from keras.utils.vis_utils import plot_model
from src.models import dm, dbow
from src.data_processing.generators import dm_generator as batch_dm
from src.data_processing.generators import dbow_generator as batch_dbow


if __name__ == "__main__":

    task = sys.argv[1]
    bucket = sys.argv[3]
    model = sys.argv[4]
    window_size = sys.argv[5]
    embedding_size = sys.argv[6]
    num_epochs = sys.argv[7]
    steps_per_epoch = sys.argv[8]
    early_stopping_patience = sys.argv[9]
    save_period = sys.argv[10]
    save_path = sys.argv[11]
    save_cust_embeddings = sys.argv[12]
    save_cust_embeddings_period = sys.argv[13]

    log.info("Running task {}".format(task))

    spark = SparkSession.builder.appName("cust2vec").getOrCreate()

    # ========== Download data and dictionaries ==========

    log.info("Download data and dictionaries from s3")
    s3 = boto3.resource("s3")

    with open("all_cust_data.txt", "wb") as data:
        s3.download_fileobj(bucket, "all_cust_data.txt", data)
    with open("all_cust_data.txt", "rb") as data:
        all_cust_data = pickle.load(data)

    # Customer index list (lists of customers in same order as all_cust_data)
    with open("cust_index_list.txt", "wb") as data:
        s3.download_fileobj(bucket, "cust_index_list.txt", data)
    with open("cust_index_list.txt", "rb") as data:
        cust_index_list = pickle.load(data)

    # Product dictionary mapping
    with open("prod_dictionary.pkl", "wb") as data:
        s3.download_fileobj(bucket, "prod_dictionary.pkl", data)
    with open("prod_dictionary.pkl", "rb") as data:
        prod_dictionary = pickle.load(data)

    # Reversed product dictionary mapping
    with open("reversed_prod_dictionary.pkl", "wb") as data:
        s3.download_fileobj(bucket, "reversed_prod_dictionary.pkl", data)
    with open("reversed_prod_dictionary.pkl", "rb") as data:
        reversed_prod_dictionary = pickle.load(data)

    # Customer dictionary mapping
    with open("cust_dictionary.pkl", "wb") as data:
        s3.download_fileobj(bucket, "cust_dictionary.pkl", data)
    with open("cust_dictionary.pkl", "rb") as data:
        cust_dictionary = pickle.load(data)

    # Reversed customer dictionary mapping
    with open("reversed_cust_dictionary.pkl", "wb") as data:
        s3.download_fileobj(bucket, "reversed_cust_dictionary.pkl", data)
    with open("reversed_cust_dictionary.pkl", "rb") as data:
        reversed_cust_dictionary = pickle.load(data)

    num_custs = len(cust_index_list)
    num_prods = len(prod_dictionary)

    MODEL_TYPES = {
        "dm": (dm.DM, batch_dm.data_generator, batch_dm.batch),
        "dbow": (dbow.DBOW, batch_dbow.data_generator, batch_dbow.batch),
    }

    model_class, data_generator, batcher = MODEL_TYPES[model]

    m = model_class(window_size, num_prods, num_custs, embedding_size=embedding_size)

    m.build()
    m.compile()

    # Plot model
    plot_model(
        m._model,
        to_file="model_plot_{}.png".format(model),
        show_shapes=True,
        show_layer_names=True,
    )

    all_data = batcher(
        data_generator(cust_index_list, all_cust_data, window_size, num_prods)
    )

    history = m.train(
        all_data,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        early_stopping_patience=early_stopping_patience,
        save_path=save_path,
        save_period=save_period,
        save_cust_embeddings_path=save_cust_embeddings,
        save_cust_embeddings_period=save_cust_embeddings_period,
    )

    elapsed_epochs = len(history.history["loss"])

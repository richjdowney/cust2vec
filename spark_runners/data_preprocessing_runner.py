import sys

sys.path.insert(1, "/home/ubuntu/cust2vec")

import os
from pyspark.sql import SparkSession
from src.data_processing.data_pre_processing import *
from utils.logging_framework import log

if __name__ == "__main__":

    task = sys.argv[1]
    bucket = sys.argv[2]
    staging_path = sys.argv[3]
    sample = sys.argv[5]
    sample_rate = float(sys.argv[6])
    num_prods = int(sys.argv[7])

    log.info("Running task {}".format(task))

    spark = SparkSession.builder.appName("cust2vec").getOrCreate()

    # ========== Import transaction data from staging ==========
    staging_trans_path = os.path.join(staging_path, "trans-data/")
    log.info("Importing transaction staging data from")
    trans_df = spark.read.parquet(staging_trans_path)

    # ========== Sample DataFrame if requested ==========

    if sample == "True":
        log.info(
            "Sampling transaction DataFrame with sample rate {}".format(sample_rate)
        )

        trans_df = sample_custs(trans_df, sample_rate)

    # ========== Create the sequence lists needed for the skipgrams algorithm ==========
    log.info("Generating lists for skipgrams algorithm")
    trans_df = trans_df.orderBy("CUST_CODE")
    prod_code_list, prod_code_group_list, cust_code_list = create_prod_lists(trans_df)

    # ========== Create dictionaries and convert PROD_CODE and CUST_CODE to indices ==========
    log.info("Creating dictionaries and converting PROD_CODE and CUST_CODE to indices")
    all_cust_data, prod_dictionary, reversed_prod_dictionary, cust_dictionary, reversed_cust_dictionary, cust_index_list = create_data(
        prod_code_list,
        prod_code_group_list,
        num_prods=num_prods,
        cust_list=cust_code_list,
    )

    # ========== Upload training data to s3 ==========
    log.info("Uploading training data, product and customer index mappings to s3")

    training_data_to_s3(obj=all_cust_data, bucket=bucket, key="all_cust_data.txt")
    training_data_to_s3(obj=cust_index_list, bucket=bucket, key="cust_index_list.txt")
    training_data_to_s3(obj=prod_dictionary, bucket=bucket, key="prod_dictionary.pkl")
    training_data_to_s3(
        obj=reversed_prod_dictionary, bucket=bucket, key="reversed_prod_dictionary.pkl"
    )
    training_data_to_s3(obj=cust_dictionary, bucket=bucket, key="cust_dictionary.pkl")
    training_data_to_s3(
        obj=reversed_cust_dictionary, bucket=bucket, key="reversed_cust_dictionary.pkl"
    )

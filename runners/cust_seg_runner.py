import sys

sys.path.insert(1, "/home/ubuntu/cust2vec")

import h5py
import pandas as pd
import boto3
import pickle
import io
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from utils.logging_framework import log
from src.models.kmeans import *


if __name__ == "__main__":

    task = sys.argv[1]
    bucket = sys.argv[2]
    create_cost_plot = bool(sys.argv[3])
    scored_kmeans_path = sys.argv[4]
    saved_kmeans_model_path = sys.argv[5]
    num_clusts = int(sys.argv[6])

    spark = SparkSession.builder.appName("cust2vec").getOrCreate()

    # ========== Load the customer embeddings ==========
    log.info("Loading customer embeddings")
    s3 = boto3.client("s3")
    with open("cust_embeddings.hdf5", "wb") as data:
        s3.download_fileobj(bucket, "cust_embeddings.hdf5", data)

    with h5py.File("cust_embeddings.hdf5", "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        cust_embeddings = list(f[a_group_key])

    log.info("Example of customer embedding...")
    print(cust_embeddings[0])

    # ========== Map embeddings to CUST_CODE ==========

    log.info("Loading customer index list")
    s3 = boto3.client("s3")

    # Customer index list (lists of customers in same order as all_cust_data)
    with open("cust_index_list.txt", "wb") as data:
        s3.download_fileobj(bucket, "cust_index_list.txt", data)
    with open("cust_index_list.txt", "rb") as data:
        cust_index_list = pickle.load(data)
    cust_index_pd = pd.DataFrame(cust_index_list, columns=["index"])

    log.info("Loading index to customer ID mapping dictionary")

    s3 = boto3.resource("s3")
    with open("reversed_cust_dictionary.pkl", "wb") as data:
        s3.Bucket(bucket).download_fileobj("reversed_cust_dictionary.pkl", data)
    with open("reversed_cust_dictionary.pkl", "rb") as data:
        reversed_dictionary = pickle.load(data)

    cust_embeddings_pd = pd.DataFrame(cust_embeddings)
    # create column headers
    cols = ["embedd_" + str(sub) for sub in cust_embeddings_pd.columns]
    cust_embeddings_pd.columns = cols
    cust_embeddings_pd.loc[:, "index"] = cust_index_pd["index"]
    cust_embeddings_pd.loc[:, "CUST_CODE"] = cust_index_pd["index"].map(
        reversed_dictionary
    )

    # Convert to Spark DataFrame
    cust_embeddings_df = spark.createDataFrame(cust_embeddings_pd)

    # ========== Prep data for k-means ==========
    feature_cols = [elem for elem in cust_embeddings_df.columns if "embedd" in elem]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    kmeans_input = assembler.transform(cust_embeddings_df)
    kmeans_input = kmeans_input.select("CUST_CODE", "features")

    # ========== Create cost plot ==========
    if create_cost_plot:
        log.info("Creating k-means cost plot")
        cost_plot = kmeans_compute_cost(
            df=kmeans_input, num_clusters=25, feature_col="features"
        )

        # Upload plot to s3
        img_data = io.BytesIO()
        cost_plot.savefig(img_data, format="png")
        img_data.seek(0)

        bucket = s3.Bucket(bucket)
        bucket.put_object(Body=img_data, ContentType="image/png", Key="cost_plot.png")

    # ========== Fit K-Means model ==========
    model = fit_kmeans(
        df=kmeans_input,
        num_clusters=num_clusts,
        feature_column="features",
        save_path=saved_kmeans_model_path,
    )

    scored_kmeans_df = score_kmeans(
        kmeans_input, model=model, save_path=scored_kmeans_path
    )

    scored_kmeans_df.show()


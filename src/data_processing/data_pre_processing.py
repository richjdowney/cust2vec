import collections
import boto3
import pickle
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def sample_custs(df: DataFrame, sample_rate: float) -> DataFrame:
    """Function to sample customers for modelling

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        name of the DataFrame with customer ID to sample
    sample_rate : float
        percentage of the customers to sample

    Returns
    -------
    df : pyspark.sql.DataFrame
       sampled DataFrame

    """

    cust_samp = df.select("CUST_CODE").dropDuplicates()
    cust_samp = cust_samp.sample(withReplacement=False, fraction=sample_rate, seed=42)
    df = df.join(cust_samp, "CUST_CODE", how="inner")

    return df


def create_prod_lists(df: DataFrame):
    """ Function to create list of all products purchased by all customers and an array of lists containing all
        products purchased by EACH customer

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            transaction DataFrame

        Returns
        -------
        prod_code_list : list
            list of unique PROD_CODE purchased by every customer as a single list
        prod_code_group_list : list
            list containing the unique PROD_CODE grouped by customer
        cust_code_list : list
            list containing the customers for every basket

        """

    prod_code_list = df.select("PROD_CODE").rdd.flatMap(lambda x: x).collect()

    prod_code_set = df.groupBy("CUST_CODE").agg(F.collect_set("PROD_CODE").alias("PROD_CODE_SET"))
    prod_code_group_list = prod_code_set.select("PROD_CODE_SET").rdd.flatMap(lambda x: x).collect()

    cust_codes = df.select("CUST_CODE").dropDuplicates()
    cust_code_list = cust_codes.rdd.flatMap(lambda x: x).collect()

    return prod_code_list, prod_code_group_list, cust_code_list


def create_data(prod_list: list, prod_group_list: list, num_prods: int, cust_list: list) -> tuple:
    """ Function to create counts of products, a dictionary mapping between PROD_CODE and an index,
        a reversed dictionary that maps back index to PROD_CODE, the customer data with PROD_CODE
        mapped to the index and a dictionary and reversed dictionary mapping between CUST_CODE, an index
        and back

        Parameters
        ----------
        prod_list : list
            list of unique PROD_CODE purchased in by every customer as a single list
        prod_group_list : list
            list containing the unique PROD_CODE purchased grouped by customer
        num_prods : int
            the number of products on which to train the embeddings e.g. top X products
            (all others are tagged as "UNK" (unknown))
        cust_list : list
            list of unique customers

        Returns
        -------
        all_cust_data : array
           array of lists containing the index of the PROD_CODE purchased by each customer
        prod_dictionary : dict
            dictionary containing the mapping of PROD_CODE to index
        reversed_prod_dictionary : dict
            dictionary containing the reverse mapping of index to PROD_CODE
        cust_dictionary : dict
            dictionary containing the mapping of CUST_CODE to index
        reversed_cust_dictionary : dict
            dictionary containing the reverse mapping of index to CUST_CODE
        all_cust_data : list
            list containing the index of the customers

        """

    # Create counts of products
    count = [["UNK", -1]]  # Placeholder for unknown
    count.extend(collections.Counter(prod_list).most_common(num_prods - 1))

    # Create a dictionary mapping of product to index
    prod_dictionary = dict()
    for prod, _ in count:
        prod_dictionary[prod] = len(prod_dictionary)

    # Create a reversed mapping of index to product
    reversed_prod_dictionary = dict(
        zip(prod_dictionary.values(), prod_dictionary.keys())
    )

    # Get counts for unknown products and map the product index from the dictionary
    # to each product for each customer
    unk_count = 0
    all_cust_data = list()
    for i in range(0, len(prod_group_list)):
        cust_prod_list = list()
        for prod in prod_group_list[i]:
            if prod in prod_dictionary:
                index = prod_dictionary[prod]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            cust_prod_list.append(index)
        all_cust_data.append(cust_prod_list)
    count[0][1] = unk_count

    # Create customer index dictionary
    cust_count = collections.Counter(cust_list)

    # Map the index to the customer list
    cust_dictionary = dict()
    for cust in cust_count:
        cust_dictionary[cust] = len(cust_dictionary)

    reversed_cust_dictionary = dict(
        zip(cust_dictionary.values(), cust_dictionary.keys())
    )

    cust_index_list = list(map(cust_dictionary.get, cust_list))

    return (
        all_cust_data,
        prod_dictionary,
        reversed_prod_dictionary,
        cust_dictionary,
        reversed_cust_dictionary,
        cust_index_list
    )


def training_data_to_s3(obj: any, bucket: str, key: str):
    """ Function to upload the training data to s3 - required for Sagemaker to access the files for training

        Parameters
        ----------
        obj : list, np.ndarray, dict
            object to upload, either a list, numpy array or dict
        bucket : str
            name of the s3 bucket
        key : str
            name of the file to upload

        """

    bucket = bucket
    key = key
    s3c = boto3.client("s3")

    if isinstance(obj, list):
        with open(key, "wb") as obj_pickle:
            pickle.dump(obj, obj_pickle)
        s3c.upload_file(key, bucket, key)

    if isinstance(obj, dict):
        with open(key, "wb") as obj_pickle:
            pickle.dump(obj, obj_pickle)
        s3c.upload_file(key, bucket, key)

    if isinstance(obj, np.ndarray):
        np.savetxt(key, obj, delimiter=",")
        s3c.upload_file(key, bucket, key)

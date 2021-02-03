import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


def kmeans_compute_cost(df, num_clusters, feature_col):
    """Function to compute the k-means cost as defined by the sum of squared distances between each observation
    and the cluster center

    Parameters
    ----------
    df : pyspark.sql.dataframe.DataFrame
      DataFrame containing the feature vector for the k-means, must contain a column of type udt containing the feature vector
    num_clusters : int
      The number of clusters to fit and calculate the cost
    feature_col : str
      Name of the column containing the feature vector

    Returns
    -------
    plt : matplotlib.pyplot
      Chart of cost over number of clusters

    """

    cost = np.zeros(num_clusters + 1)
    evaluator = ClusteringEvaluator()
    for k in range(2, num_clusters + 1):
        kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol(feature_col)
        model = kmeans.fit(df.sample(False, 0.5, seed=1))
        predictions = model.transform(df)
        cost[k] = evaluator.evaluate(predictions)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(range(2, num_clusters + 1), cost[2: num_clusters + 1])
    ax.set_xlabel("k")
    ax.set_ylabel("cost")

    return plt


def fit_kmeans(df, num_clusters, feature_column, save_path=None):
    """Function to fit k-means model

    Parameters
    ----------
    df : pyspark.sql.dataframe.DataFrame
      DataFrame containing the feature vector for the k-means, must contain a column of type udt containing the feature vector
    num_clusters : int
      The number of clusters to fit and calculate the cost
    feature_col : str
      Name of the column containing the feature vector
    save_path : str
      Path to save the fitted model

    Returns
    -------
    model : pyspark.ml.clustering.KMeans
      Fitted k-means model

    """

    k = num_clusters
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol(feature_column)
    model = kmeans.fit(df)

    if save_path:
        print("Saving fitted k-means model to {}".format(save_path))
        model.write().overwrite().save(save_path)

    return model


def score_kmeans(df, model, save_path=None):
    """Function to score transaction file with fitted k-means model

    Parameters
    ----------
    df : pyspark.sql.dataframe.DataFrame
      DataFrame containing the feature vector for the k-means, must contain a column of type udt containing the feature vector
    model : pyspark.ml.clustering.KMeans
      Fitted k-means model
    save_path : str
      Path to save the scored DataFrame

    Returns
    -------
    df_scored : pyspark.sql.dataframe.DataFrame
      DataFrame containing the customer ID (CUST_CODE) and the assigned cluster

    """

    df_scored = model.transform(df).select("CUST_CODE", "prediction")
    df_scored = df_scored.withColumnRenamed("prediction", "CLUSTER_NUM")

    if save_path:
        print("Saving K-Means clusters to {}".format(save_path))
        df_scored.write.parquet(save_path, mode="overwrite")

    return df_scored

import sys

sys.path.insert(1, "/home/ubuntu/cust2vec")

import os
import boto3
from io import StringIO
from pyspark.sql import SparkSession
from src.profiling.profiling import *
from utils.logging_framework import log
from pyspark.sql import functions as F

if __name__ == "__main__":

    task = sys.argv[1]
    bucket = sys.argv[2]
    staging_path = sys.argv[3]
    scored_kmeans_path = sys.argv[4]

    spark = SparkSession.builder.appName("cust2vec").getOrCreate()

    log.info("Running task {}".format(task))

    # ========== Import transaction data from staging ==========
    staging_trans_path = os.path.join(staging_path, "trans-data/")
    log.info("Importing transaction staging data from")
    trans_df = spark.read.parquet(staging_trans_path)

    # Keep only customers
    trans_df = trans_df.where(F.col("CUST_CODE").isNotNull())

    # ========== Import scored k-means DataFrame ==========
    log.info("Reading scored DataFrame from {}".format(scored_kmeans_path))
    scored_df = spark.read.parquet(scored_kmeans_path)

    # ========== Calculate overall KPIs by segment
    log.info("Calculating KPIs by segment")
    seg_kpis = overall_seg_kpis(
        trans_df=trans_df, scored_df=scored_df, seg_col="CLUSTER_NUM"
    )

    # ========== Create profiling features ==========
    log.info("Creating profiling features")

    # Get the core variables to profile into a DataFrame making unique at the customer and
    # variable level
    unique_cust_segs = trans_df.select(
        "CUST_CODE",
        "CUST_PRICE_SENSITIVITY",
        "CUST_LIFESTAGE",
        "PROD_CODE",
        "PROD_CODE_10",
        "PROD_CODE_20",
        "PROD_CODE_30",
        "PROD_CODE_40",
    ).dropDuplicates()

    # For all basket level segmentations get the dominant segment by customer
    # based on the percentage of baskets
    dom_basket_type = get_cust_dominant_seg(trans_df, "BASKET_TYPE")
    dom_basket_size = get_cust_dominant_seg(trans_df, "BASKET_SIZE")
    dom_basket_mission = get_cust_dominant_seg(trans_df, "BASKET_DOMINANT_MISSION")
    dom_store = get_cust_dominant_seg(trans_df, "STORE_CODE")
    dom_store_format = get_cust_dominant_seg(trans_df, "STORE_FORMAT")
    dom_store_region = get_cust_dominant_seg(trans_df, "STORE_REGION")
    dom_shop_weekday = get_cust_dominant_seg(trans_df, "SHOP_WEEKDAY")
    dom_shop_hour = get_cust_dominant_seg(trans_df, "SHOP_HOUR")

    # Combine all profiles
    profile_df = unique_cust_segs.join(dom_basket_type, "CUST_CODE")
    profile_df = profile_df.join(dom_basket_size, "CUST_CODE")
    profile_df = profile_df.join(dom_basket_mission, "CUST_CODE")
    profile_df = profile_df.join(dom_store, "CUST_CODE")
    profile_df = profile_df.join(dom_store_format, "CUST_CODE")
    profile_df = profile_df.join(dom_store_region, "CUST_CODE")
    profile_df = profile_df.join(dom_shop_weekday, "CUST_CODE")
    profile_df = profile_df.join(dom_shop_hour, "CUST_CODE")
    profile_df = profile_df.join(scored_df, "CUST_CODE")

    # ========== Run the profiling ==========

    log.info("Running profiling")
    # Create the profiling object
    profiler = ProfileModel(
        df=profile_df, seg_col="CLUSTER_NUM"
    )

    # Dominant basket mission (Fresh, Grocery, Mixed)
    log.info("Profiling - dominant basket mission executing")
    dom_miss_profile = profiler.run_profiles("DOM_CUST_BASKET_DOMINANT_MISSION")

    # Basket size (Small, Medium, Large)
    log.info("Profiling - basket size executing")
    bask_size_profile = profiler.run_profiles("DOM_CUST_BASKET_SIZE")

    # Basket type (Small Shop, Full Shop, Top Up Shop)
    log.info("Profiling - basket type executing")
    bask_type_profile = profiler.run_profiles("DOM_CUST_BASKET_TYPE")

    # Store format (SS, MS, LS, XLS - small store, medium store, large store, extra large store)
    log.info("Profiling - store format executing")
    store_format_profile = profiler.run_profiles("DOM_CUST_STORE_FORMAT")

    # Customer price sensitivty - LA (low affluence), MM (mid-market), UM (up-market)
    log.info("Profiling - customer price sensitivity executing")
    cust_ps_profile = profiler.run_profiles("CUST_PRICE_SENSITIVITY")

    # Day of week shopped
    log.info("Profiling - day of week shopped executing")
    day_profile = profiler.run_profiles("DOM_CUST_SHOP_WEEKDAY")

    # Hour shopped
    log.info("Profiling - hour shopped executing")
    hour_profile = profiler.run_profiles("DOM_CUST_SHOP_HOUR")

    # Prod code 40
    log.info("Profiling - prod code 40")
    prod_code_40_profile = profiler.run_profiles("PROD_CODE_40")

    # Prod code 30
    log.info("Profiling - prod code 30")
    prod_code_30_profile = profiler.run_profiles("PROD_CODE_30")

    # Set the profiles together
    all_profiles = (
        dom_miss_profile.union(bask_size_profile)
        .union(bask_type_profile)
        .union(store_format_profile)
        .union(cust_ps_profile)
        .union(day_profile)
        .union(hour_profile)
        .union(prod_code_40_profile)
        .union(prod_code_30_profile)
    )

    # ========== Save profiles ==========
    bucket = bucket
    csv_buffer = StringIO()
    all_profiles = all_profiles.toPandas()
    all_profiles.to_csv(csv_buffer)

    log.info("Saving model profiles")
    s3_resource = boto3.resource("s3")
    s3_resource.Object(bucket, "model-profiles/model_profiles.csv").put(
        Body=csv_buffer.getvalue()
    )

    # Save KPIs
    log.info("Saving KPIs")
    csv_buffer = StringIO()
    seg_kpis = seg_kpis.toPandas()
    seg_kpis.to_csv(csv_buffer)

    s3_resource.Object(bucket, "model-profiles/seg_kpis.csv").put(
        Body=csv_buffer.getvalue()
    )

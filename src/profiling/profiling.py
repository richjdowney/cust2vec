from pyspark.sql import functions as F
from pyspark.sql import Window


class ProfileModel:
    """Method to score missions and create profiles"""

    def __init__(self, df, seg_col):
        """
        Parameters
        ----------
        df : pyspark.sql.DataFrame
            DataFrame containing basket ID, profiling variables and segment feature
        seg_col : str
            Name of the column containing the segment to profile

        """

        self.df = df
        self.seg_col = seg_col
        self.tot_custs_by_seg = self.get_segment_counts()
        self.tot_custs = self.get_tot_counts()

    def get_segment_counts(self):
        """ Function to get counts of customers by segment

        Returns
        ----------
        tot_custs_by_segment : pyspark.sql.DataFrame
            DataFrame containing total customer counts by segment

        """

        # Get the count of total baskets by mission
        tot_custs_by_segment = self.df.groupBy(self.seg_col).agg(
            F.countDistinct("CUST_CODE").alias("TOT_CUSTS_SEG")
        )

        tot_custs_by_segment.persist()

        return tot_custs_by_segment

    def get_tot_counts(self):
        """ Function to get count of total customers

        Returns
        ----------
        tot_custs : int
            Count of total customers

        """

        # Get the count of total baskets
        tot_custs = self.df.agg(
            F.countDistinct("CUST_CODE").alias("TOT_CUSTS")
        ).collect()[0][0]

        return tot_custs

    def run_profiles(self, var):
        """ Function to create a profile by the specified variable

        Parameters
        ----------
        var  : str
            The name of the variable to profile

        Returns
        ----------
        trans_profile : pyspark.sql.DataFrame
            DataFrame containing the profile by segment

        """

        # Make the DataFrame unique by customer and profile variable
        trans_profile = self.df.select("CUST_CODE", self.seg_col, var).dropDuplicates()

        # Get the count of baskets by mission and profile variable
        trans_profile = trans_profile.groupby(self.seg_col, var).agg(
            F.countDistinct("CUST_CODE").alias("NUM_CUSTS_PROFILE")
        )

        trans_profile = trans_profile.join(self.tot_custs_by_seg, self.seg_col)

        # Get percentage of baskets by mission and profile variable
        trans_profile = trans_profile.withColumn(
            "PERC_SEG", F.col("NUM_CUSTS_PROFILE") / F.col("TOT_CUSTS_SEG")
        )

        # Get percentage of baskets over all missions
        tot_custs_profile = self.df.groupby(var).agg(
            F.countDistinct("CUST_CODE").alias("NUM_CUSTS_OVERALL_PROFILE")
        )

        # Calculate the index
        index_perc = tot_custs_profile.withColumn(
            "PERC_OVERALL", F.col("NUM_CUSTS_OVERALL_PROFILE") / self.tot_custs
        )

        trans_profile = trans_profile.join(index_perc, var)
        trans_profile = trans_profile.withColumn(
            "INDEX", (F.col("PERC_SEG") / F.col("PERC_OVERALL")) * 100
        )
        trans_profile = trans_profile.drop(
            "NUM_CUSTS_PROFILE",
            "TOT_CUSTS_SEG",
            "NUM_CUSTS_OVERALL_PROFILE",
            "PERC_OVERALL",
        )

        # Rename the columns to allow profiles to be set together
        trans_profile = trans_profile.withColumnRenamed(var, "var_detail")
        trans_profile = trans_profile.withColumn("var", F.lit(var))

        trans_profile = trans_profile.orderBy([self.seg_col, var])

        return trans_profile


def overall_seg_kpis(trans_df, scored_df, seg_col):
    """ Function to create overall segment KPIs

        Parameters
        ----------
        trans_df : pyspark.sql.DataFrame
            DataFrame containing raw transactions
        scored_df : str
            DataFrame containing the customer segment to profile
        seg_col : str
            Name of the column containing the profile to segment

    """

    trans_df_kpis = trans_df.select("CUST_CODE", "QUANTITY", "SPEND", "BASKET_ID")
    trans_df_kpis = trans_df_kpis.join(scored_df, ["CUST_CODE"])
    trans_df_kpis = trans_df_kpis.groupBy(seg_col).agg(
        F.sum("SPEND").alias("TOT_SALES"),
        F.sum("QUANTITY").alias("TOT_UNITS"),
        F.countDistinct("BASKET_ID").alias("TOT_BASKS"),
        F.countDistinct("CUST_CODE").alias("TOT_CUSTS"),
    )

    trans_df_kpis = (
        trans_df_kpis.withColumn(
            "SPEND_PER_BASK", F.col("TOT_SALES") / F.col("TOT_BASKS")
        )
        .withColumn("UNITS_PER_BASK", F.col("TOT_UNITS") / F.col("TOT_BASKS"))
        .withColumn("SPEND_PER_CUST", F.col("TOT_SALES") / F.col("TOT_CUSTS"))
        .withColumn("UNITS_PER_CUST", F.col("TOT_UNITS") / F.col("TOT_CUSTS"))
        .withColumn("PPU", F.col("TOT_SALES") / F.col("TOT_UNITS"))
    )

    return trans_df_kpis


def get_cust_dominant_seg(df, var):
    """ Function to get the dominant segment at the customer level for any basket level segment -
        Dominant is based on the percentage of baskets with that segment

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            DataFrame containing raw transactions
        var : str
            Name of the basket level segment

    """

    # Get counts for segment
    dom_bask_seg = df.groupBy("CUST_CODE", var).agg(
        F.countDistinct("BASKET_ID").alias("BASK_COUNT")
    )

    # Get counts of baskets for each customer
    tot_basks_cust = dom_bask_seg.groupBy("CUST_CODE").agg(
        F.sum("BASK_COUNT").alias("TOT_BASK_COUNT")
    )

    # Merge back and calcuate percentage of baskets
    dom_bask_seg = dom_bask_seg.join(tot_basks_cust, ["CUST_CODE"])
    dom_bask_seg = dom_bask_seg.withColumn(
        "PERC_BASKS", F.col("BASK_COUNT") / F.col("TOT_BASK_COUNT")
    )

    w = Window.partitionBy("CUST_CODE")
    dom_bask_seg = dom_bask_seg.withColumn(
        "MAX_PERC", F.max("PERC_BASKS").over(w)
    ).where(F.col("PERC_BASKS") == F.col("MAX_PERC"))

    dom_bask_seg = dom_bask_seg.select(
        "CUST_CODE", F.col(var).alias("DOM_CUST_{}".format(var))
    )

    return dom_bask_seg

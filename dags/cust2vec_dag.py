# Add path to modules to sys path
import sys

sys.path.insert(1, "/home/ubuntu/cust2vec")

from airflow import DAG
from airflow.contrib.operators.emr_create_job_flow_operator import (
    EmrCreateJobFlowOperator,
)
from airflow.contrib.sensors.emr_step_sensor import EmrStepSensor
from airflow.contrib.operators.emr_add_steps_operator import EmrAddStepsOperator
from airflow.contrib.operators.emr_terminate_job_flow_operator import (
    EmrTerminateJobFlowOperator,
)
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from config.load_config import load_yaml
from config import constants
from config.load_config import Config
from utils.send_email import notify_email
from utils.logging_framework import log
from utils.copy_app_to_s3 import copy_app_to_s3
from runners import cust2vec_model_fit_runner

# Load the config file
config = load_yaml(constants.config_path)

# Check the config types
try:
    Config(**config)
except TypeError as error:
    log.error(error)

with DAG(**config["dag"]) as dag:

    # Create egg file
    create_egg = BashOperator(
        task_id="create_app_egg",
        bash_command="cd /home/ubuntu/cust2vec && python /home/ubuntu/cust2vec/setup.py bdist_egg",
        run_as_user="airflow",
    )

    # Copy application files to s3
    upload_code = PythonOperator(
        task_id="upload_app_to_s3", python_callable=copy_app_to_s3, op_args=[config]
    )

    # Start the cluster for data prep
    data_prep_cluster_creator = EmrCreateJobFlowOperator(
        task_id="create_data_prep_job_flow",
        job_flow_overrides=config["emr"],
        aws_conn_id="aws_default",
        emr_conn_id="emr_default",
        on_failure_callback=notify_email,
    )

    # ========== DATA STAGING ==========
    task = "data_staging"
    data_staging = EmrAddStepsOperator(
        task_id="add_step_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull(task_ids='create_job_flow', key='return_value') }}",
        aws_conn_id="aws_default",
        steps=[
            {
                "Name": "Run data staging step",
                "ActionOnFailure": "CONTINUE",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": [
                        "spark-submit",
                        "--deploy-mode",
                        "cluster",
                        "--py-files",
                        config["s3"]["egg"],
                        config["s3"]["StageRunner"],
                        task,
                        config["s3"]["Bucket"],
                        config["s3"]["DataFolder"],
                        config["s3"]["StagingDataPath"],
                        "{{ execution_date }}",
                    ],
                },
            }
        ],
        on_failure_callback=notify_email,
    )

    step_name = "add_step_{}".format(task)
    data_staging_step_sensor = EmrStepSensor(
        task_id="watch_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull('create_job_flow', key='return_value') }}",
        step_id="{{{{ task_instance.xcom_pull(task_ids='{}', key='return_value')[0] }}}}".format(
            step_name
        ),
        aws_conn_id="aws_default",
        on_failure_callback=notify_email,
    )

    # ========== DATA PRE-PROCESSING ==========
    task = "data_preprocessing"
    data_preprocessing = EmrAddStepsOperator(
        task_id="add_step_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull(task_ids='create_job_flow', key='return_value') }}",
        aws_conn_id="aws_default",
        steps=[
            {
                "Name": "Run data pre-processing step",
                "ActionOnFailure": "CONTINUE",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": [
                        "spark-submit",
                        "--deploy-mode",
                        "cluster",
                        "--py-files",
                        config["s3"]["egg"],
                        config["s3"]["DataPreProcessingRunner"],
                        task,
                        config["s3"]["Bucket"],
                        config["s3"]["StagingDataPath"],
                        "{{ execution_date }}",
                        config["DataPreProcessing"]["Sample"],
                        config["DataPreProcessing"]["SampleRate"],
                        config["DataPreProcessing"]["NumProds"],
                    ],
                },
            }
        ],
        on_failure_callback=notify_email,
    )

    step_name = "add_step_{}".format(task)
    data_preprocessing_step_sensor = EmrStepSensor(
        task_id="watch_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull('create_job_flow', key='return_value') }}",
        step_id="{{{{ task_instance.xcom_pull(task_ids='{}', key='return_value')[0] }}}}".format(
            step_name
        ),
        aws_conn_id="aws_default",
        on_failure_callback=notify_email,
    )

    # Remove the EMR cluster
    data_prep_cluster_remover = EmrTerminateJobFlowOperator(
        task_id="remove_EMR_cluster_for_data_prep",
        job_flow_id="{{ task_instance.xcom_pull(task_ids='create_job_flow', key='return_value') }}",
        aws_conn_id="aws_default",
        on_failure_callback=notify_email,
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # ========== CUST2VEC MODEL FIT ==========
    task = "cust2vec_model_fit"
    cust2vec_fit = PythonOperator(
        task_id="run_cust2vec_model_fit",
        dag=dag,
        provide_context=False,
        python_callable=cust2vec_model_fit_runner.task_cust2vec_model_fit,
        op_kwargs={
            "bucket": config["s3"]["Bucket"],
            "model": config["Cust2VecModel"]["Model"],
            "window_size": config["Cust2VecModel"]["WindowSize"],
            "embedding_size": config["Cust2VecModel"]["EmbeddingSize"],
            "num_epochs": config["Cust2VecModel"]["NumEpochs"],
            "steps_per_epoch": config["Cust2VecModel"]["StepsPerEpoch"],
            "early_stopping_patience": config["Cust2VecModel"]["EarlyStoppingPatience"],
            "save_period": config["Cust2VecModel"]["SavePeriod"],
            "save_path": config["Cust2VecModel"]["SavePath"],
            "save_cust_embeddings": config["Cust2VecModel"]["SaveCustEmbeddings"],
            "save_cust_embeddings_period": config["Cust2VecModel"][
                "SaveCustEmbeddingsPeriod"
            ],
            "task": task,
        },
        on_failure_callback=notify_email,
    )

    cust_segmentation_cluster_creator = EmrCreateJobFlowOperator(
        task_id="create_segmentation_job_flow",
        job_flow_overrides=config["emr"],
        aws_conn_id="aws_default",
        emr_conn_id="emr_default",
        on_failure_callback=notify_email,
    )

    # ========== CUSTOMER CLUSTERING ==========
    task = "customer_clustering"
    customer_clustering = EmrAddStepsOperator(
        task_id="add_step_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull(task_ids='create_job_flow', key='return_value') }}",
        aws_conn_id="aws_default",
        steps=[
            {
                "Name": "Run customer clustering step",
                "ActionOnFailure": "CONTINUE",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": [
                        "spark-submit",
                        "--deploy-mode",
                        "cluster",
                        "--py-files",
                        config["s3"]["egg"],
                        config["s3"]["CustClusteringRunner"],
                        task,
                        config["s3"]["Bucket"],
                        config["KMeansModel"]["CreateCostPlot"],
                        config["s3"]["ScoredKMeansPath"],
                        config["s3"]["SavedKmeansModel"],
                        config["KMeansModel"]["NumClusts"],
                        "{{ execution_date }}",
                    ],
                },
            }
        ],
        on_failure_callback=notify_email,
    )

    step_name = "add_step_{}".format(task)
    cust_clustering_step_sensor = EmrStepSensor(
        task_id="watch_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull('create_job_flow', key='return_value') }}",
        step_id="{{{{ task_instance.xcom_pull(task_ids='{}', key='return_value')[0] }}}}".format(
            step_name
        ),
        aws_conn_id="aws_default",
        on_failure_callback=notify_email,
    )

    # ========== CUSTOMER PROFILING ==========
    task = "customer_profiling"
    customer_profiling = EmrAddStepsOperator(
        task_id="add_step_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull(task_ids='create_job_flow', key='return_value') }}",
        aws_conn_id="aws_default",
        steps=[
            {
                "Name": "Run cluster profiling step",
                "ActionOnFailure": "CONTINUE",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": [
                        "spark-submit",
                        "--deploy-mode",
                        "cluster",
                        "--py-files",
                        config["s3"]["egg"],
                        config["s3"]["ProfilingRunner"],
                        task,
                        config["s3"]["Bucket"],
                        config["s3"]["StagingDataPath"],
                        config["s3"]["ScoredKMeansPath"],
                        "{{ execution_date }}",
                    ],
                },
            }
        ],
        on_failure_callback=notify_email,
    )

    step_name = "add_step_{}".format(task)
    customer_profiling_step_sensor = EmrStepSensor(
        task_id="watch_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull('create_job_flow', key='return_value') }}",
        step_id="{{{{ task_instance.xcom_pull(task_ids='{}', key='return_value')[0] }}}}".format(
            step_name
        ),
        aws_conn_id="aws_default",
        on_failure_callback=notify_email,
    )

    # Remove the EMR cluster
    segmentation_cluster_remover = EmrTerminateJobFlowOperator(
        task_id="remove_EMR_cluster_for_segmentation",
        job_flow_id="{{ task_instance.xcom_pull(task_ids='create_job_flow', key='return_value') }}",
        aws_conn_id="aws_default",
        on_failure_callback=notify_email,
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    create_egg >> upload_code >> data_prep_cluster_creator >> data_staging >> data_staging_step_sensor >>\
    data_preprocessing >> data_preprocessing_step_sensor >> data_prep_cluster_remover >> cust2vec_fit >> \
    cust_segmentation_cluster_creator >> customer_clustering >> cust_clustering_step_sensor >> customer_profiling >> \
    customer_profiling_step_sensor >> segmentation_cluster_remover


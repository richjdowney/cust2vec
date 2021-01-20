import yaml
from typing import Any, Dict
from utils.logging_framework import log
import pydantic


class ConfigDefaultArgs(pydantic.BaseModel):
    """Configuration for the default args when setting up the DAG"""

    owner: str
    start_date: str
    end_date: str
    depends_on_past: bool
    retries: int
    catchup: bool
    email: str
    email_on_failure: bool
    email_on_retry: bool


class ConfigDag(pydantic.BaseModel):
    """Configuration for the DAG runs"""

    # Name for the DAG run
    dag_id: str

    # Default args for DAG run e.g. owner, start_date, end_date
    default_args: ConfigDefaultArgs

    # DAG schedule interval
    schedule_interval: str


class ConfigEmr(pydantic.BaseModel):
    """Configuration for EMR clusters"""

    Instances: Dict[str, Any]

    # EMR ec2 role
    JobFlowRole: str

    # EMR role
    ServiceRole: str

    # Cluster name
    Name: str

    # Path to save logs
    LogUri: str

    # EMR version
    ReleaseLabel: str

    # Cluster configurations
    Configurations: Dict[str, Any]

    # Path to dependencies shell script on s3
    BootstrapActions: Dict[str, Any]

    # Number of steps EMR can run concurrently
    StepConcurrencyLevel: int


class ConfigApp(pydantic.BaseModel):
    """Configuration for application paths"""

    # Path to the root directory on EC2
    RootPath: str

    # Path to the runner files
    PathToRunners: str

    # Path to the bin directory on EC2
    PathToBin: str

    # Path to the egg file on EC2
    PathToEgg: str

    # Path to the utils directory on EC2
    PathToUtils: str

    # Name of the main application egg object
    EggObject: str

    # Name of Spark runner to stage tables
    StageRunner: str

    # Name of Spark runner to pre-process the models data
    DataPreProcessingRunner: str

    # Name of Spark runner to fit cust2vec models
    Cust2VecRunner: str

    # Name of the shell script for bootstrapping
    DependenciesShell: str

    # Name of the package requirements
    Requirements: str


class ConfigDataPreProcessing(pydantic.BaseModel):
    """Configuration for cust2vec processing and training"""

    # Config for sampling
    Sample: str
    SampleRate: str
    NumProds: str


class ConfigAirflow(pydantic.BaseModel):
    """Configuration for Airflow access to AWS"""

    # Config for airflow defaults
    AwsCredentials: str


class ConfigS3(pydantic.BaseModel):
    """Configuration for application paths"""

    # Bucket with input data on s3
    Bucket: str

    # Folder where the input data is located
    DataFolder: str

    # Path to staging data
    StagingDataPath: str

    # Path to egg file
    egg: str

    # Path to staging tables runner file
    StageRunner: str

    # Path to data pre-processing runner file
    DataPreProcessingRunner: str

    # Path to cust2vec models train runner file
    Cust2VecRunner: str


class ConfigCust2VecModel(pydantic.BaseModel):
    """Configuration for cust2vec model"""

    # dm or dbow model
    Model: str

    # Window size for context
    WindowSize: str

    # Size for embeddings
    EmbeddingSize: str

    # Number of epochs
    NumEpochs: str

    # Steps per epoch
    StepsPerEpoch: str

    # Early stopping patience
    EarlyStoppingPatience: str

    # Number of epochs per save
    SavePeriod: str

    # Path to save cust2vec model
    SavePath: str

    # Path to save customer embeddings
    SaveCustEmbeddings: str

    # Number of epochs between embedding saves
    SaveCustEmbeddingsPeriod: str


class Config(pydantic.BaseModel):
    """Main configuration"""

    dag: ConfigDag
    emr: ConfigEmr
    app: ConfigApp
    s3: ConfigS3
    airflow: ConfigAirflow
    DataPreProcessing: ConfigDataPreProcessing
    Cust2VecModel: ConfigCust2VecModel


class ConfigException(Exception):
    pass


def load_yaml(config_path):

    """Function to load yaml file from path

    Parameters
    ----------
    config_path : str
        string containing path to yaml

    Returns
    ----------
    config : dict
        dictionary containing config

    """
    log.info("Importing config file from {}".format(config_path))

    if config_path is not None:
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)

        log.info("Successfully imported the config file from {}".format(config_path))

    if config_path is None:
        raise ConfigException("Must supply path to the config file")

    return config

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

from src.etl import prepare_separate_datasets
from src.ml import run_train_predict


def main():
    
    spark = SparkSession.builder \
        .appName("SparkApp") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.jars.packages", "io.delta:delta-spark_3.5:3.2.0") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.memory.fraction", "0.6") \
        .config("spark.memory.storageFraction", "0.3") \

    spark = configure_spark_with_delta_pip(spark).getOrCreate()

    raw_data_path = '/app/data/train.csv'
    prepare_separate_datasets(spark, raw_data_path)
    run_train_predict(spark)
    spark.stop()


if __name__ == "__main__":
    main()

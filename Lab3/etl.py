import logging

from pyspark.sql.functions import when,col, split
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder


def cast_numeric_types(df):
    logging.info("Casting types")

    float_columns = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa"]
    for column in float_columns:
        df = df.withColumn(column, df[column].cast(FloatType()))

    return df

def encode_categories(df):
    logging.info("Encoding")

    categorical_columns = ["HomePlanet", "CryoSleep", "Destination", "VIP"]

    indexers = [StringIndexer(
        inputCol=column, 
        outputCol=column + "_INDEX") 
        for column in categorical_columns]
    
    encoders = [OneHotEncoder(
        inputCol=indexer.getOutputCol(), 
        outputCol=column + "_ENC") 
        for indexer, column in zip(indexers, categorical_columns)]

    pipeline = Pipeline(stages=indexers + encoders)

    encoder = pipeline.fit(df)
    df = encoder.transform(df)

    for column in categorical_columns:
        df = df.drop(column, column + "_INDEX")
        
    return df

def prepare_dataframe(df):
    columns = ["HomePlanet", "CryoSleep", "Destination", "VIP", 
               "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa",
               "Transported"]
    
    df = df.select(*columns)
    df = cast_numeric_types(df)
    df = encode_categories(df)
    return df

def prepare_bronze(spark, input_path):
    raw_data = spark.read.csv(input_path, header=True)
    raw_data.repartition(spark.sparkContext.defaultParallelism) \
        .write.format("delta") \
        .mode("overwrite") \
        .save("/app/data/bronze_ds")

def prepare_silver(spark):
    data = (spark.read.format('delta').load('/app/data/bronze_ds')
            .repartition(spark.sparkContext.defaultParallelism)
            .dropna(subset=["PassengerId", "Name", "Cabin"])
            )
    data.repartition(spark.sparkContext.defaultParallelism) \
        .write.format("delta") \
        .mode("overwrite") \
        .save("/app/data/silver_ds")

def prepare_gold(spark):
    data = spark.read.format('delta').load('/app/data/silver_ds')
    data = prepare_dataframe(data)
    data.repartition(spark.sparkContext.defaultParallelism) \
        .write.format("delta") \
        .mode("overwrite") \
        .save("/app/data/gold_ds")

def prepare_separate_datasets(spark, input_path):
    prepare_bronze(spark, input_path)
    prepare_silver(spark)
    prepare_gold(spark)
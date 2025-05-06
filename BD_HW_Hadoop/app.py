# coding: utf-8
import time
import psutil
import sys
import os
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, Imputer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F

OPTIMIZED = sys.argv[1] == "True"

categorical_columns = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
numeric_columns = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa"]
remove_columns = ["PassengerId", "Name"]
target_column = "Transported"

conf = SparkConf()
conf.set("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
conf.set("spark.executor.memory", "1536m")
conf.set("spark.driver.memory", "2g")
conf.set("spark.driver.maxResultSize", "1g")
conf.set("spark.ui.showConsoleProgress", "false")
conf.set("spark.jars.ivy", "/tmp/.ivy2")

spark = SparkSession.builder \
    .appName('binary_classification') \
    .master("spark://spark-master:7077") \
    .config(conf=conf) \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

start_time = time.time()
process = psutil.Process(os.getpid())

HDFS_PATH = "hdfs:///data/train.csv"
df = spark.read.csv(HDFS_PATH, header=True, inferSchema=True)

if OPTIMIZED:
    df.cache()
    df = df.repartition(4)

df = df.drop(*remove_columns)

for col in categorical_columns:
    df = df.fillna("Unknown", subset=[col])

# Разделение колонки Cabin (по дефолту идет в формате A/0/S, B/1/P и так далее)
df = df.withColumn("Cabin_1", F.split("Cabin", "/")[0]) \
       .withColumn("Cabin_2", F.split("Cabin", "/")[1]) \
       .withColumn("Cabin_3", F.split("Cabin", "/")[2]) \
       .drop("Cabin")

categorical_columns.extend(["Cabin_1", "Cabin_2", "Cabin_3"])

imputer = Imputer(
    inputCols=numeric_columns,
    outputCols=[f"{col}_imputed" for col in numeric_columns],
    strategy="median"
)

indexers = [
    StringIndexer(inputCol=col, outputCol=f"{col}_indexed").setHandleInvalid("keep") 
    for col in categorical_columns
]

vector_to_double = F.udf(lambda v: float(v[0]), DoubleType())

pipeline_stages = [imputer] + indexers

numeric_assemblers = []
numeric_scalers = []

for col_name in numeric_columns:
    assembler = VectorAssembler(
        inputCols=[f"{col_name}_imputed"], 
        outputCol=f"{col_name}_vector"
    )
    scaler = StandardScaler(
        inputCol=f"{col_name}_vector", 
        outputCol=f"{col_name}_scaled"
    )
    numeric_assemblers.append(assembler)
    numeric_scalers.append(scaler)

pipeline_stages += numeric_assemblers + numeric_scalers

feature_columns = [f"{col}_indexed" for col in categorical_columns] + \
                 [f"{col}_scaled" for col in numeric_columns]

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)
pipeline_stages.append(assembler)

pipeline = Pipeline(stages=pipeline_stages)
pipeline_model = pipeline.fit(df)
processed_data = pipeline_model.transform(df)

train_df, test_df = processed_data.randomSplit([0.75, 0.25])

if OPTIMIZED:
    train_df.cache()
    test_df.cache()

lr = LogisticRegression(featuresCol="features", labelCol=target_column)

param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

evaluator = BinaryClassificationEvaluator(
    labelCol=target_column,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

crossval = CrossValidator(
    estimator=lr,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=4
)

cv_model = crossval.fit(train_df)

best_model = cv_model.bestModel

predictions = best_model.transform(test_df)

auc = evaluator.evaluate(predictions)

evaluator_pr = BinaryClassificationEvaluator(
    labelCol=target_column,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)
aupr = evaluator_pr.evaluate(predictions)

if OPTIMIZED:
    train_df.unpersist()
    test_df.unpersist()

total_time = time.time() - start_time
ram_usage = process.memory_info().rss / (1024 * 1024)

print(f"Time: {total_time:.2f} sec, RAM: {ram_usage:.2f} MB")
print(f"AUC-ROC: {auc:.4f}, AUC-PR: {aupr:.4f}")
print(f"Best model params: RegParam={best_model.getRegParam()}, ElasticNetParam={best_model.getElasticNetParam()}")

with open("/log.txt", "a") as f:
    f.write(f"Time: {total_time:.2f} sec, RAM: {ram_usage:.2f} MB, AUC-ROC: {auc:.4f}, AUC-PR: {aupr:.4f}\n")
    f.write(f"Best params: RegParam={best_model.getRegParam()}, ElasticNetParam={best_model.getElasticNetParam()}\n")

spark.stop()
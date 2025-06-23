import logging
import mlflow
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from src.etl import prepare_dataframe


def vectorize(df):
    logging.info("Vectorizing data for ML")

    features_name = "features"
    features_columns = df.columns[1:]

    assembler = VectorAssembler(inputCols=features_columns, outputCol=features_name)
    df = assembler.transform(df)

    return df

def run_train_predict(spark):
    df = spark.read.format('delta').load('/app/data/gold_ds')
    
    mlflow.set_tracking_uri("http://0.0.0.0:8118")
    
    if not mlflow.get_experiment_by_name("GBTClassifier"):
        mlflow.create_experiment("GBTClassifier")
    mlflow.set_experiment("GBTClassifier")

    print('Start pipeline training')
    with mlflow.start_run():
        label_name = "Transported"
        features_name = "features"

        prepared_df = vectorize(df)

        train, test = prepared_df.randomSplit([0.75, 0.25], seed=0)

        clf = GBTClassifier(featuresCol=features_name, labelCol=label_name)
        model = clf.fit(train)
        mlflow.spark.log_model(model, "GBTClassifier")
        mlflow.log_artifact("/app/logs/logging_file.log")
        prediction = model.transform(test)

        evaluator = BinaryClassificationEvaluator(labelCol=label_name)
        auc = evaluator.evaluate(prediction, {evaluator.metricName: "areaUnderROC"})
        mlflow.log_metric("AUC", auc)

FROM bitnami/spark:3.4.1  

USER root

# Установка Python зависимостей
RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    delta-spark==2.4.0 \
    pyspark==3.4.1 \
    pandas \
    py4j==0.10.9.7 \
    scikit-learn

# Создаем структуру папок
RUN mkdir -p /BD_lab_3/data/bronze \
    /BD_lab_3/data/silver \
    /BD_lab_3/data/gold \
    /BD_lab_3/logs \
    /BD_lab_3/mlruns

WORKDIR /BD_lab_3

# Копируем файлы
COPY ./src/spark.py /BD_lab_3/src/
COPY ./src /BD_lab_3/src
COPY ./data/train.csv /BD_lab_3/data/train.csv

COPY ./src/run.sh /BD_lab_3/run.sh
RUN chmod +x /BD_lab_3/run.sh

ENV SPARK_LOCAL_DIRS=/tmp/spark
ENV SPARK_WORKER_MEMORY=1g

ENTRYPOINT ["/BD_lab_3/run.sh"]
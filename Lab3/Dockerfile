FROM bitnami/spark:3.5.0

USER root
RUN apt-get update && apt-get install -y python3-pip curl git

RUN pip3 install delta-spark==3.2.0 scikit-learn psutil tqdm mlflow pandas numpy --no-cache-dir 

ENV PATH=$PATH:/home/spark/.local/bin
ENV PYTHONPATH=/app:$PYTHONPATH

ENV SPARK_HOME=/opt/bitnami/spark/
RUN export PYSPARK_PYTHON=$(which python3)
ENV PYSPARK_PYTHON=$PYSPARK_PYTHON
ENV SPARK_VERSION=3.5
ENV SPARK_JARS_PACKAGES=io.delta:delta-spark_3.5:3.2.0

RUN curl -L -o /opt/bitnami/spark/jars/delta-core_2.12-3.2.0.jar https://repo1.maven.org/maven2/io/delta/delta-core_2.12/3.2.0/delta-core_2.12-3.2.0.jar && \
    curl -L -o /opt/bitnami/spark/jars/delta-storage-2.2.0.jar https://repo1.maven.org/maven2/io/delta/delta-storage/2.2.0/delta-storage-2.2.0.jar

RUN mkdir -p /app/{src,data,logs}

WORKDIR /app

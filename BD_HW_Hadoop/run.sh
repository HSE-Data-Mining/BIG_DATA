#!/bin/bash

DATA_PATH="data/train.csv" 
FILENAME="train.csv"
SPARK_APP="app.py"      
HDFS_PATH="/data"

NODES_COUNT=1
OPTIMIZED=False

while [[ $# -gt 0 ]]
do
    case "$1" in
        --optimized) 
            OPTIMIZED="$2"
            shift 2
            ;;
        --nodes)
            NODES_COUNT="$2"
            shift 2
            ;;
    esac
done

if [[ $NODES_COUNT -eq 1 ]]
then
    docker compose -f docker-compose-1.yml up -d
elif [[ $NODES_COUNT -eq 3 ]]
then
    docker compose -f docker-compose-3.yml up -d
fi

docker cp "$DATA_PATH" namenode:/
sleep 10
docker exec -it namenode hdfs dfs -mkdir ${HDFS_PATH}

docker exec -it namenode hdfs dfs -put -f "/$FILENAME" "$HDFS_PATH/$FILENAME"

docker cp "$SPARK_APP" spark-master:/tmp/

docker exec spark-master /bin/sh -c "apk update && apk add --no-cache python3 py3-pip python3-dev make automake gcc g++ linux-headers"
docker exec spark-worker-1 /bin/sh -c "apk update && apk add --no-cache python3 py3-pip python3-dev make automake gcc g++ linux-headers"
docker exec -it spark-master pip3 install psutil

docker exec -it spark-master pip3 install numpy
docker exec spark-worker-1 pip3 install numpy

N=5
for ((i=1; i<=N; i++))
do
    docker exec -it spark-master /spark/bin/spark-submit --master spark://spark-master:7077 /tmp/"$SPARK_APP" ${OPTIMIZED}
done

docker cp spark-master:/log.txt ./logs/nodes_${NODES_COUNT}_opt_${OPTIMIZED}.txt

if [[ $DATANODES_COUNT -eq 1 ]]
then
    docker compose -f docker-compose.yml down
else
    docker compose -f docker-compose-3.yml down
fi
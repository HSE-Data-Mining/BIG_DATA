bootstrap_servers = "kafka-inference:9092"

cfg_producer = {
    'bootstrap.servers' : bootstrap_servers,
    'group.id': 'mnist_group',
}

cfg_consumer = {
    'bootstrap.servers': bootstrap_servers,
    'group.id': 'mnist_group',
    'auto.offset.reset': 'earliest'
}

kafka_topic = "mnist_images"
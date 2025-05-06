import sys
from pathlib import Path
import torch
from confluent_kafka import Producer, Consumer
import pickle

from backend.data import load_mnist_data, SimpleCNN, run_train
from backend.config_file import cfg_producer, cfg_consumer

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))


class producer_mnist_inference:
    def __init__(self):
        self.producer = Producer(cfg_producer)

    def send_to_kafka(self, topic, data, model_path):
        model = SimpleCNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        with torch.no_grad():
            output = model(data)
            _, predicted = torch.max(output, 1)

            predicted_label = predicted.item()
            print(f'Predicted Label: {predicted_label}')

        message = pickle.dumps(predicted_label)
        self.producer.produce(topic, value=message)
        self.producer.flush()

class consumer_mnist_inference:
    def __init__(self):
        self.consumer = Consumer(cfg_consumer)

    def receive_from_kafka(self, topic):
        self.consumer.subscribe([topic])
        print(f"Subscribed to topic: {topic}")
        msg = self.consumer.poll(timeout=30.0)

        if msg is None:
            return None
        try:
            message = pickle.loads(msg.value())
            return message
        except pickle.PickleError as e:
            print(f"Error decoding message: {e}")
            return None
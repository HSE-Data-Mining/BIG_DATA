from pathlib import Path
import sys
import torch

import streamlit as st

from PIL import Image
import time

from backend.producers_consumers import producer_mnist_inference, consumer_mnist_inference
from backend.config_file import kafka_topic
from backend.data import get_mnist_data
from frontend.upload import main_tab, tab_input_image, tab_2

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from backend.data import inference_transforms

def get_weights_and_testloader(need_train):
    test_loader, model_path = get_mnist_data(need_train)

    return test_loader, model_path

def main():
    logs_file = Path("outs_and_logs/logs.txt")
    model_path = Path("outs_and_logs/mnist_cnn.pth")


    def general_window():
        if st.session_state.current_tab == "Main":
            main_tab()
        elif st.session_state.current_tab == "Tab 1":
            tab_input_image()
        elif st.session_state.current_tab == "Tab 2":
            tab_2()

    general_window()
    # st.title('MNIST Image Classification with Kafka')

    
    # if st.button("Train model"):
    #     test_loader, model_path = get_weights_and_testloader(need_train=True)
    # else:
    #     test_loader, model_path = get_weights_and_testloader(need_train=False)

    # if st.button("Upload image"):
    if False:
        producer_inference = producer_mnist_inference()
        consumer_inference = consumer_mnist_inference()

        uploaded_file = st.file_uploader("Upload an MNIST image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('L')
            image = image.resize((28, 28))
            image_tensor = inference_transforms(image)

            producer_inference.send_to_kafka(kafka_topic, image_tensor, model_path)

            prediction_data = consumer_inference.receive_from_kafka(kafka_topic)
            
            if prediction_data is not None:
                if isinstance(prediction_data, int):
                    st.write(f'Predicted Label: {prediction_data}')    
                else:
                    st.write(f'Received data: {prediction_data}')
                
                with open(logs_file, "w") as f:
                    file_name = uploaded_file["name"]
                    f.write(f"Predicted: {prediction_data} --> For name='{file_name}'")

            else:
                st.write("No prediction received from Kafka.")

            st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button("Close"):
            st.write("Closing the application...")
            st.stop()

    # if st.button("Test data"):
    #     print('нужно доделать')
    #     print(f"Use test_loader : {len(test_loader)}")

if __name__ == "__main__":
    main()
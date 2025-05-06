FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .

COPY . /app

RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu 
    
RUN pip install streamlit confluent_kafka pillow matplotlib pandas

# RUN python backend/data.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
RUN pip install -e .

EXPOSE 8080

ENTRYPOINT ["specinferkit"]
CMD ["serve", "--checkpoint", "/models/default", "--port", "8080"]

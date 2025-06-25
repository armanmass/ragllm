# Retrievment Augmented Generation

Creates vector search database which is queried at prompt time and most relevant indexes will be added to context prior to generation. Good for internal systems with private documents. Currently only supports PDF documents. Context size is limited but can easily be scaled by changing model.
## Usage

Step 1: Clone Repo
```bash
git clone https://github.com/armanmass/ragllm.git
```

Step 2: Creat .env file with HF Token
```
HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXX
```

Step 3: Place PDF files in data/ directory.

Step 4: Build Docker Image

*This will automatically chunk PDF files and create vector search indexes.*

```
docker build -t ragllm .
```

```Docker
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY .env /app/.env

RUN python scripts/extract_and_chunk.py
RUN python scripts/build_faiss_index.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Step 5: Run Docker Container
```
docker run -p 8000:8000 ragllm
```

Step 6: Prompt LLM

```bash
$ curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "top_k": 6}'
```
```response
{"answer":"computers can be taught to perform almost any task, so long as we feed them enough training data"}
```

FROM python:3.12-slim as builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml .

RUN uv pip compile pyproject.toml -o requirements.txt

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

EXPOSE 8000
EXPOSE 8501

RUN echo '#!/bin/bash \n\
uvicorn src.app.api:app --host 0.0.0.0 --port 8000 & \n\
streamlit run src/app/streamlit.py --server.port 8501 --server.address 0.0.0.0 \
' > ./start.sh

RUN chmod +x ./start.sh

CMD ["./start.sh"]
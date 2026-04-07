FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn pydantic openai faker rapidfuzz python-dotenv openenv-core

COPY . .

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]

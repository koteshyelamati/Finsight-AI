FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["uvicorn", "mcp_server.server:create_server", "--factory", "--host", "0.0.0.0", "--port", "8000"]
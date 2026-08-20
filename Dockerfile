FROM python:3.11-slim

# opencv 运行所需的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces 容器内仅 /tmp 与用户目录可写
ENV MODEL_DIR=/tmp/models \
    UPLOAD_DIR=/tmp/uploads \
    PORT=7860 \
    PYTHONUNBUFFERED=1

EXPOSE 7860

# 单 worker + 多线程：任务状态存在进程内存中，多 worker 会导致轮询不到任务
CMD ["gunicorn", "-w", "1", "--threads", "16", "-b", "0.0.0.0:7860", \
     "--timeout", "0", "app:app"]

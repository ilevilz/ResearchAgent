# ResearchAgent 容器镜像
# 构建：docker build -t research-agent:latest .
# 运行：docker run --rm -p 8000:8000 --env-file .env research-agent:latest
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 系统依赖：faiss 需要 libgomp；curl 用于健康检查
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖，最大化利用镜像层缓存
COPY requirements.txt .
RUN pip install -r requirements.txt && pip install faiss-cpu

# 再拷代码
COPY pyproject.toml README.md ./
COPY research_agent/ ./research_agent/
COPY scripts/ ./scripts/
COPY web/ ./web/
COPY eval/ ./eval/ 2>/dev/null || true

# 以非 root 用户运行
RUN useradd -m -u 10001 agent && mkdir -p /app/data && chown -R agent:agent /app
USER agent

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "research_agent.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

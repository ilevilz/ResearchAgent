# ResearchAgent —— 常用命令
# 提示：本项目使用 conda 环境 `oct`，PY 变量可按需覆盖
PY ?= python
PORT ?= 8000

.PHONY: help install doctor tools ask chat serve mcp index search eval test docker-build docker-up clean

help:  ## 显示所有命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## 安装依赖
	$(PY) -m pip install -r requirements.txt
	@echo "可选：pip install -r requirements-optional.txt"

doctor:  ## 环境自检（依赖 / Key / 索引 / LLM 连通性）
	$(PY) -m research_agent doctor

tools:  ## 列出全部工具
	$(PY) -m research_agent tools

ask:  ## 单次任务：make ask Q="找几篇 RAG 论文"
	$(PY) -m research_agent ask "$(Q)"

chat:  ## 交互式多轮对话
	$(PY) -m research_agent chat

serve:  ## 启动 FastAPI + Web Demo
	$(PY) -m research_agent serve --port $(PORT)

mcp:  ## 以 MCP Server (stdio) 启动
	$(PY) -m research_agent mcp

index:  ## 建库：make index Q="retrieval augmented generation" N=8
	$(PY) scripts/build_kb.py --query "$(Q)" -n $(or $(N),8)

search:  ## 知识库检索：make search Q="ReAct"
	$(PY) -m research_agent search "$(Q)"

eval:  ## 运行 Agent 评测并生成报告
	$(PY) -m research_agent eval

test:  ## 运行单元测试
	$(PY) -m pytest -q

docker-build:  ## 构建镜像
	docker build -t research-agent:latest .

docker-up:  ## 启动容器
	docker compose up -d --build

clean:  ## 清理缓存与索引
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf data/vector_index/* eval/results/* 2>/dev/null || true

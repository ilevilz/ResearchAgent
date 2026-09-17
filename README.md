# ResearchAgent

> 基于 **ReAct + RAG + MCP** 的科研智能分析 Agent —— 检索论文、解析 PDF、抽取实验指标、对比分析，并给出带引用的分析报告。

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![MCP](https://img.shields.io/badge/protocol-MCP-orange)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 它解决什么问题

做科研调研时，一个真实需求长这样：

> 「帮我找几篇 RAG 的论文，把它们的实验指标抽出来做个对比。」

这件事手动做要 1~2 小时：检索 → 下载 → 读 → 抄表格 → 整理。ResearchAgent 把它变成一次对话，并且**每一步推理和工具调用都可见**（ReAct 轨迹），结论**带原文引用**（RAG 检索片段编号），而不是一个无法验证的答案。

---

## 核心特性

- **ReAct 推理循环** —— Thought → Action → Observation 三段式轨迹，基于 DeepSeek 原生 Function Calling，最多 `MAX_STEPS` 轮后强制收敛
- **混合检索 RAG** —— 稠密向量 + BM25 稀疏召回，**RRF 融合** + 轻量 Rerank，PDF 解析 → 句子级切分 → Embedding → FAISS 索引
- **11 个工具，一次定义三处复用** —— 同一个 `Tool` 对象既是本地函数、又是 Function Calling Schema、又是 MCP 工具定义
- **Research MCP Server** —— 全部工具通过 MCP 协议标准化暴露（`stdio` / `inprocess` 两种传输）
- **四维 Agent 评测** —— 任务成功率 / 工具调用准确率 / 检索准确率 / 延迟，附测试集与报告生成
- **优雅降级** —— FAISS→NumPy、语义 Embedding→Hashing、arXiv→S2→离线库、LLM→正则，任一组件失效都不中断
- **CLI + Web + HTTP API** —— 三种交付形态，Docker 一键部署

---

## 快速开始

### 1. 安装

```bash
git clone https://github.com/<your-name>/ResearchAgent.git
cd ResearchAgent

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-optional.txt            # 可选：faiss-cpu / sentence-transformers
```

### 2. 配置

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 DeepSeek Key：

```ini
DEEPSEEK_API_KEY=sk-your-real-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

### 3. 自检

```bash
python -m research_agent doctor
```

会逐项检查：依赖 → API Key → 向量索引 → 工具注册 → LLM 连通性。

### 4. 跑起来

```bash
# 单次任务
python -m research_agent ask "找几篇 ReAct 相关的论文"

# 交互式多轮对话
python -m research_agent chat

# Web Demo（浏览器打开 http://127.0.0.1:8000）
python -m research_agent serve

# 以 MCP Server 启动
python -m research_agent mcp
```

---

## 使用示例

### CLI

```console
$ python -m research_agent ask "找3篇RAG的论文并抽取它们的实验指标"

[第 1 步] Thought: 需要先检索 RAG 相关论文，再解析 PDF 抽取指标。
          Action: search_papers({"query": "retrieval augmented generation", "max_results": 3})
          Observation: 命中 3 篇（arxiv）：
            1. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2005.11401)
            2. Dense Passage Retrieval for Open-Domain Question Answering (2004.04906)
            3. REALM: Retrieval-Augmented Language Model Pre-Training (2002.08909)
          ⏱ 1240ms

[第 2 步] Thought: 论文摘要中通常不含完整指标，需要下载 PDF 后抽取。
          Action: download_pdf({"arxiv_id": "2005.11401"})
          Observation: 已下载至 data/papers/2005.11401.pdf（1.2MB）
          ⏱ 2103ms

[第 3 步] Thought: PDF 已就绪，解析并抽取实验指标。
          Action: extract_metrics({"pdf_path": "data/papers/2005.11401.pdf"})
          Observation: 抽取到 6 条指标：RAG-Token NQ EM 44.5 / RAG-Sequence NQ EM 44.0 ...
          ⏱ 4820ms

[第 4 步] Thought: 信息已足够，可以对比并给出结论。
          Action: compare_experiments({"records": [...]})
          Observation: 已按 (dataset, metric) 分组对比，best_model = RAG-Token
          ⏱ 12ms

━━━━━━━━━━━━━━━━━━ 最终回答 ━━━━━━━━━━━━━━━━━━
三篇论文中，RAG 在开放域问答上表现最优……

（共 4 步 · 4 次工具调用 · 8.2s · 6420 tokens）
```

### HTTP API

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "总结一下 RAG 的核心贡献", "include_trace": true}'
```

```json
{
  "query": "总结一下 RAG 的核心贡献",
  "answer": "RAG 的核心贡献在于……",
  "success": true,
  "steps": 2,
  "tools_used": ["search_knowledge_base"],
  "latency_ms": 4310.5,
  "total_tokens": 3120,
  "trace": [
    {"index": 1, "thought": "需要检索知识库", "action": "search_knowledge_base",
     "observation": "[1] RAG 论文 (p.1) ...", "ok": true, "latency_ms": 87.2}
  ]
}
```

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health?deep=true` | 健康检查（`deep=true` 真实探活 LLM） |
| `GET` | `/tools` | 工具清单 |
| `POST` | `/chat` | 执行 Agent 任务，返回答案 + ReAct 轨迹 |
| `POST` | `/index` | 论文 / PDF 写入知识库 |
| `POST` | `/search` | 纯检索（不经 Agent） |
| `GET` | `/sessions` · `DELETE` `/sessions/{id}` | 会话记忆管理 |

交互式文档：http://127.0.0.1:8000/docs

### MCP

```bash
python -m research_agent mcp          # stdio 传输，供 Claude Desktop / 任意 MCP Client 接入
```

客户端配置示例：

```json
{
  "mcpServers": {
    "research-agent": {
      "command": "python",
      "args": ["-m", "research_agent", "mcp"],
      "cwd": "/path/to/ResearchAgent"
    }
  }
}
```

---

## 工具清单

| 工具 | 作用 |
| --- | --- |
| `search_papers` | arXiv + Semantic Scholar 论文检索（含离线兜底） |
| `fetch_paper_detail` | 获取单篇论文详情 |
| `download_pdf` | 下载并缓存论文 PDF |
| `parse_pdf` | PDF 解析为分页文本 |
| `index_papers` | 论文 / PDF 写入向量知识库 |
| `search_knowledge_base` | 混合检索知识库（向量 + BM25 + RRF） |
| `extract_metrics` | 从文本 / PDF 抽取实验指标（LLM + 正则兜底） |
| `compare_experiments` | 按 (数据集, 指标) 分组对比实验 |
| `analyze_data` | 表格数据分析：summary / groupby / correlation |
| `preview_table` | 表格预览 |
| `compute_statistics` | 数值列统计（mean / p50 / p95 / std） |

---

## 项目结构

```
ResearchAgent/
├── research_agent/
│   ├── agent/          # ReAct 循环、Prompt、轨迹记录
│   ├── rag/            # loader / chunker / embeddings / vector_store / retriever / knowledge_base
│   ├── tools/          # 11 个工具 + 注册表 + 离线论文兜底库
│   ├── mcp_server/     # Research MCP Server
│   ├── mcp_client/     # MCP 客户端（stdio / inprocess）
│   ├── memory/         # 会话记忆 + 工作记忆
│   ├── eval/           # 测试集 / 四维指标 / 执行器
│   ├── api/            # FastAPI 路由、Schema、服务层
│   ├── core/           # DeepSeek 客户端
│   ├── config.py
│   └── cli.py
├── web/index.html      # Web Demo（零构建，单文件）
├── scripts/build_kb.py # 知识库构建脚本
├── tests/              # 50 个单元测试
├── docs/architecture.md
├── Dockerfile / docker-compose.yml
└── Makefile
```

---

## 评测

评测集 12 条用例，覆盖 6 类场景：`search` / `rag` / `experiment` / `analysis` / `multi_step` / `robustness`。

```bash
python -m research_agent eval
# 报告落盘至 eval/results/latest.md 与 latest.json
```

| 维度 | 指标 | 结果 |
| --- | --- | --- |
| 任务成功率 | Task Success Rate | **100.0%** (13/13) |
| 工具调用准确率 | Tool Calling F1 | **80.3%**（Precision 75.8% / Recall 95.5%） |
| 检索准确率 | Hit Rate / MRR | **100.0%** / **1.000** |
| 延迟 | mean / p50 / p95 | 43.7s / 35.2s / 107.0s |

> **延迟数字需要说明**：本次测量时 arXiv 源站封禁了测试机 IP，每次论文检索要空转
> 约 20s 重试后才降级到离线库，把 search / multi_step 类别的延迟显著抬高。
> analysis / experiment 这类纯本地计算的用例延迟在 **2.3–5.3s**，更能反映系统本身的开销。
> MRR = 1.000 表示检索用例中正确答案**全部排在第 1 位**。

> 检索指标由独立探针测量（绕过 Agent 直接检索），用于区分**检索没召回**与**Agent 没去检索**两类根因 —— 前者优化 RAG，后者优化 Prompt，混在一起就定位不到问题。详见 [架构文档](docs/architecture.md#5-agent-evaluation四维指标)。

---

## 设计取舍

| 依赖 | 首选 | 降级路径 |
| --- | --- | --- |
| 向量索引 | FAISS `IndexFlatIP` | NumPy 精确检索 |
| Embedding | BGE (`sentence-transformers`) | Hashing Embedder |
| 论文检索 | arXiv API | Semantic Scholar → 离线论文库 |
| PDF 解析 | pypdf | pypdfium2 |
| 指标抽取 | LLM 抽取 | 正则模板抽取 |

降级**不静默**：结果中带 `degraded: true` 与 `sources` 字段，`/health` 状态转为 `degraded`，调用方明确知道当前数据质量。

---

## 测试

```bash
make test        # 50 个单元测试，不依赖网络与 LLM
```

覆盖：工具注册与 Schema 有效性、延迟埋点、检索限速、RAG 全链路（切分/Embedding/索引持久化/BM25/RRF）、ReAct 循环（正常收敛 / 步数耗尽强制收敛 / 工具异常不崩溃）、MCP 双传输一致性、API 契约与错误码。

---

## Docker

```bash
docker compose up -d --build
# http://127.0.0.1:8000
```

---

## License

MIT

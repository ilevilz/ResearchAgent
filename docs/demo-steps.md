# Demo 操作步骤（照做即可）

> 所有命令都在 **Git Bash** 里执行。
> 路径固定为 `E:\ResearchAgent`。

---

## 准备：设置环境变量（每个新终端都要做一次）

```bash
cd /e/ResearchAgent
export PY=/e/environment/CustomEnvPath/envs/oct/python.exe
```

> 后面的 `$PY` 都指这个 Python（conda `oct` 环境，装了 faiss / torch / BGE）。

---

## 第 0 步：开场前 3 分钟预热（**必做，否则第一句会卡 20 秒**）

### 终端 A

```bash
cd /e/ResearchAgent
export PY=/e/environment/CustomEnvPath/envs/oct/python.exe

# ① 自检：确认 Key / 知识库 / 工具都就绪
$PY -m research_agent doctor
```
✅ 期望看到：`✅ API Key：已配置`、`文本块：44`、`已注册 11 个`

```bash
# ② 预热 BGE 模型（首次加载约 20 秒，预热后秒回）
$PY -c "from research_agent.rag.embeddings import get_embedder; get_embedder().encode(['预热'])"

# ③ 启动 Web 服务（保持运行，别关）
$PY -m research_agent serve
```
✅ 期望看到：`Uvicorn running on http://0.0.0.0:8000`

### 终端 B

```bash
cd /e/ResearchAgent
export PY=/e/environment/CustomEnvPath/envs/oct/python.exe

# ④ 预热 CLI（加载模型到内存）
$PY -m research_agent ask "你好"
```
✅ 期望看到：一段带 `[Step 1]` 或直接回答的输出，**说明链路通了**

### 检查浏览器

打开 http://127.0.0.1:8000 —— 能看到聊天界面、右上角状态正常。

**到这里准备完毕。下面按顺序演示。**

---

## 第 1 幕：Web 问答（约 2 分钟）—— 最稳，优先演

**在浏览器里输入**：

```
RAG 是怎么结合参数化与非参数化记忆的？
```

**说什么**：
> 「回答里的 `[1] [2]` 是引用编号，对应知识库里的原文片段。检索用的是稠密向量 + BM25 双路召回，再用 RRF 融合——纯向量在专有名词上很弱，而科研场景全是 SQuAD、F1、BLEU-4 这类词，BM25 的精确匹配正好补上。」

**为什么放第一**：纯本地检索，不依赖 arXiv，必成功，延迟约 2-7 秒。

---

## 第 2 幕：CLI 多步任务（约 4 分钟）—— 核心卖点

**在终端 B 输入**：

```bash
$PY -m research_agent ask "找2篇RAG的论文，抽取它们的实验指标并对比"
```

⚠️ **这个要等 1-2 分钟**（arXiv 被封，每次检索空转 ~20 秒重试）。**等待时不要冷场**，指着已经出来的 `[Step N]` 讲：

> 「模型先产出 Thought 再决定调哪个工具，工具的 Observation 会喂回去影响下一步决策——这就是 ReAct 的推理-行动闭环。这里它跨了检索、下载、解析、抽取、对比五类工具。」

**跑完后重点指这段**（如果下载失败，Agent 会这样表现）：
> 「注意它没有编造数字。它明确说了哪些拿不到、为什么拿不到——因为工具返回的结构化 Observation 让它知道自己缺什么。」

---

## 第 3 幕：MCP（约 1 分钟）

**新开终端 C**：

```bash
cd /e/ResearchAgent
export PY=/e/environment/CustomEnvPath/envs/oct/python.exe
$PY -m research_agent mcp
```
（它会挂起等待 stdin，这是正常的——说明 MCP Server 在监听）

**说什么**：
> 「同一套工具，既能被 Agent 通过 Function Calling 调用，也能通过标准 MCP 协议暴露给任意 MCP Client。关键是**一次定义三处复用**——一个 Tool 对象同时是本地函数、Function Calling Schema 和 MCP Tool，不存在改了一处忘另一处的不一致。」

按 `Ctrl+C` 退出。

---

## 第 4 幕：评测（约 30 秒）—— 展示严谨性

**在终端 B 输入**（用子集，秒出）：

```bash
$PY -m research_agent eval --category experiment analysis
```

**说什么**：
> 「四个维度：任务成功率、工具调用 F1、检索准确率、延迟。最有价值的是检索指标用**独立探针**测——绕过 Agent 直接查知识库。这样才能区分失败根因：探针也失败说明检索没召回，该优化 RAG；探针成功但 Agent 没去检索，说明是 Prompt 问题。这两类优化手段完全不同。」

**完整结果直接看 README 的表格**（别现场跑完整评测，要 10 分钟）：
> 「完整 13 条用例的结果：任务成功率 100%，工具 F1 80.3%，检索 MRR 1.000——正确答案全部排在第 1 位。」

---

## 第 5 幕（加分）：翻代码

按顺序打开，每个讲 30 秒：

| 顺序 | 文件 | 讲什么 |
|---|---|---|
| 1 | `research_agent/tools/base.py` | Tool 的三重身份 |
| 2 | `research_agent/agent/react_agent.py` | ReAct 循环 + 步数耗尽强制收敛 |
| 3 | `research_agent/rag/retriever.py` | RRF 融合公式 |
| 4 | `research_agent/eval/metrics.py` | 四维指标 + 独立探针 |

---

## 出问题了怎么办

| 现象 | 处理 |
|---|---|
| 检索结果带 `degraded: true` | **正常演示**，主动解释：「arXiv 封了测试机 IP，三层兜底降级到离线库——这是设计好的，而且降级不静默，结果里明确标了数据来源。」 |
| 终端 A 的 serve 挂了 | 重启：`$PY -m research_agent serve` |
| 端口 8000 被占 | `$PY -m research_agent serve --port 8011`，浏览器改 8011 |
| 某个命令报错 | 直接跳到第 5 幕翻代码，不影响演示完整性 |
| 完全跑不起来 | 展示 README + `docs/architecture.md` 的架构图，改讲设计 |

---

## 一句话速查

```bash
export PY=/e/environment/CustomEnvPath/envs/oct/python.exe

$PY -m research_agent doctor     # 自检
$PY -m research_agent serve      # Web（→ http://127.0.0.1:8000）
$PY -m research_agent ask "..."  # 单次任务（看 ReAct 轨迹）
$PY -m research_agent chat       # 交互式多轮
$PY -m research_agent mcp        # MCP Server
$PY -m research_agent eval       # 评测
$PY -m research_agent tools      # 工具清单
$PY -m pytest -q                 # 67 个单测
```

# 面试演示手册

> 目标：11:00 面试时，用最短路径证明「这个项目是我做的、跑得起来、我知道每一处取舍」。

---

## 0. 开场前 5 分钟（必做）

```bash
cd /e/ResearchAgent
export PY=/e/environment/CustomEnvPath/envs/oct/python.exe   # 用 conda oct 环境

# ① 自检：Key / 索引 / 工具 / LLM 连通性
$PY -m research_agent doctor

# ② 预热：把 BGE 模型加载进内存（首次约 20s，避免演示时卡顿）
$PY -c "from research_agent.rag.embeddings import get_embedder; get_embedder().encode(['预热'])"

# ③ 预热服务（另开一个终端，保持运行）
$PY -m research_agent serve
```

**检查清单**

- [ ] `doctor` 里 DeepSeek Key 显示 ✅
- [ ] 知识库 `num_chunks` > 0 且 `向量后端：faiss`
- [ ] 浏览器 http://127.0.0.1:8000 能打开且右上角状态显示 LLM 已就绪

> ⚠️ **断网预案**：arXiv 会因超频封禁来源 IP，S2 无 Key 时大概率 429。
> 此时检索会**自动降级为离线论文库**（内置 8 篇经典论文），演示依然完整，
> 而且降级这件事本身就是加分项——如实展示 `degraded: true` 并解释设计动机。

---

## 1. 演示主线（约 8 分钟）

### 第 1 幕：单轮 ReAct —— 展示推理轨迹（CLI）

```bash
$PY -m research_agent ask "找3篇RAG相关的论文"
```

**要指出的东西**：终端里打印的 `[第 N 步] Thought / Action / Observation`。
话术：

> 「这不是把问题丢给模型就完事——每一轮模型先产出 Thought，再决定调哪个工具，
> 工具返回的 Observation 会被喂回去影响下一步决策。这就是 ReAct 的推理-行动闭环。
> 底部会汇总步数、工具调用次数、延迟和 token 消耗。」

### 第 2 幕：多步任务 —— 展示任务规划（CLI）

```bash
$PY -m research_agent ask "找3篇RAG的论文，抽取它们的实验指标并对比"
```

**要指出的东西**：跨了 3 个工具（检索 → 下载/解析 → 抽取 → 对比），是真正的多步任务。

### 第 3 幕：RAG 检索质量 —— 展示混合检索（Web）

浏览器打开 http://127.0.0.1:8000 ，输入：

```
RAG 是怎么结合参数化与非参数化记忆的？
```

**要指出的东西**：回答里的 `[1] [2]` 引用编号，对应知识库里的原文片段。

话术（这里是技术深度最强的一段）：

> 「检索用的是混合方案：稠密向量 + BM25，再用 RRF 融合。
> 为什么不用纯向量？因为科研场景全是专有名词和缩写——`SQuAD`、`F1`、`BLEU-4`，
> 这些词向量模型反而弱，BM25 的精确匹配更强。两者互补。
>
> RRF 的好处是只用**排名**不用分数：余弦相似度和 BM25 分数不在一个量纲上，
> 加权求和要先做归一化，而 RRF 的 `Σ 1/(k+rank)` 天然免掉了这个麻烦。」

### 第 4 幕：MCP 标准化（终端）

```bash
# 另开终端，把工具以标准 MCP 协议暴露
$PY -m research_agent mcp
```

话术：

> 「同一套工具，既能让 Agent 通过 Function Calling 调，也能通过 MCP 协议暴露给
> 任意 MCP Client（比如 Claude Desktop）。关键是**一次定义三处复用**：
> 一个 `Tool` 对象同时是本地函数、Function Calling Schema 和 MCP Tool，
> 不存在改了一处忘另一处的不一致。」

### 第 5 幕：评测（终端）

```bash
$PY -m research_agent eval
```

话术：

> 「评测集 13 条用例覆盖 6 类场景，测四个维度：任务成功率、工具调用 F1、
> 检索准确率、延迟分布。
>
> 最有价值的一点是**检索指标用独立探针测**——绕过 Agent 直接查知识库。
> 这样才能区分失败根因：探针也失败说明检索没召回，该优化切分和 Embedding；
> 探针成功但 Agent 没去检索，说明是 Prompt 或工具描述的问题。
> 这两类问题的优化手段完全不同，混在一个指标里就永远定位不到。」

---

## 2. 高频追问与回答要点

**Q：为什么用 Function Calling 而不是解析 ReAct 的文本输出？**

> 保留了三段式轨迹（每步照样有 thought/action/observation），但工具调用走 DeepSeek 原生
> `tool_calls`。原因有三：文本解析对格式极度敏感，模型少个换行就崩；native 调用带 JSON Schema，
> 参数类型由 Schema 约束而不是靠提示词求模型「请传数组」；失败模式从「解析失败」变成
> 「参数不匹配」，后者能被工具层捕获并作为 Observation 回传，Agent 可以自我纠正。

**Q：Agent 会不会死循环 / 无限调工具？**

> 有 `max_steps` 上限，到顶会**强制收敛**——用已有 Observation 直接生成回答，
> 而不是把错误抛给用户。这条有单测覆盖（`test_agent_forces_convergence_at_max_steps`）。

**Q：工具报错怎么办？**

> 工具异常在 `Tool.run()` 里被捕获，转成 `ToolResult(ok=False, error=...)` 作为 Observation
> 回传，Agent 知道失败了可以换个策略，而不是整个任务崩掉。也有单测覆盖。

**Q：检索质量怎么调优的？**

> 三个手段：切分粒度、检索策略、Rerank。当前是 500/80 字符切分 + RRF 混合检索 + 轻量重排。
> 已知不足：Rerank 只做了 query 词重叠加权，没上 cross-encoder；
> 切分也没按论文章节结构走，而实验指标恰恰集中在 Experiment 章节——这是下一步方向。

**Q：项目里最难的问题是什么？**（⭐ 这题用 arXiv 406 的故事，非常加分）

> 论文检索工具间歇性返回 406。我一开始的判断是错的：以为是 httpx 把查询里的冒号
> 百分号编码了，还专门写了个 URL 拼接函数。改完还是 406。
>
> 于是做了对照实验：同一时刻用 4 种 User-Agent（含浏览器 UA）请求，**全部 406**；
> 但几分钟后**又全部 200**。看起来像随机抖动，我一度就按「WAF 抖动」加了三层重试兜底。
>
> 直到我把「已知 URL」和「全新 URL」放在一起对比，才看到真相：
> 响应头里 `cf-cache-status` 是 `MISS, MISS, HIT`、`age=735`——
> **老 URL 命中了 Cloudflare 边缘缓存所以返回 200，新 URL 回源被 406。**
> 所谓「过一会儿又好了」，其实是缓存命中，不是解封。
>
> 根因是**源站按 IP 限流**，而我的重试每次都在 8 秒内打完 4 次，**等于变相加重限流**。
> 最后修的是 `_throttle()`：按主机限速（arXiv 间隔 ≥3s），先限速再退避。
>
> 这件事让我记住：**重试不带限速是有害的**，以及「间歇性故障」要先怀疑自己的观测方式。

**Q：为什么到处都在降级？**

> 因为这是真实的外部依赖环境。FAISS 在部分 Python 版本没有 wheel、BGE 模型要联网下载、
> arXiv 会封 IP、S2 会 429。我的原则是**任一组件失效，功能减弱但不中断，且降级不静默**——
> 结果里带 `degraded: true` 和 `sources`，健康检查状态变 `degraded`，
> 让调用方明确知道当前数据质量，而不是悄悄给一份差的答案。

**Q：如果让你重做，会改什么？**

> ① Rerank 换成 cross-encoder 并用评测集验证收益；
> ② 按论文章节结构切分，而不是固定字符数；
> ③ 做「预算感知」的规划——先规划再执行，而不是走一步看一步。

---

## 3. 万一出问题

| 现象 | 处理 |
| --- | --- |
| 检索显示 `degraded: true` | **正常演示**，这是设计好的降级，主动解释（见断网预案） |
| LLM 调用超时 | 网络问题，重试一次；或在 Web 上换个短问题 |
| 端口被占 | `$PY -m research_agent serve --port 8011` |
| 服务起不来 | 直接演示 CLI，效果完全一样 |
| 想展示代码 | 按这个顺序打开：`tools/base.py`（Tool 三重身份）→ `agent/react_agent.py`（循环）→ `rag/retriever.py`（RRF）→ `eval/metrics.py`（四维指标） |

---

## 4. 一句话总结（被打断时用）

> 「这是一个能跑的科研分析 Agent：ReAct 做任务规划，RAG 做知识检索，MCP 做工具标准化，
> 四维评测驱动迭代。11 个工具、68 个单测、完整降级链路——
> 我更想聊的是每个取舍背后的判断，比如 RRF 为什么比加权求和好、
> 以及那个把我误导了半天的 arXiv 406。」

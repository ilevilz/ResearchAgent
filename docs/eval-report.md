# ResearchAgent 评测报告

- 用例总数：**13**
- 任务成功率 (Task Success Rate)：**100.0%**
- 工具调用准确率 (Tool Calling F1)：**80.3%**（Precision 75.8% / Recall 95.5%）
- 检索准确率 (Retrieval Accuracy)：**100.0%**（MRR 1.000）
- 端到端延迟：**43665 ms**（P50 35247 ms / P95 107014 ms）
- 平均步数：3.69　平均 Token：13165

## 分类表现

| 类别 | 用例数 | 成功率 | 工具 F1 | 平均延迟(ms) |
|------|-------:|-------:|--------:|-------------:|
| analysis | 2 | 100.0% | 83.3% | 3813 |
| experiment | 2 | 100.0% | 83.3% | 2844 |
| multi_step | 2 | 100.0% | 83.3% | 41073 |
| rag | 2 | 100.0% | 75.0% | 96056 |
| robustness | 2 | 100.0% | 0.0% | 18680 |
| search | 3 | 100.0% | 77.8% | 80904 |

## 用例明细

| 用例 | 类别 | 成功 | 工具 F1 | 期望工具 | 实际工具 | 延迟(ms) |
|------|------|:----:|--------:|----------|----------|---------:|
| search-01 | search | ✅ | 1.00 | search_papers | search_papers, search_papers | 63424 |
| search-02 | search | ✅ | 0.67 | search_papers | search_papers, search_papers, fetch_paper_detail, fetch_paper_detail | 72272 |
| search-03 | search | ✅ | 0.67 | search_papers | search_papers, fetch_paper_detail, search_papers, search_papers | 107014 |
| rag-01 | rag | ✅ | 1.00 | search_knowledge_base | search_knowledge_base, search_knowledge_base, search_knowledge_base, search_knowledge_base, search_knowledge_base | 7456 |
| rag-02 | rag | ✅ | 0.50 | search_knowledge_base | search_knowledge_base, search_knowledge_base, search_knowledge_base, search_knowledge_base, search_knowledge_base, search_knowledge_base, search_papers, search_papers, search_papers, search_papers, search_papers, download_pdf, download_pdf, download_pdf, download_pdf, download_pdf, search_knowledge_base, search_knowledge_base, search_knowledge_base | 184655 |
| exp-01 | experiment | ✅ | 1.00 | extract_metrics | extract_metrics | 3219 |
| exp-02 | experiment | ✅ | 0.67 | extract_metrics, compare_experiments | compare_experiments | 2468 |
| ana-01 | analysis | ✅ | 1.00 | compute_statistics | compute_statistics | 2307 |
| ana-02 | analysis | ✅ | 0.67 | analyze_data | analyze_data, preview_table, analyze_data | 5318 |
| multi-01 | multi_step | ✅ | 1.00 | search_papers, index_papers, search_knowledge_base | search_papers, index_papers, search_knowledge_base, search_knowledge_base | 43343 |
| multi-02 | multi_step | ✅ | 0.67 | search_papers | search_papers, fetch_paper_detail | 38802 |
| robust-01 | robustness | ✅ | 0.00 | - | search_papers | 35247 |
| robust-02 | robustness | ✅ | 0.00 | - | preview_table | 2112 |
# 写作工厂 2.0

面向人文社科论文写作的本地桌面应用。阶段 0–8 已贯通：文档入库与混合检索、人物 / 主题 PersonaSpec 蒸馏、带事实硬门的五阶段写作循环、代码引用拼装、长文一致性、评估与注入防御，以及可恢复、可编辑的 PyQt6 项目工作流。

## 开发环境

- Python 3.12
- `uv`
- Windows / UTF-8

依赖源已在 `pyproject.toml` 固定为清华镜像。首次安装：

```powershell
uv sync --all-groups --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

项目根目录的 `key_test.txt` 使用两行原始 token：第一行为 SiliconFlow API key，第二行为 MinerU API token。该文件已被 Git 忽略。生产环境可改用 `SILICONFLOW_API_KEY` 和 `MINERU_API_TOKEN` 环境变量。

## 运行

```powershell
uv run writing-factory
```

知识库页支持一次多选 PDF、Word、PPT 和 UTF-8 TXT，并在单个后台任务中按顺序入库，避免并发占用 MinerU 配额。PDF/Word/PPT 通过 MinerU 解析，TXT 使用本地兜底 loader；导入文件会复制到内容寻址的本地托管目录。当前按文件名生成默认书目标题，不弹出书目确认对话框。

作者档案页可分别勾选目标语料和可选的同领域对照语料，运行可恢复的两级学术蒸馏：切片 Map 先按论文归并，再做跨论文聚类、留出生成力检验和对照排他性检验。同一批语料只生成一个顶层档案，新蒸馏结果作为版本保存。核心列表保留 3–7 个模型并优先选择作者个性化模型；通用模型仅在不足 3 个时补位，其余进入“通用学术惯例”，部分通过验证的候选确定性降级为启发式。表达统计由本地代码计算，矛盾与信息不足不会被自动补造。付费的 Nüwa 保真度自检只在用户点击“自检”后运行，并由独立的出题、作答和中性评分调用组成。

双击档案可查看和编辑完整审计 JSON、Markdown、无证据运行时投影及版本历史。生成阶段只允许使用运行时投影，蒸馏证据锚点和旧论文来源不会随作者模型传入；若写作任务没有明确授权，作者语料也会从新任务事实来源中排除。设置页的 SiliconFlow 最大并发数统一约束 chat、embedding 和 rerank 请求，并持久化到本地 SQLite。

运行数据位于被 Git 忽略的 `data/`：SQLite 保存规范文本、精确字符区间、蒸馏断点和 PersonaSpec，LanceDB 保存 bge-m3 向量，BM25 在启动或语料变化后通过 SQLite 文本和 jieba 分词确定性重建。长推理调用通过统一客户端读取 SSE 流，并在每个 Map 单元完成后立即提交断点。

知识库页底部提供非阻塞的检索测试面板。阶段 3 默认并发执行查询改写与 HyDE，将原查询、去重后的子查询和 HyDE 文本合并成一次 embedding 批请求；随后对 bge-m3 与 BM25/jieba 的多路结果累计 RRF，扩展父级上下文，再用 bge-reranker-v2-m3 重排。元数据过滤在 SQLite 先解析成统一的允许范围，同时约束稠密与稀疏检索。返回父块时始终保留精确的 `matched_child_ids`，供后续事实核对和引用拼装回到原始小块。知识库内容变化会改变检索缓存指纹，不会复用旧结果。查询改写与 HyDE 可在设置页分别关闭，所有 SiliconFlow 调用仍受同一个全局并发数约束。

项目页用于创建和管理论文项目。写作任务页选择项目、作者档案和本任务事实语料后启动生成；作者蒸馏语料默认从事实白名单排除，确需引用时必须显式开启复用选项。流水线按“选题 → 框架 → 逐节证据锁定与起草 → 中性核对 → 文风打磨 → 全局一致性”运行，每个 `fact` 论断必须绑定本节真实 child chunk 和正文引用标记。核对存在 `partial` 或 `unsupported` 时会循环修订，达到上限仍不通过则终止，不会进入成稿。

每个任务使用独立 LangGraph SQLite checkpoint，并在业务数据库保存配置、状态、成稿和评估结果。停止任务后可从项目任务列表继续；双击任务可载入结果，论文和提纲允许人工编辑并保存。参考文献只收集正文实际使用且核对通过的来源，通过 citeproc-py 按 GB/T 7714 生成。

评估页同时给出代码计算的引用可溯性、幻觉率、证据上下文忠实度和中立 LLM Judge 结果。检索文本进入任何生成 prompt 前先经过注入检测，并始终位于明确的数据边界内；检测或评估服务失败时按失败闭合处理，不会伪装成安全或合格结果。

## 测试

```powershell
uv run pytest
```

真实 SiliconFlow smoke test 默认跳过，显式启用：

```powershell
$env:RUN_LIVE_API_TESTS="1"
uv run pytest tests/integration/test_siliconflow_live.py
```

完整 MinerU 入库测试需要指定本地文件和能命中正文的查询：

```powershell
$env:RUN_LIVE_INGEST_TESTS="1"
$env:LIVE_INGEST_FILE="C:\path\paper.pdf"
$env:LIVE_INGEST_QUERY="文档中的关键词"
uv run pytest tests/integration/test_stage1_live.py
```

阶段 3 的真实语料回归只调用 SiliconFlow，不调用 MinerU。查询必填，预期文件名可选：

```powershell
$env:RUN_LIVE_RETRIEVAL_TESTS="1"
$env:LIVE_RETRIEVAL_QUERY="你的检索问题"
$env:LIVE_RETRIEVAL_EXPECTED_FILENAME="应命中的文件名片段"
uv run pytest tests/integration/test_stage3_live.py
```

上述集成测试默认跳过，因为它们需要本地语料、外部服务或付费额度；普通 `pytest` 始终运行全部离线测试。

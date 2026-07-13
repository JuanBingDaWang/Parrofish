# 写作工厂 2.0

面向人文社科论文写作的本地桌面应用。当前完成阶段 0–2：项目骨架、统一外部服务客户端、SQLite 状态与调用记录、非阻塞 PyQt6 外壳、可追溯的双索引知识库入库链，以及人物 / 主题两模式的 PersonaSpec 蒸馏链。

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

# 写作工厂 2.0

面向人文社科论文写作的本地桌面应用。当前完成阶段 0–1：项目骨架、统一外部服务客户端、SQLite 状态与调用记录、非阻塞 PyQt6 外壳，以及可追溯的双索引知识库入库链。

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

知识库页支持直接导入 PDF、Word、PPT 和 UTF-8 TXT。PDF/Word/PPT 通过 MinerU 解析，TXT 使用本地兜底 loader；导入文件会复制到内容寻址的本地托管目录。当前按文件名生成默认书目标题，不弹出书目确认对话框。

运行数据位于被 Git 忽略的 `data/`：SQLite 保存规范文本、精确字符区间和元数据，LanceDB 保存 bge-m3 向量，BM25 在启动或语料变化后通过 SQLite 文本和 jieba 分词确定性重建。

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

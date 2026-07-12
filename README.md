# 写作工厂 2.0

面向人文社科论文写作的本地桌面应用。当前完成阶段 0：项目骨架、统一外部服务客户端、SQLite 状态与调用记录，以及非阻塞 PyQt6 外壳。

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

## 测试

```powershell
uv run pytest
```

真实 SiliconFlow smoke test 默认跳过，显式启用：

```powershell
$env:RUN_LIVE_API_TESTS="1"
uv run pytest tests/integration/test_siliconflow_live.py
```


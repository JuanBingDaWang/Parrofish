# 开发协作规范（AGENTS.md）

> 交给 Codex 的工作约定。建议放在仓库根目录并命名为 `AGENTS.md`，Codex 会自动加载。
> 本文件是《项目说明书：人文社科学术写作助手》的配套件：说明书讲"做什么"，本文件讲"怎么做、环境怎么配、遵守哪些工程约定"。

---

## 0. 总则（工作方式）

- **质量优先，不要为省 token 牺牲正确性。** 该读依赖文档就读、该读 Nüwa 源码就读、该查 API 用法就查。宁可多花时间搞清楚，不要凭猜实现。
- **增量开发。** 严格按项目说明书第十三节的阶段顺序推进：一个阶段能跑通、能测，再进下一个。**禁止一次性生成整个项目。**
- **规格不明先问，不要臆测行为。** 尤其涉及引用格式、知识库 schema、PersonaSpec 结构、写作流水线的边界条件时，先向用户确认。
- **每个阶段开工前，先复述项目说明书第三节的"设计铁律"，并说明本阶段如何不违反它们。** 那八条是正确性底线，赶实现时最容易被悄悄绕过（尤其"事实先冻结、文风最后加""引用由代码拼装不由模型敲""不让作者校验自己"）。

---

## 1. 模块化（禁止 god-file）

**明确禁止把所有东西写进一个 `.py`。** 按职责分包，单一职责，函数短小，跨模块通过明确的数据契约（对应说明书第五节的数据模型）通信。这样改动只碰相关文件、不必重读整个大文件——既省 token 又易维护易测。

建议目录结构：

```
project/
├── config/          # 配置与密钥加载；模型名 / base_url / 镜像地址 / 引用样式集中在此
├── llm/             # 外部服务客户端封装（SiliconFlow chat/embedding/rerank + MinerU）
├── store/           # SQLite + LanceDB + BM25 索引的读写
├── kb/              # 入库链：解析 → 切片 → 双索引；以及检索器
├── distill/         # 蒸馏器：map-reduce → PersonaSpec（人物 / 主题两模式）
├── generate/        # 生成流水线：选题 / 框架 / 起草 / 核对 / 打磨 + claim typing
├── orchestration/   # LangGraph 写作循环、状态、checkpointer
├── eval/            # recall@k、引用可溯性、RAGAS、LLM-judge、黄金回归集
├── ui/              # PyQt 界面 + worker 线程
└── main.py          # 只做装配与启动，不写业务逻辑
```

- 每个模块顶部写简短 docstring 说明职责；公共接口加类型注解。
- 一个文件过长（例如超过 300–400 行）就考虑拆分。

---

## 2. 统一的外部服务客户端（Day 1 就建，最关键的一条）

在 `llm/` 里建薄封装层，统一封装 SiliconFlow（chat / embedding / rerank）、MinerU 和可选博查 Web Search。**所有业务模块只调这个封装，不碰裸 HTTP。**

这一层统一处理：
- 重试、限流、超时、错误处理；
- 缓存（相同请求不重复烧钱）；
- 思考 / 非思考模式切换（V4-Flash 的 reasoning_effort），让"不同步骤用不同模式"有单一入口；
- token 用量统计（配合第 4 节日志）。

理由：不这样做，API 调用会散落到各处，成本不可控、改不动、也无法统一观测。**这条比模块化还优先。**

---

## 3. 配置与密钥

- SiliconFlow、MinerU、博查的 API key 放 `.env`、系统凭据库或本地配置文件，**绝不硬编码、绝不提交**（写进 `.gitignore`）。
- `config/` 里做一个集中配置模块：模型名、base_url、镜像地址、引用样式选择，全部在一处可改。

---

## 4. 日志与可观测性

- 记录每一次 LLM 调用：prompt 摘要、模型、模式、token 数、耗时、结果，落到 SQLite 或日志文件。
- 多阶段 + 成本敏感 + 需要调试 → 这是刚需。也让你随时看得见 token 花在哪（"质量优先不省 token"不等于"看不见花销"）。

---

## 5. 可恢复性（断点续跑）

- 写一篇论文是几十次调用的长任务，中间任何一步都可能失败。
- 中间产物（提纲、证据包、草稿、核对结果）必须持久化（数据模型里已经设计好）。
- 用 LangGraph 的 SQLite checkpointer 做断点续跑：阶段 4 失败不要重跑 1–3。

---

## 6. 测试与评估随行

- 每条流水线配最小 smoke test + 项目说明书第九节的指标（检索 recall@k、生成引用可溯性）。**边写边测，不要留到最后。**
- 提取 / 核对这类需要稳定性的步骤，用低温度 / 固定 seed 保证可复现；起草 / 打磨可用较高温度。

---

## 7. 中国大陆网络与镜像

所有从 pip / npm / GitHub 拉资源的操作都要走镜像。**GitHub 代理服务经常失效，遇到不通先换一个，别死磕。**

**pip（清华源，稳定）：**
```bash
# 临时
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <包名>
# 永久
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
（推荐用 `uv` 装依赖，速度快，同样认 index-url 镜像。）

**npm（`npx skills add` 会用到）：**
```bash
npm config set registry https://registry.npmmirror.com
```

**git clone GitHub（两种择一）：**
```bash
# 方式一：全局 insteadOf，之后用原始链接 clone 即可自动走镜像
git config --global url."https://gitclone.com/github.com/".insteadOf "https://github.com/"

# 方式二：per-clone 加代理前缀
git clone https://gh-proxy.com/https://github.com/<user>/<repo>.git
```
代理域名会变动：若上面不通，去 `https://ghproxy.link/` 查当前可用的加速域名；只需在线浏览源码则用 `bgithub.xyz` 或 `kkgithub.com`。

**HuggingFace（本项目一般用不到）：** 模型全部走 SiliconFlow API，通常无需在本地下载模型权重。万一某个依赖要从 HF 拉东西，设环境变量 `HF_ENDPOINT=https://hf-mirror.com`。

---

## 8. Nüwa 的定位与安装

**先澄清定位：** 按项目说明书，最终程序**运行时并不调用 Nüwa**，而是照它的提炼方法论自研一个 map-reduce 蒸馏器（产出 PersonaSpec）。所以在本项目里，安装 Nüwa 的目的主要是**让开发者和 Codey 读懂它的方法论与目标产物格式**，它是参考资料，不是运行时依赖。

步骤：
1. **安装 / 克隆到本地**（走镜像）。Codex 本身是 Nüwa 支持的 runtime，可以 `npx skills add alchaincyf/nuwa-skill`（走 npm 镜像 + GitHub 加速）把它装进自己；或直接 `git clone` 仓库。
2. **精读三份东西**：`references/extraction-framework.md`（提炼方法论）、`references/skill-template.md`（产物模板）、`examples/` 里 1–2 个人物（如 `steve-jobs-perspective`，含多轮对话 demo），理解目标行为与输出结构。
3. **可以在 Codex 里实际用一下装好的 skill**，建立直觉。
4. **然后再实现我们自己的 PersonaSpec 蒸馏器**，忠于它的方法：心智模型三重验证（跨域复现 / 有生成力 / 有排他性，保留 3–7 个）、表达 DNA 量化（句式指纹 / 风格标签 / 禁忌词与口癖）、矛盾要保留而非调和、信息不足要标注。
5. **继承 Nüwa 的诚实边界**：抓不到直觉 / 灵感、只是调研截止日的快照、公开言论 ≠ 真实想法。

---

## 9. PyQt 并发约定（强制）

- 所有 LLM / 网络 / MinerU 调用**一律走 QThread worker + 信号，或 qasync**；GUI 线程绝不执行阻塞调用，否则界面卡死。
- 早点定一个标准 worker 模式，全项目复用。

---

## 10. 编码与中文处理

- 全程 UTF-8。文件读写、jieba 分词、BM25 索引、日志、控制台输出都要能正确处理中文（Windows 上尤其注意默认编码问题，显式指定 `encoding="utf-8"`）。

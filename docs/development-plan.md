# 行人流方向问答智能体 (Ped-Agent) — 开发计划

## Context

需要从零构建一个行人流方向的问答智能体系统。该系统需具备：自主检索补充学术文献的能力、实验方案评估、真实场景数据结构化分析与建议生成，以及预留视觉模块（目标检测 + 轨迹提取）。项目目录当前为空，属于全新开发。

---

## 系统架构总览

```
用户查询 → Agent Graph (LangGraph) → 路由分发
                                        ├─ 文献问答 → RAG 检索 → 合成回答
                                        ├─ 实验评估 → 评估标准 + 文献支撑 → 评分建议
                                        ├─ 数据分析 → 指标计算 + 可视化 → 结论
                                        └─ 推荐建议 → 综合分析 → 建议输出

视觉模块 (可选) → 视频 → YOLO26 + ByteTrack → 轨迹数据 → 分析管道
```

---

## 目录结构

```
ped-agent/
├── pyproject.toml
├── config/
│   ├── default.yaml          # 主配置
│   ├── llm.yaml              # LLM 后端配置
│   ├── rag.yaml              # RAG 管道配置
│   ├── vision.yaml           # 视觉模块配置
│   └── langsmith.yaml        # LangSmith 追踪与评估配置
├── src/ped_agent/
│   ├── main.py               # 入口
│   ├── agent/                # Agent 编排层 (LangGraph)
│   │   ├── graph.py          # 状态机定义
│   │   ├── nodes.py          # 路由、检索、分析、合成节点
│   │   ├── state.py          # AgentState 类型定义
│   │   ├── tools.py          # LangChain Tool 定义
│   │   └── prompts/          # LangChain PromptTemplate (支持 LangSmith 版本管理)
│   ├── llm/                  # LLM 后端 (LangChain 集成)
│   │   ├── factory.py        # LLM 工厂函数 (Claude/GPT/DeepSeek/Qwen/GLM/Ollama)
│   │   └── callbacks.py      # 自定义 LangChain callbacks
│   ├── knowledge/            # RAG + 文献管理
│   │   ├── retriever.py      # HybridRetriever (BGE-M3 dense+sparse + optional graph)
│   │   ├── vector_store.py   # Chroma/Qdrant/Milvus vector store adapter
│   │   ├── embeddings.py     # BGE-M3 wrapper (FlagEmbedding) + dense fallback
│   │   ├── indexer.py        # 文档加载、切分、索引 (RecursiveCharacterTextSplitter)
│   │   ├── sources/          # 文献源适配器
│   │   │   ├── semantic_scholar.py
│   │   │   ├── arxiv.py
│   │   │   ├── cnki.py       # 知网
│   │   │   └── wanfang.py    # 万方
│   │   └── parsers/          # 文献解析与格式转换
│   │       ├── marker_parser.py
│   │       ├── grobid_parser.py
│   │       └── caj_converter.py
│   ├── analysis/             # 结构化数据分析
│   │   ├── pipeline.py       # 分析编排
│   │   ├── metrics.py        # 密度/速度/流量指标
│   │   ├── od_matrix.py      # OD 矩阵计算
│   │   ├── fundamental_diagram.py  # 基本图拟合
│   │   └── visualizer.py     # 图表生成
│   ├── experiment/           # 实验方案评估
│   │   ├── evaluator.py      # 评估逻辑
│   │   └── criteria.py       # 评估标准与评分规则
│   ├── vision/               # 视觉模块 (可插拔)
│   │   ├── interface.py      # VisionBackend Protocol
│   │   ├── registry.py       # 插件注册表
│   │   ├── schemas.py        # TrajectoryData 标准化输出
│   │   └── plugins/
│   │       ├── yolo26_bytetrack.py
│   │       └── yolo26_deepsort.py
│   ├── models/               # Pydantic 数据模型
│   │   ├── trajectory.py     # 轨迹数据
│   │   ├── literature.py     # 论文/引用
│   │   └── scenario_data.py  # 场景/分析结果
│   ├── evals/                # LangSmith 评估与回归测试
│   │   ├── datasets.py       # 评估数据集定义
│   │   ├── evaluators.py     # 自定义 LangSmith evaluator
│   │   └── experiments.py    # A/B 测试脚本
│   └── utils/
│       ├── config.py         # OmegaConf 配置加载
│       ├── logging.py
│       └── langsmith.py      # LangSmith 初始化与追踪装饰器
├── tests/
│   ├── unit/
│   ├── integration/
│   └── langsmith_eval/       # LangSmith 评估测试
├── scripts/
│   ├── index_papers.py       # 批量索引文献
│   ├── run_vision.py         # 独立视觉管道
│   └── evaluate_agent.py     # 运行 LangSmith 评估
└── data/                     # 本地数据目录
    ├── vectordb/             # 向量数据库持久化
    ├── papers/               # 已下载论文
    └── scenarios/            # 场景数据
```

---

## 核心技术选型

| 层次 | 选型 | 理由 |
|------|------|------|
| Agent 编排 | LangGraph 1.x | 显式状态机，检查点持久化，人在回路 (interrupt()) |
| LLM 框架 | LangChain 1.x + provider packages | 统一模型/工具接口；必要时用 LangGraph 控制复杂状态 |
| LLM 后端 | langchain-anthropic / langchain-openai / langchain-community | 统一接口；DeepSeek/Qwen/GLM 通过 OpenAI 兼容协议接入 |
| 可观测性与评估 | LangSmith | 全链路追踪、提示管理、评估数据集、A/B 测试 |
| Embedding | BAAI/bge-m3 (FlagEmbedding) | dense + sparse + ColBERT 三模式，单模型替代多模型 |
| 向量库 | Chroma (开发) → Qdrant (生产) → Milvus (规模化) | 分阶段升级 |
| 混合检索 | BGE-M3 dense+sparse + RRF + reranker | 不再维护独立 BM25 索引；必要时自定义 HybridRetriever |
| 重排序 | bge-reranker-v2-m3 (Cross-encoder) | 提升 10-15% NDCG@10 |
| 引用图 | Neo4j | Citation Graph RAG，论文关系追踪 |
| PDF 解析 | Marker + GROBID | Marker (10x 速度) + GROBID (结构化元数据) |
| 工具接入 | MCP Servers + langchain-mcp-adapters | 动态工具发现，JSON-RPC 2.0 标准协议 |
| Agent 记忆 | LangGraph Checkpoint + Qdrant + Mem0 | 分层：会话/知识/偏好 |
| 数据模型 | Pydantic v2 | 验证、序列化、类型安全 |
| 配置 | OmegaConf | YAML 合并 + 环境变量插值 |
| 数据分析 | NumPy / Pandas / SciPy | 数值计算 |
| 可视化 | Plotly / Matplotlib | 交互式 + 静态图表 |
| 视觉检测 | ultralytics (YOLO26) | 官方 YOLO26 检测模型，适合离线高精度行人检测 |
| 多目标跟踪 | boxmot | ByteTrack/DeepSORT 统一接口 |
| 测试 | pytest + pytest-asyncio + LangSmith evaluate | 异步测试 + LLM 应用质量评估 |
| RAG 调参 | AutoRAG | 自动化管道组件评估与优化 |

---

## 关键设计决策

1. **LangGraph 状态机 + 渐进式多 Agent** — 初期单 Agent (Self-Reflective ReAct)，当 LangSmith 监测到上下文污染时拆分为 Hierarchical Teams（Literature Agent / Analysis Agent / Evaluation Agent）。

2. **LangChain + MCP 双轨工具接入** — LangChain `@tool` 装饰器用于核心工具；MCP Servers 用于动态发现外部数据源（学术数据库、文件系统、实验数据库），通过 `langchain-mcp` 桥接。

3. **BGE-M3 单模型三模式检索** — dense + sparse + ColBERT 同时输出，sparse 替代独立 BM25，减少索引维护成本；加 bge-reranker-v2-m3 重排序提升 10-15%。

4. **Marker + GROBID 学术 PDF 解析** — Marker 提供 10x 速度和高格式保真度，GROBID 补充结构化元数据（DOI/作者/引用），比 PyMuPDF 在学术场景精度显著提升。

5. **Neo4j Citation Graph RAG** — 论文引用关系存入图数据库，检索时结合向量相似度 + 图遍历，支持"这篇论文引用了什么"/"谁引用了这篇"等链式查询。

6. **分层记忆架构** — L0 会话 (LangGraph State) → L1 检查点 (PostgresSaver) → L2 知识 (Qdrant) → L3 关系 (Neo4j) → L4 偏好 (Mem0, 中期引入)。

7. **Protocol-based 视觉接口** — 用 `typing.Protocol` 定义结构化子类型，视觉插件无需继承基类，第三方扩展零耦合。

8. **LangSmith 评估驱动开发** — 每个 Phase 建立评估数据集，量化跟踪 Citation Accuracy、Retrieval Precision、Task Completion，防止迭代过程中性能退化。

9. **可选依赖分组** — 视觉模块通过 `pip install ped-agent[vision]` 安装，核心功能不依赖重量级 CV 库。

10. **LangSmith 可观测性与评估** — 所有 LLM 调用、检索、工具调用通过 LangSmith 追踪；建立评估数据集（文献问答、实验评估）定期回归测试；提示模板托管在 LangSmith Hub，支持版本迭代和 A/B 测试。

---

## LLM 多后端配置

所有国产/国际大模型通过 LangChain `BaseChatModel` 统一接入，配置示例：

```yaml
# config/llm.yaml
llm:
  default_provider: "claude"
  providers:
    claude:
      model: "claude-sonnet-4-20250514"
      api_key: "${oc.env:ANTHROPIC_API_KEY,null}"
      max_tokens: 4096
      temperature: 0.1
    openai:
      model: "gpt-4o"
      api_key: "${oc.env:OPENAI_API_KEY,null}"
      max_tokens: 4096
    deepseek:
      model: "deepseek-chat"
      api_key: "${oc.env:DEEPSEEK_API_KEY,null}"
      base_url: "https://api.deepseek.com/v1"
    qwen:
      model: "qwen-plus"
      api_key: "${oc.env:DASHSCOPE_API_KEY,null}"
      base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    glm:
      model: "glm-4-plus"
      api_key: "${oc.env:ZHIPU_API_KEY,null}"
      base_url: "https://open.bigmodel.cn/api/paas/v4"
    local:
      model: "qwen2.5:72b"
      base_url: "http://localhost:11434"
```

**适配原理**：DeepSeek、Qwen (DashScope)、GLM (智谱) 均提供 OpenAI 兼容 API，在 `factory.py` 中统一使用 `ChatOpenAI(base_url=..., model=..., api_key=...)` 接入，无需额外依赖包。

---

## Agent 工作流设计

### 查询路由

```
classify_query → route_by_type
  ├─ "literature_qa"   → retrieve_literature → synthesize
  ├─ "experiment_eval" → retrieve + evaluate → synthesize
  ├─ "data_analysis"   → load_data → compute_metrics → visualize → synthesize
  └─ "recommendation"  → retrieve + analyze → generate_recommendations → synthesize
```

### Agent 可调用的工具 (LangChain Tool)

```python
from langchain.tools import tool

@tool
def search_literature(query: str, sources: list[str], max_results: int = 20) -> str:
    """在线搜索学术数据库（Semantic Scholar/arXiv/CNKI/万方）"""
    ...

@tool
def retrieve_knowledge(query: str, filters: dict | None = None) -> str:
    """从本地向量库检索已索引的文献"""
    ...

@tool
def load_scenario_data(scenario_id: str) -> str:
    """加载场景数据（CSV/JSON）"""
    ...

@tool
def compute_metrics(data: str, metric_types: list[str]) -> str:
    """运行分析管道（密度/速度/流量/OD 矩阵）"""
    ...

@tool
def extract_trajectories(video_path: str, config: dict) -> str:
    """调用视觉插件从视频提取轨迹"""
    ...

@tool
def evaluate_plan(plan_text: str, criteria: list[str]) -> str:
    """评估实验方案"""
    ...

@tool
def generate_chart(data: str, chart_type: str) -> str:
    """生成可视化图表并返回路径"""
    ...
```

所有 Tool 调用自动记录到 LangSmith。

---

## 视觉模块接口设计

```python
@runtime_checkable
class VisionBackend(Protocol):
    def configure(self, config: dict) -> None: ...
    def process_video(self, video_path: Path, roi: ROI | None = None) -> TrajectoryData: ...
    def process_frame(self, frame: np.ndarray) -> list[Detection]: ...
    
    @property
    def capabilities(self) -> set[str]: ...

# 标准化输出
class TrajectoryData(BaseModel):
    video_meta: VideoMetadata
    tracks: list[PedestrianTrack]  # 每条轨迹: track_id, positions, timestamps, confidence
```

核心模块仅依赖 `TrajectoryData` schema，不导入任何 `vision.plugins.*`。

---

## 开发阶段规划

### Phase 1 — 基础框架 (1-2 周)
- 项目脚手架：pyproject.toml、配置系统、日志
- Pydantic 数据模型
- LLM 抽象层 (Claude + OpenAI 后端)
- 基础 Agent Graph (路由 + 单路径响应)
- pytest 测试框架

### Phase 2 — RAG 管道 (2 周)
- Embedding 集成 (`FlagEmbedding.BGEM3FlagModel` + BGE-M3；dense-only fallback 可用 `langchain_huggingface.HuggingFaceEmbeddings`)
- ChromaDB 向量库 (`langchain-chroma`)
- PDF 解析器（Marker 主解析器 + GROBID 元数据；PyMuPDF 仅作为轻量 fallback）
- 文档切分 (`langchain.text_splitter.RecursiveCharacterTextSplitter`)
- Semantic Scholar + arXiv 适配器
- 混合检索实现（BGE-M3 dense+sparse + RRF；独立 BM25 不作为默认路径）
- LangSmith 追踪集成（环境变量配置）
- 文献问答端到端流程

### Phase 3 — 中文文献源 (1 周)
- 知网适配器 (CNKI)
- 万方适配器
- CAJ 格式解析
- 双语查询扩展

### Phase 4 — 数据分析模块 (2 周)
- 行人流指标计算 (密度/速度/流量)
- OD 矩阵构建
- 基本图拟合
- 可视化生成
- Agent Graph 集成

### Phase 5 — 实验评估模块 (1 周)
- 评估标准与评分规则定义
- LLM 结构化评分
- 推荐建议生成

### Phase 6 — 视觉模块 (2 周)
- 插件注册系统
- YOLO26 + ByteTrack 实现
- 像素→世界坐标变换
- 与分析管道对接
- 可选依赖打包

### Phase 7 — 评估体系与完善 (1-2 周)
- LangSmith 评估数据集构建（文献问答 30 题、实验评估 20 题）
- 自定义 evaluator（引用准确性、分析完整性、推荐可行性）
- A/B 测试框架（不同提示模板、不同检索策略）
- FastAPI 服务层 (可选)
- 会话记忆管理 (`langchain.memory`)
- 错误处理与优雅降级
- 集成测试
- 文档

---

## 系统建议

1. **从 Phase 1-2 快速验证** — 先跑通"文献问答"这条最小闭环，确认 RAG 效果再扩展其他功能。

2. **LangSmith 从 Phase 1 就开启** — 配置 `LANGSMITH_TRACING=true` 和 `LANGSMITH_API_KEY`，所有开发阶段自动记录调用链，便于调试和后续评估。

3. **中文文献源可能需要逆向** — CNKI/万方没有稳定公开 API，需评估爬虫合规性或考虑替代方案（如已有本地 PDF 库直接索引）。

4. **视觉模块独立开发** — 插件化设计允许视觉模块与核心并行开发，互不阻塞。

5. **优先使用 Claude 作为默认 LLM** — 上下文窗口大、中文能力强，适合长文献综合分析；通过 `langchain-anthropic` 集成；本地模型作为离线降级方案。

6. **数据分析标准化** — 定义统一的场景数据 schema（JSON/CSV），确保不同来源的数据可直接进入分析管道。

7. **提示模板版本管理** — 关键提示模板托管在 LangSmith Hub，通过 API 拉取，支持无代码更新和 A/B 测试。

8. **建立评估基准** — Phase 2 完成后立即建立初版评估数据集（至少 20 题），后续每次重构都跑回归测试，防止性能退化。

---

## 验证方法

- **Phase 1**: `pytest` 运行通过，Agent 能正确路由简单查询，LangSmith 追踪可见

- **Phase 2**: 
  - 索引 5-10 篇论文后，文献问答能返回带引用的相关回答
  - LangSmith 追踪显示检索命中率和 token 消耗
  - 建立初版评估数据集（20 题），基准准确率 > 70%

- **Phase 4**: 输入样例行人流数据，生成正确的密度/速度图表

- **Phase 6**: 输入短视频片段，输出轨迹数据并传入分析管道

- **Phase 7**: 
  - 完整评估数据集（50 题）回归测试，准确率 > 80%
  - A/B 测试验证新提示模板优于基线 5% 以上

- **全流程**: 用户提出复合问题（如"评估这个实验方案并推荐改进"），Agent 能组合多个工具完成回答，LangSmith 追踪显示完整调用链

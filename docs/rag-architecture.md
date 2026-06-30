# 行人流问答智能体 RAG 方案详细设计

## 研究背景

基于对主流开源 RAG 项目、Agent 框架、学术文献系统的调研，本文档提供生产级 RAG 架构设计。

---

## 一、学术 RAG 核心挑战

### 1.1 与通用 RAG 的差异

| 维度 | 通用 RAG | 学术 RAG |
|------|---------|---------|
| 文档类型 | Markdown/HTML/纯文本 | PDF（公式/表格/多栏）、CAJ |
| 引用要求 | 无 | 必须准确溯源到原文 |
| 术语密度 | 低 | 高（领域专有术语） |
| 跨语言 | 单语为主 | 中英双语混合 |
| 更新频率 | 静态或增量 | 需持续补充新文献 |

### 1.2 主要技术瓶颈

1. **Vanilla RAG 失败率 40%** (来源: hybrid RAG 论文研究)
2. **PDF 解析精度** — 公式、表格、图表损失严重
3. **引用链追踪** — 检索到的片段难以关联到完整 citation
4. **中文学术数据库** — CNKI/万方无公开 API，需爬虫
5. **领域术语** — 行人流领域的 "社会力模型"、"基本图" 等专有词汇

---

## 二、整体架构：Hybrid Graph RAG

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户查询                                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│ 查询优化层 (Query Transformation)                                 │
│  ├─ HyDE (Hypothetical Document Embeddings)                      │
│  ├─ Multi-Query (生成多角度子查询)                                │
│  └─ Step-Back (抽象层次提升)                                      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│ 混合检索层 (Hybrid Retrieval)                                     │
│  ├─ 语义检索 (BGE-M3 dense vectors)                               │
│  ├─ 稀疏检索 (BGE-M3 lexical weights)                             │
│  ├─ Graph 检索 (Citation Graph - Neo4j)                          │
│  └─ Reciprocal Rank Fusion (RRF) 融合                            │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│ 重排序层 (Reranking)                                              │
│  └─ Cross-encoder reranker (bge-reranker-v2-m3)                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│ 上下文压缩层 (Contextual Compression)                             │
│  └─ LLM-based extractive compression                             │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│ LLM 生成 + Citation 注入                                          │
│  └─ 带引用格式的答案生成                                           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 三、核心技术栈

### 3.1 PDF 解析方案

| 工具 | 用途 | 优势 | 劣势 |
|------|------|------|------|
| **Marker** | 主解析器 | 10x 速度，高精度，保留格式 | 对复杂表格支持一般 |
| **GROBID** | 元数据提取 | 结构化输出（DOI/作者/引用） | 需要 Java 环境 |
| **Nougat** | 科学文档备选 | LaTeX 输出，公式准确 | 速度慢，需 GPU |

**推荐方案**：
- **常规论文**：Marker + GROBID 组合
- **公式密集**：Nougat（按需切换）
- **CAJ 格式**：先用 `cajviewer` 或 `caj2pdf-wasm` 转 PDF

```python
from marker.convert import convert_single_pdf
from marker.models import load_all_models

# Marker 解析
models = load_all_models()
full_text, images, metadata = convert_single_pdf(pdf_path, models)

# GROBID 元数据补充
from grobid_client import GrobidClient
client = GrobidClient(config_path="./config.json")
metadata = client.process_pdf("processHeaderDocument", pdf_path)
```

### 3.2 双语 Embedding 模型

**选型**：**BAAI/bge-m3** (FlagEmbedding)

理由：
- 支持中英文在同一向量空间（无需翻译）
- 8192 token 上下文（覆盖整篇论文摘要）
- 同一模型可输出 dense、sparse、ColBERT 三种表示
- sparse lexical weights 替代默认 BM25 索引，降低索引维护成本

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
output = model.encode(
    ["行人流基本图关系研究", "fundamental diagram of pedestrian flow"],
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False,
)

dense_vectors = output["dense_vecs"]
sparse_weights = output["lexical_weights"]
```

### 3.3 向量数据库选择

| 数据库 | 适用场景 | 优势 | 劣势 |
|--------|---------|------|------|
| **Chroma** | 开发/小规模 (<10万篇) | 零配置，本地持久化 | 性能瓶颈明显 |
| **Milvus** | 生产/大规模 (>100万篇) | 分布式，高性能，混合搜索 | 部署复杂 |
| **Qdrant** | 中等规模 (10-100万篇) | 易部署，过滤查询强 | 社区规模小 |

**分阶段策略**：
- Phase 1-3: Chroma (快速验证)
- Phase 4+: 切换到 Milvus (配置 Docker Compose)

### 3.4 文档切分策略

**分层切分 (Hierarchical Chunking)**：

```
论文
 ├─ Title + Abstract (作为 parent chunk，始终返回)
 ├─ Section 1
 │   ├─ Paragraph 1.1 (child chunk)
 │   └─ Paragraph 1.2 (child chunk)
 ├─ Section 2
 │   └─ ...
```

实现：LangChain 的 `ParentDocumentRetriever`

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Parent splitter (粗粒度 - 按 section)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n# ", "\n## ", "\n### "]  # Markdown header 分割
)

# Child splitter (细粒度 - 按 paragraph)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=InMemoryStore(),
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
```

**优势**：检索时匹配细粒度 chunk（高精度），返回时带上 parent context（避免上下文缺失）。

---

## 四、高级检索技术

### 4.1 Query Transformation

#### HyDE (Hypothetical Document Embeddings)

```python
from langchain.chains import HypotheticalDocumentEmbedder

# 让 LLM 生成假想的答案文档，再用这个文档去检索
hyde_prompt = """给定这个问题，请生成一段假设的学术论文摘要来回答它。

问题：{question}

假设摘要："""

hyde_chain = LLMChain(llm=llm, prompt=hyde_prompt)
hyde_embeddings = HypotheticalDocumentEmbedder(
    llm_chain=hyde_chain,
    base_embeddings=base_embeddings
)
```

#### Multi-Query

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

# 自动生成多角度子查询
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm,
    prompt="""你是一个学术研究助手。针对用户问题，生成 3 个不同角度的搜索查询，
    覆盖：定义、方法论、应用案例。

    原始问题：{question}
    """
)
```

### 4.2 Reranking

使用 Cross-encoder 对初排结果重排序：

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=model, top_n=5)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)
```

**性能提升**：在 BEIR 基准上，加 reranker 可提升 10-15% NDCG@10。

### 4.3 Graph RAG (Citation Network)

使用 Neo4j 存储论文引用关系：

```
(Paper_A)-[:CITES]->(Paper_B)
(Paper_A)-[:AUTHORED_BY]->(Author_X)
(Paper_A)-[:PUBLISHED_IN]->(Venue_Y)
```

检索时结合向量相似度 + 图遍历：

```cypher
// 找到相似论文及其引用链
MATCH (p:Paper)
WHERE p.embedding_similarity > 0.8
OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
RETURN p, collect(cited) as citations
```

实现：`langchain-neo4j` + `Neo4jGraph`

---

## 五、中文学术数据库集成

### 5.1 CNKI (知网)

**挑战**：无公开 API，需爬虫 + CAJ 格式处理

**方案**：
1. 使用 `MagicCNKI` 项目 (GitHub) 作为爬虫基础
2. CAJ 转 PDF：`caj2pdf-wasm` (WebAssembly 版本，无需安装 CAJViewer)
3. 元数据提取：从网页 HTML 解析（标题/作者/摘要/关键词）

```python
# 伪代码
from magic_cnki import CNKIClient

client = CNKIClient(session_cookie=cnki_cookie)
results = client.search("行人流 基本图", max_results=20)

for paper in results:
    # 下载 CAJ
    caj_path = client.download(paper.id)
    # 转 PDF
    pdf_path = convert_caj_to_pdf(caj_path)
    # 索引
    index_paper(pdf_path, metadata=paper.metadata)
```

### 5.2 Wanfang (万方)

类似 CNKI，需爬虫。API 端点：`http://d.wanfangdata.com.cn/`

### 5.3 合规性建议

- 爬取频率限制（1 req/sec）
- 仅索引已订阅/授权的文献
- 添加 robots.txt 遵守声明

---

## 六、开源 RAG 项目架构分析

### 6.1 主流框架对比

| 项目 | Stars | 核心定位 | 适用场景 |
|------|-------|---------|---------|
| **LangChain** | 最高 | 通用 LLM 编排框架 | 多步工作流、Agent + RAG |
| **LlamaIndex** | 高 | RAG 专用全流程工具链 | RAG-first 应用 |
| **RAGFlow** | 高 | 深度文档理解引擎 | 表格/图表/层级结构解析 |
| **Quivr** | 高 | 个人 AI 知识库 | 个人知识管理 + RAG |
| **AutoRAG** | 中 | AutoML 风格 RAG 优化 | 管道自动调参与评估 |

### 6.2 LangChain vs LlamaIndex 选择

**LangChain** (本项目选择):
- 三层架构：编排层 (LangChain) + 向量存储 + 推理层
- 优势：灵活的多步编排，与 LangGraph/LangSmith 无缝集成
- 适合：需要 Agent + RAG + 多工具协作的复杂系统

**LlamaIndex**:
- 三阶段优化：预检索（sentence window）→ 检索（hybrid search）→ 后检索（reranking + compression）
- 优势：RAG 组件最全面，开箱即用
- 适合：纯 RAG 应用，无复杂 Agent 编排需求

**决策**：本项目选 LangChain，因为核心是 Agent（需要工具调用、状态管理、多路路由），RAG 是 Agent 的一个能力而非全部。

### 6.3 RAGFlow 借鉴点

RAGFlow 的深度文档理解值得借鉴：
- 表格结构识别（学术论文中的数据表）
- 图表标题解析（Figure caption → 可检索文本）
- 标题层级保留（Section → Subsection → Paragraph）

可在文档解析层参考其实现，但不直接依赖（避免引入过重框架）。

---

## 七、向量数据库性能评测

### 7.1 基准测试 (1M 向量, 768 维)

| 数据库 | QPS | P99 延迟 | 内存 | 推荐场景 |
|--------|-----|---------|------|---------|
| **Qdrant** | 1200-2100 | 12ms | 3.8GB | 生产级（中等规模） |
| **Milvus** | 400-500 (baseline) | - | 5GB/节点 | 亿级规模 + K8s |
| **Chroma** | 372-650 | 38ms (500ms+ at scale) | 1.2GB | 仅原型开发 |

### 7.2 修订后的分阶段策略

- **Phase 1-2 (验证)**：Chroma（零配置快速启动）
- **Phase 3+ (小规模生产)**：**Qdrant**（3.2x 快于 Chroma，Docker 部署简单）
- **规模化需求**：Milvus（需 K8s）

> 关键调整：原方案的 Chroma → Milvus 路径中间应插入 Qdrant，适合 10-100 万文档量级。

---

## 八、Embedding 模型最终选型

### BGE-M3 vs mGTE 对比

| 模型 | 中英双语 | 上下文长度 | 检索模式 | 优势 |
|------|---------|-----------|---------|------|
| BAAI/bge-m3 | 100+ 语言 | 8192 tokens | dense + sparse + ColBERT | 多功能融合检索 |
| Alibaba/mGTE-large | 多语言 | 8192 tokens | dense only | 较新，部分基准更优 |

**最终选择：BGE-M3**

理由：
1. 支持 dense + sparse + ColBERT 三种检索模式（单模型替代多模型）
2. Sparse 输出替代默认独立 BM25 索引（无需额外维护关键词索引）
3. 有 RAG 对话式检索变体 (`HIT-TMG/bge-m3_RAG-conversational-IR`)
4. 社区生态最完善，问题排查资源多

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# 同时获得 dense + sparse + colbert 向量
output = model.encode(
    ["行人流社会力模型参数标定方法"],
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True
)

dense_embedding = output['dense_vecs']     # 用于向量检索
sparse_embedding = output['lexical_weights']  # 用于稀疏检索
colbert_vecs = output['colbert_vecs']      # 用于精细匹配
```

---

## 九、高级 RAG 关键洞察

> **"多数 RAG 失败可追溯到弱检索或差的文档摄入，而非 LLM 质量。"** — LlamaIndex Blog

### 9.1 检索失败的主因

1. **Vanilla RAG 40% 失败率** — 纯向量检索在精确匹配上不足
2. **Chunking 不当** — 语义切分可提升 70% 准确率 (vs naive fixed-size)
3. **无 Reranking** — 加 reranker 可提升 10-15% NDCG@10

### 9.2 修订后的检索管道

```
BGE-M3 Sparse (替代独立 BM25) ─┐
                                 ├─ RRF 融合 → Reranker → LLM
BGE-M3 Dense (语义检索) ────────┘
                                 │
Neo4j Graph (引用链) ────────────┘
```

优势：用 BGE-M3 单模型同时产出 dense + sparse 向量，无需维护单独的 BM25 索引。

### 9.3 Memory 层建议

集成 **Mem0** 作为长期记忆层：
- 从对话中自动提取关键信息
- 图结构存储，支持关系推理
- 跨会话知识积累（用户的研究方向、已读论文、常用方法论）

---

## 十、实施建议

### 10.1 最小可行方案 (MVP)

```
Marker 解析 PDF → RecursiveCharacterTextSplitter 切分 
→ BGE-M3 dense+sparse 向量化 → Chroma 存储 
→ Hybrid Search (dense + sparse) 
→ RRF 融合 → LLM 生成带引用答案
```

### 10.2 生产级增强

```
MVP + GROBID 元数据 + Neo4j Citation Graph 
+ Reranker + Query Transformation 
+ Qdrant 替换 Chroma；规模化后再评估 Milvus
+ LangSmith 全链路追踪
```

### 10.3 领域适配

- 收集行人流领域论文 50-100 篇作为初始语料库
- 微调或替换 Embedding 模型（可选，如果 BGE-M3 表现不佳）
- 构建领域术语词典，用于查询扩展和稀疏检索权重增强

---

## 十一、关键文件映射

基于此方案，更新项目结构：

```
src/ped_agent/knowledge/
├── parsers/
│   ├── marker_parser.py      # Marker PDF 解析
│   ├── grobid_parser.py      # GROBID 元数据提取
│   └── caj_converter.py      # CAJ → PDF
├── embeddings.py             # BGE-M3 wrapper
├── retriever.py              # HybridRetriever (BGE-M3 dense+sparse + graph)
├── reranker.py               # Cross-encoder reranking
├── query_transform.py        # HyDE, Multi-Query, Step-Back
├── graph_store.py            # Neo4j citation graph
└── sources/
    ├── cnki.py               # CNKI 爬虫
    └── wanfang.py            # 万方爬虫
```

---

## 十二、Agent 框架选择与架构

### 12.1 框架对比

| 框架 | 架构模式 | 成熟度 | 适用场景 |
|------|---------|--------|---------|
| **LangGraph 1.x** | 有向状态图 | 生产级 | 复杂工作流 + 分支路由 + 持久化 |
| **CrewAI** | 角色扮演 + 顺序任务 | 快速原型 | 多角色协作、内容生成 |
| **AutoGen** (Microsoft) | 群聊会话 | 维护模式 | 研究/Azure 场景 |

**本项目选择：LangGraph**

理由：
1. 状态控制精确 — 多步检索→验证→分析需要显式分支
2. 检查点机制 — `MemorySaver` (开发) / `PostgresSaver` (生产)，支持断点续跑
3. 人在回路 — `interrupt()` 机制，关键分析结论需人工确认
4. LangSmith 原生集成 — 每个 node 执行自动追踪
5. 流式输出 — 长分析过程可实时返回中间结果

### 12.2 Agent 架构模式选择

| 模式 | 适用条件 | 准确率 |
|------|---------|--------|
| Self-Reflective ReAct | < 10 tools，单一领域 | 94% (vs 71% 无反思) |
| **Hierarchical Teams** | > 10 tools，多阶段工作 | 推荐本项目 |
| Supervisor-Worker | 需审批/审计链 | 高安全场景 |
| Multi-Agent Debate | 高风险模糊决策 | 学术争议评估 |

### 12.3 本项目 Agent 架构

```
┌─────────────────────────────────────────────────────────┐
│  Supervisor Node (LangGraph StateGraph)                  │
│  ├── classify_query → route                             │
│  │                                                       │
│  ├── Literature Agent (subgraph)                         │
│  │   Tools: semantic_scholar, arxiv, cnki, vector_store │
│  │   职责: 文献检索、引用链追踪、知识问答                    │
│  │                                                       │
│  ├── Analysis Agent (subgraph)                           │
│  │   Tools: compute_metrics, od_matrix, visualizer      │
│  │   职责: 数据分析、指标计算、图表生成                     │
│  │                                                       │
│  ├── Evaluation Agent (subgraph)                         │
│  │   Tools: evaluate_plan, retrieve_knowledge           │
│  │   职责: 实验方案评估、推荐建议                          │
│  │                                                       │
│  └── Synthesis Node                                      │
│      职责: 综合多 agent 结果、格式化引用、生成最终回答       │
└─────────────────────────────────────────────────────────┘
```

**渐进策略**：先实现单 Agent（Self-Reflective ReAct），当观测到上下文污染（文献检索内容干扰数据分析准确性）时，再拆分为多 Agent。

---

## 十三、MCP (Model Context Protocol) 集成

### 13.1 MCP 核心概念

MCP 是 LLM 连接外部工具和数据的开放标准，适合把学术检索、文件系统、实验数据库等外部能力以统一协议暴露给 Agent：

- **传输层**：JSON-RPC 2.0 (stdio / Streamable HTTP)
- **三种原语**：
  - **Tools** — 可调用函数（搜索、计算、文件操作）
  - **Resources** — 可读数据（论文全文、数据集）
  - **Prompts** — 可复用模板（分析模板、评估标准）
- **动态发现**：Agent 运行时查询可用工具，避免 prompt 膨胀

### 13.2 本项目 MCP Server 规划

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "./data/papers"]
    },
    "semantic-scholar": {
      "command": "python",
      "args": ["-m", "ped_agent.mcp.semantic_scholar_server"]
    },
    "experiment-db": {
      "command": "python",
      "args": ["-m", "ped_agent.mcp.database_server"],
      "env": {"DATABASE_URL": "${oc.env:DATABASE_URL}"}
    },
    "arxiv": {
      "command": "python",
      "args": ["-m", "ped_agent.mcp.arxiv_server"]
    }
  }
}
```

### 13.3 推荐的 MCP 服务器

| MCP Server | 功能 | 用途 |
|-----------|------|------|
| Filesystem | 读写本地文件 | 访问下载的 PDF、数据集 |
| AIRA-SemanticScholar | Semantic Scholar API | 英文文献检索 |
| Database (自建) | SQL 查询 | 实验数据、场景元数据 |
| arXiv (自建) | arXiv API | 预印本检索 |
| Web Search | 网络搜索 | 补充检索 |
| Python REPL (自建) | 代码执行 | 数据分析计算 |

### 13.4 MCP 与 LangChain Tool 的关系

```python
# MCP tools 可通过 langchain-mcp-adapters 接入 LangChain/LangGraph
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(mcp_config["mcpServers"])
tools = await client.get_tools()

agent = create_agent(model=llm, tools=tools)
```

---

## 十四、Agent Memory 架构

### 14.1 分层记忆设计

| 层级 | 类型 | 存储 | 内容 |
|------|------|------|------|
| L0 | 短期记忆 | LangGraph State | 当前对话消息历史 |
| L1 | 会话记忆 | PostgresSaver | 检查点，支持断点续跑 |
| L2 | 长期记忆 | 向量库 (Qdrant) | 已索引论文、分析结果 |
| L3 | 关系记忆 | Neo4j | 论文引用图、实验关联 |
| L4 | 用户偏好 | Mem0 | 研究方向、已读论文、常用方法 |

### 14.2 记忆选型建议

| 系统 | 架构 | 适用场景 |
|------|------|---------|
| **Mem0** | 向量 + 知识图谱 | 用户偏好、语义回忆、个性化 |
| **Zep/Graphiti** | 时间知识图谱 | 时间感知事实、多跳关系 |
| **LangGraph Checkpoint** | 内置状态持久化 | 会话级断点恢复 |

**推荐方案**：
- L0-L1：LangGraph 内置（零额外依赖）
- L2：Qdrant 向量库（论文检索已有）
- L3：Neo4j（引用图已规划）
- L4：初期跳过，中期引入 Mem0（用户研究偏好积累）

---

## 十五、LangSmith 评估体系

### 15.1 评估数据集结构

```python
# 评估数据集示例
evaluation_examples = [
    {
        "inputs": {"question": "社会力模型的基本参数有哪些？"},
        "outputs": {
            "answer": "社会力模型包含期望速度、松弛时间、行人间排斥力、边界排斥力等参数...",
            "citations": ["Helbing1995", "Helbing2000"]
        }
    },
    {
        "inputs": {"question": "评估这个疏散实验方案的合理性：..."},
        "outputs": {
            "evaluation_score": 7.5,
            "key_issues": ["缺少对照组", "样本量不足"]
        }
    }
]
```

### 15.2 评估指标

| 指标类型 | 指标 | 方法 |
|---------|------|------|
| 检索质量 | Retrieval Precision@K | 检索结果与标注相关文档的重叠 |
| 回答质量 | Answer Correctness | LLM-as-Judge (对比参考答案) |
| 引用准确 | Citation Accuracy | 自动验证引用是否可追溯到原文 |
| 任务完成 | Task Completion | Agent 是否完整执行工作流 |
| 轨迹评估 | Trajectory Quality | 工具调用序列是否合理 |
| 简洁性 | Concision | 回答是否精炼不冗余 |

### 15.3 评估流程

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# 1. 创建数据集
dataset = client.create_dataset("ped-agent-literature-qa")
client.create_examples(inputs=[...], outputs=[...], dataset_id=dataset.id)

# 2. 定义 evaluator
def citation_accuracy(run, example):
    """检查生成的引用是否在知识库中可找到"""
    predicted_citations = extract_citations(run.outputs["answer"])
    expected_citations = example.outputs["citations"]
    precision = len(set(predicted_citations) & set(expected_citations)) / len(predicted_citations)
    return {"score": precision, "key": "citation_accuracy"}

# 3. 运行评估
results = evaluate(
    agent_executor.invoke,
    data="ped-agent-literature-qa",
    evaluators=[citation_accuracy, "qa"],  # "qa" 是内置 LLM 评估器
    experiment_prefix="v1.0-bge-m3"
)
```

### 15.4 常见失败模式与防护

| 失败模式 | 表现 | 防护措施 |
|---------|------|---------|
| 幻觉 | 引用不存在的论文 | Citation Accuracy evaluator + 强制引用验证节点 |
| 工具误用 | 参数格式错误 | Pydantic Tool schema 验证 + LangSmith 轨迹检查 |
| 无限循环 | Agent 重复查询无进展 | LangGraph 最大迭代数限制 (recursion_limit) |
| 上下文污染 | 无关信息影响推理 | 拆分 subgraph / 上下文压缩 |

---

## 十六、多 Agent vs 单 Agent 决策框架

### 何时拆分为多 Agent

| 信号 | 触发条件 |
|------|---------|
| 上下文污染 | 文献检索内容导致数据分析准确率下降 >5% |
| 工具数超限 | 单 Agent 工具 >10 个，选择准确率下降 |
| 安全边界 | 不同数据访问权限需求 |
| 并行需求 | 检索和分析可同时进行 |

### 推荐的渐进路线

```
Phase 1-4: 单 Agent (Self-Reflective ReAct)
  → 在 LangSmith 中监测工具调用准确率和回答质量
  
Phase 5+: 评估是否需要拆分
  → 如果观测到上下文污染或工具误选 → 拆分为 Hierarchical Teams
  → 如果表现良好 → 保持单 Agent
```

---

## 十七、完整技术栈汇总

```
Agent 层:      LangGraph (状态机) + LangSmith (追踪/评估)
LLM 层:       langchain-anthropic / langchain-openai (Claude/GPT/DeepSeek/Qwen/GLM)
检索层:        BGE-M3 (dense+sparse) + Reranker (bge-reranker-v2-m3) + Neo4j (图)
存储层:        Qdrant (向量) + PostgreSQL (检查点) + Neo4j (引用图)
解析层:        Marker + GROBID (PDF) + caj2pdf-wasm (CAJ)
分析层:        NumPy + Pandas + SciPy + Plotly
视觉层:        ultralytics (YOLO26) + boxmot (ByteTrack/DeepSORT)
工具层:        MCP Servers (Filesystem/SemanticScholar/arXiv/Database)
记忆层:        LangGraph Checkpoint + Qdrant + Mem0 (渐进引入)
评估层:        LangSmith Evaluation + AutoRAG (管道优化)
配置层:        OmegaConf + .env
```

---

## 十八、与主计划的修订对照

| 原方案 | 修订后 |
|--------|--------|
| langchain-core (轻量) | langchain 完整生态 + LangGraph + LangSmith |
| bge-m3 仅 dense | BGE-M3 dense + sparse (替代独立 BM25) |
| rank-bm25 (独立) | BGE-M3 sparse 输出替代 |
| Chroma → Milvus | Chroma → **Qdrant** → Milvus |
| PyMuPDF | **Marker** + GROBID (更高精度) |
| 无图数据库 | + **Neo4j** (引用图) |
| 无 MCP | + **MCP Servers** (动态工具发现) |
| 无 Memory | + **分层记忆** (LangGraph + Qdrant + Mem0) |
| 单 Agent | **渐进式**: 单 Agent → Hierarchical Teams |
| 无系统评估 | + **LangSmith 评估数据集** + Citation Accuracy |

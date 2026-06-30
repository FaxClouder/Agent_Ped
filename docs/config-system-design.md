# 配置系统详细设计

## 一、设计原则

- **分层配置**：默认值 → 环境配置 → 环境变量 → CLI 参数
- **模块隔离**：每个模块独立配置文件，按需加载
- **安全**：敏感信息（API Key）通过环境变量注入，不落盘
- **热更新**：非核心配置支持运行时修改（如切换 LLM provider）

---

## 二、配置文件结构

```
config/
├── default.yaml          # 全局默认配置
├── llm.yaml              # LLM 后端配置
├── rag.yaml              # RAG 管道配置
├── analysis.yaml         # 数据分析配置
├── vision.yaml           # 视觉模块配置
├── langsmith.yaml        # 追踪与评估配置
└── .env.example          # 环境变量模板
```

---

## 三、配置文件详细定义

### 3.1 default.yaml

```yaml
app:
  name: "Ped-Agent"
  version: "0.1.0"
  language: "zh"  # 默认回答语言: zh | en
  log_level: "INFO"  # DEBUG | INFO | WARNING | ERROR
  data_dir: "./data"
  output_dir: "./outputs"

agent:
  mode: "single"  # "single" | "multi" (渐进式，初期 single)
  recursion_limit: 25  # LangGraph 最大递归深度
  max_tool_calls: 10  # 单次对话最大工具调用次数
  streaming: true  # 启用流式输出
  checkpoint:
    backend: "memory"  # "memory" | "sqlite" | "postgres"
    connection_string: null  # postgres 时使用
```

### 3.2 llm.yaml

```yaml
llm:
  default_provider: "claude"
  fallback_provider: "deepseek"  # 主模型失败时降级
  
  providers:
    claude:
      model: "claude-sonnet-4-20250514"
      api_key: "${oc.env:ANTHROPIC_API_KEY,null}"
      max_tokens: 4096
      temperature: 0.1
      timeout: 60
      max_retries: 3
      
    openai:
      model: "gpt-4o"
      api_key: "${oc.env:OPENAI_API_KEY,null}"
      max_tokens: 4096
      temperature: 0.1
      timeout: 60
      
    deepseek:
      model: "deepseek-chat"
      api_key: "${oc.env:DEEPSEEK_API_KEY,null}"
      base_url: "https://api.deepseek.com/v1"
      max_tokens: 4096
      temperature: 0.1
      
    qwen:
      model: "qwen-plus"
      api_key: "${oc.env:DASHSCOPE_API_KEY,null}"
      base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
      max_tokens: 4096
      
    glm:
      model: "glm-4-plus"
      api_key: "${oc.env:ZHIPU_API_KEY,null}"
      base_url: "https://open.bigmodel.cn/api/paas/v4"
      max_tokens: 4096
      
    local:
      model: "qwen2.5:72b"
      base_url: "http://localhost:11434"
      max_tokens: 4096
      temperature: 0.1

  # 用途分配：不同任务可使用不同模型
  routing:
    classification: "default"  # 查询分类用默认模型
    generation: "default"      # 回答生成
    evaluation: "default"      # 实验评估
    embedding_query_expansion: "deepseek"  # 查询扩展（便宜模型即可）
```

### 3.3 rag.yaml

```yaml
rag:
  embedding:
    model: "BAAI/bge-m3"
    device: "cuda"  # "cuda" | "cpu" | "mps"
    batch_size: 32
    normalize: true
    modes:
      dense: true
      sparse: true  # 替代 BM25
      colbert: false  # 按需开启

  vector_store:
    backend: "chroma"  # "chroma" | "qdrant" | "milvus"
    
    chroma:
      persist_directory: "${oc.env:APP_DATA_DIR,./data}/vectordb/chroma"
      collection_name: "pedestrian_literature"
      
    qdrant:
      url: "http://localhost:6333"
      collection_name: "pedestrian_literature"
      api_key: "${oc.env:QDRANT_API_KEY,null}"
      
    milvus:
      uri: "http://localhost:19530"
      collection_name: "pedestrian_literature"

  retrieval:
    top_k: 20  # 初始检索数量
    rerank_top_k: 5  # 重排序后保留数量
    hybrid_weight: 0.7  # dense 权重 (sparse = 1 - weight)
    reranker:
      enabled: true
      model: "BAAI/bge-reranker-v2-m3"
      device: "cuda"
      
  query_transform:
    enabled: true
    strategies: ["multi_query"]  # "hyde" | "multi_query" | "step_back"
    multi_query_count: 3
    
  chunking:
    strategy: "hierarchical"  # "fixed" | "semantic" | "hierarchical"
    parent_chunk_size: 2000
    parent_chunk_overlap: 200
    child_chunk_size: 400
    child_chunk_overlap: 50
    separators: ["\n# ", "\n## ", "\n### ", "\n\n", "\n"]
    
  parsing:
    pdf_parser: "marker"  # "marker" | "pymupdf" | "nougat"
    metadata_extractor: "grobid"  # "grobid" | "none"
    grobid_url: "http://localhost:8070"
    
  sources:
    semantic_scholar:
      enabled: true
      api_key: "${oc.env:S2_API_KEY,null}"
      max_results: 20
    arxiv:
      enabled: true
      max_results: 20
    cnki:
      enabled: false  # 需配置 credentials
      session_cookie: "${oc.env:CNKI_COOKIE,null}"
    wanfang:
      enabled: false
      
  graph:
    enabled: false  # Phase 3+ 开启
    neo4j_uri: "bolt://localhost:7687"
    neo4j_user: "neo4j"
    neo4j_password: "${oc.env:NEO4J_PASSWORD,null}"
```

### 3.4 analysis.yaml

```yaml
analysis:
  default_metrics: ["density", "velocity", "flow_rate", "los"]
  
  density:
    method: "voronoi"  # "voronoi" | "grid"
    grid_size: [50, 50]  # 网格方法分辨率
    
  velocity:
    smoothing: true
    window_size: 5  # 滑动平均窗口 (帧)
    
  flow:
    time_bin: 1.0  # 时间分箱 (秒)
    
  fundamental_diagram:
    enabled: true
    models: ["linear", "weidmann"]  # 拟合模型
    time_bin: 1.0
    
  od_matrix:
    enabled: true
    zone_definition_file: null  # JSON 区域定义文件
    
  visualization:
    backend: "plotly"  # "plotly" | "matplotlib"
    output_dir: "${oc.env:APP_OUTPUT_DIR,./outputs}/figures"
    theme: "plotly_white"
    dpi: 150  # matplotlib 输出分辨率
    interactive: true  # 生成交互式 HTML
    
  statistics:
    speed_distribution_fit: ["normal", "lognormal"]
    confidence_level: 0.95
```

### 3.5 vision.yaml

```yaml
vision:
  enabled: false  # 可选模块，按需启用
  
  detector:
    model: "yolo26x.pt"  # YOLO26 模型权重
    confidence: 0.5
    iou_threshold: 0.45
    classes: [0]  # COCO person class
    input_size: 1280
    device: "cuda:0"
    half_precision: true  # FP16 推理
    end2end: true  # YOLO26 默认 end-to-end / NMS-free 推理
    
  tracker:
    algorithm: "bytetrack"  # "bytetrack" | "deepsort"
    track_buffer: 30
    match_thresh: 0.8
    # DeepSORT 专用
    reid_model: null  # ReID 模型路径 (仅 deepsort)
    max_cosine_distance: 0.3
    
  coordinate_transform:
    method: "homography"  # "homography" | "scale" | "none"
    calibration_file: null  # 单应矩阵文件
    pixel_per_meter: null  # 简易缩放 (无标定时)
    
  preprocessing:
    resize: null  # [width, height] 或 null
    roi_file: null  # ROI 区域定义文件
    skip_frames: 1  # 每 N 帧检测一次 (1=逐帧)
    
  postprocessing:
    min_track_length: 10  # 最小轨迹长度 (帧)
    smoothing: "savgol"  # "none" | "moving_avg" | "savgol"
    smoothing_window: 5
    interpolation: true  # 插值缺失帧
    
  output:
    save_video: false
    video_codec: "mp4v"
    trajectory_format: "json"  # "json" | "csv"
    output_dir: "${oc.env:APP_OUTPUT_DIR,./outputs}/trajectories"
```

### 3.6 langsmith.yaml

```yaml
langsmith:
  enabled: true
  api_key: "${oc.env:LANGSMITH_API_KEY,null}"
  project_name: "ped-agent"
  
  tracing:
    enabled: true
    sample_rate: 1.0  # 采样率 (开发阶段全量)
    
  evaluation:
    datasets:
      literature_qa: "ped-agent-literature-qa"
      experiment_eval: "ped-agent-experiment-eval"
    auto_eval_on_deploy: false
    
  prompts:
    hub_owner: null  # LangSmith Hub 用户名
    version_prefix: "v"
```

---

## 四、环境变量模板

### .env.example

```bash
# === LLM API Keys ===
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
DASHSCOPE_API_KEY=sk-...
ZHIPU_API_KEY=...

# === RAG ===
S2_API_KEY=...           # Semantic Scholar API Key
QDRANT_API_KEY=...       # Qdrant Cloud (可选)
NEO4J_PASSWORD=...       # Neo4j (可选)

# === CNKI/Wanfang ===
CNKI_COOKIE=...          # 知网 session cookie

# === LangSmith ===
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=ped-agent

# === Paths ===
APP_DATA_DIR=./data
APP_OUTPUT_DIR=./outputs
```

---

## 五、配置加载实现

```python
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import Optional
import os

class ConfigManager:
    """配置管理器"""
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[DictConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self, config_dir: str = "./config", 
             env_file: str = ".env",
             overrides: list = None) -> DictConfig:
        """
        加载配置，优先级: CLI overrides > 环境变量 > 配置文件 > 默认值
        """
        # 加载 .env
        self._load_dotenv(env_file)
        
        # 加载各模块配置文件
        config_path = Path(config_dir)
        configs = []
        
        for yaml_file in ['default.yaml', 'llm.yaml', 'rag.yaml', 
                          'analysis.yaml', 'vision.yaml', 'langsmith.yaml']:
            filepath = config_path / yaml_file
            if filepath.exists():
                configs.append(OmegaConf.load(filepath))
        
        # 合并配置文件
        merged = OmegaConf.merge(*configs)

        # CLI overrides (如 --llm.default_provider=deepseek)
        if overrides:
            cli_conf = OmegaConf.from_dotlist(overrides)
            merged = OmegaConf.merge(merged, cli_conf)

        # 解析环境变量插值 (${oc.env:VAR_NAME,default})
        OmegaConf.resolve(merged)
        
        self._config = merged
        return merged
    
    @property
    def config(self) -> DictConfig:
        if self._config is None:
            self.load()
        return self._config
    
    def get(self, key: str, default=None):
        """点分路径访问: config.get('llm.default_provider')"""
        try:
            return OmegaConf.select(self.config, key)
        except:
            return default
    
    def _load_dotenv(self, env_file: str):
        """加载 .env 文件到环境变量"""
        env_path = Path(env_file)
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ.setdefault(key.strip(), value.strip())

# 全局访问
def get_config() -> DictConfig:
    return ConfigManager().config
```

---

## 六、配置验证

```python
from pydantic import BaseModel, field_validator
from typing import Literal

class LLMProviderConfig(BaseModel):
    model: str
    api_key: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.1
    base_url: str | None = None
    timeout: int = 60
    max_retries: int = 3

    @field_validator('api_key')
    @classmethod
    def api_key_resolved(cls, v):
        if isinstance(v, str) and v.startswith('${'):
            raise ValueError("API key not resolved from environment")
        return v

class AppConfig(BaseModel):
    """顶层配置验证 schema"""
    
    class App(BaseModel):
        name: str
        language: Literal["zh", "en"]
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
        
    class Agent(BaseModel):
        mode: Literal["single", "multi"]
        recursion_limit: int
        streaming: bool
    
    app: App
    agent: Agent

def validate_config(config: DictConfig) -> bool:
    """启动时验证配置完整性"""
    errors = []
    
    # 检查默认/降级 LLM provider 的 API key；local provider 不需要 key
    provider = OmegaConf.select(config, 'llm.default_provider')
    fallback = OmegaConf.select(config, 'llm.fallback_provider')
    for p in {provider, fallback} - {None, "local"}:
        api_key = OmegaConf.select(config, f'llm.providers.{p}.api_key')
        if not api_key:
            errors.append(f"LLM provider '{p}' API key not configured")
    
    # 检查 LangSmith
    if OmegaConf.select(config, 'langsmith.enabled'):
        ls_key = OmegaConf.select(config, 'langsmith.api_key')
        if not ls_key:
            errors.append("LangSmith enabled but LANGSMITH_API_KEY not set")
    
    # 检查视觉模块依赖
    if OmegaConf.select(config, 'vision.enabled'):
        try:
            import ultralytics
        except ImportError:
            errors.append("Vision enabled but ultralytics not installed (pip install ped-agent[vision])")
    
    if errors:
        for e in errors:
            print(f"[CONFIG ERROR] {e}")
        return False
    return True
```

---

## 七、使用方式

### 7.1 应用入口

```python
# src/ped_agent/main.py
from ped_agent.utils.config import ConfigManager, validate_config

def main():
    config_manager = ConfigManager()
    config = config_manager.load(
        config_dir="./config",
        env_file=".env",
        overrides=sys.argv[1:]  # CLI: --llm.default_provider=deepseek
    )
    
    if not validate_config(config):
        sys.exit(1)
    
    # 初始化各模块
    ...
```

### 7.2 模块内使用

```python
# 在任意模块中访问配置
from ped_agent.utils.config import get_config

config = get_config()
embedding_model = config.rag.embedding.model  # "BAAI/bge-m3"
llm_provider = config.llm.default_provider    # "claude"
```

### 7.3 运行时切换

```python
# CLI 切换模型
python -m ped_agent --llm.default_provider=deepseek

# 环境变量覆盖
ANTHROPIC_API_KEY=new_key python -m ped_agent
```

---

## 八、模块文件结构

```
src/ped_agent/utils/
├── __init__.py
├── config.py             # ConfigManager 主类
├── validation.py         # 配置验证
└── logging.py            # 日志配置 (基于 config.app.log_level)

config/
├── default.yaml
├── llm.yaml
├── rag.yaml
├── analysis.yaml
├── vision.yaml
├── langsmith.yaml
└── .env.example
```

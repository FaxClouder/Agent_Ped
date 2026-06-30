# 实验评估模块详细设计

## 一、模块概述

实验评估模块负责对行人流研究的实验方案进行系统化评估，包括：
- 实验方案可行性评估
- 方法论完整性检查
- 文献支撑与理论依据验证
- 改进建议生成

---

## 二、评估维度体系

### 2.1 五维评估框架

| 维度 | 权重 | 评估内容 |
|------|------|---------|
| **可行性** (Feasibility) | 0.25 | 资源可获得性、技术实现难度、时间成本 |
| **完整性** (Completeness) | 0.25 | 实验要素是否齐备（对照组、样本量、变量控制） |
| **方法论** (Methodology) | 0.25 | 数据采集方法、分析方法是否科学合理 |
| **创新性** (Innovation) | 0.15 | 相对现有研究的增量贡献 |
| **可重复性** (Reproducibility) | 0.10 | 方法描述是否清晰可复现 |

### 2.2 细化评分标准

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class ScoreLevel(str, Enum):
    EXCELLENT = "excellent"      # 9-10
    GOOD = "good"               # 7-8
    ADEQUATE = "adequate"       # 5-6
    INSUFFICIENT = "insufficient"  # 3-4
    POOR = "poor"               # 1-2

class DimensionCriteria(BaseModel):
    """单维度评估标准"""
    dimension: str
    weight: float
    checklist: List[str]  # 检查项列表
    
EVALUATION_CRITERIA = [
    DimensionCriteria(
        dimension="feasibility",
        weight=0.25,
        checklist=[
            "实验场地是否可获得并满足条件",
            "所需设备（摄像头、传感器）是否可用",
            "参与者招募是否现实（人数、年龄分布）",
            "实验时间规划是否合理",
            "预算是否充足",
            "伦理审查是否通过或可通过",
        ]
    ),
    DimensionCriteria(
        dimension="completeness",
        weight=0.25,
        checklist=[
            "是否设置了对照组/基准条件",
            "自变量和因变量是否明确定义",
            "样本量是否满足统计功效要求",
            "是否考虑了混杂变量的控制",
            "数据采集方案是否完整",
            "数据分析方法是否预先确定",
            "是否有预实验/pilot study 计划",
        ]
    ),
    DimensionCriteria(
        dimension="methodology",
        weight=0.25,
        checklist=[
            "场景设计是否贴近真实行人流条件",
            "密度/速度测量方法是否标准化",
            "视频分析方案是否可靠（帧率、分辨率、视角）",
            "轨迹提取精度是否满足分析需求",
            "统计分析方法是否恰当",
            "是否考虑了测量误差和不确定性",
        ]
    ),
    DimensionCriteria(
        dimension="innovation",
        weight=0.15,
        checklist=[
            "研究问题是否有新意",
            "是否提出了新的实验范式或方法",
            "是否填补了现有研究的空白",
            "结果是否有潜在的理论或应用价值",
        ]
    ),
    DimensionCriteria(
        dimension="reproducibility",
        weight=0.10,
        checklist=[
            "实验步骤描述是否足够详细",
            "参数设置是否明确",
            "数据格式和标注规范是否清晰",
            "是否计划公开数据/代码",
        ]
    ),
]
```

---

## 三、评估流程

### 3.1 评估管道

```
实验方案文本输入
    │
    ▼
┌──────────────────────────────┐
│  1. 方案结构化解析            │
│     提取关键要素              │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  2. 文献支撑检索              │
│     查找相关方法论参考         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  3. 多维度逐项评估            │
│     LLM 结构化评分            │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  4. 综合评分与建议生成         │
│     加权得分 + 改进建议        │
└──────────────────────────────┘
```

### 3.2 核心数据模型

```python
class ExperimentPlan(BaseModel):
    """实验方案结构化表示"""
    title: str
    research_question: str
    hypothesis: Optional[str] = None
    methodology: str
    scenario_description: str
    variables: dict = Field(default_factory=dict)  # {"independent": [...], "dependent": [...], "controlled": [...]}
    sample_size: Optional[int] = None
    equipment: List[str] = Field(default_factory=list)
    duration: Optional[str] = None
    data_collection: str = ""
    analysis_methods: List[str] = Field(default_factory=list)
    
class DimensionScore(BaseModel):
    """单维度评分"""
    dimension: str
    score: float  # 1-10
    level: ScoreLevel
    checklist_results: List[dict]  # [{"item": "...", "passed": bool, "comment": "..."}]
    strengths: List[str]
    weaknesses: List[str]
    
class EvaluationResult(BaseModel):
    """完整评估结果"""
    plan_title: str
    overall_score: float  # 加权总分
    overall_level: ScoreLevel
    dimension_scores: List[DimensionScore]
    key_issues: List[str]  # 主要问题
    recommendations: List[str]  # 改进建议
    literature_support: List[dict]  # 支撑文献 [{"claim": "...", "citation": "..."}]
    confidence: float  # 评估置信度 (0-1)
```

---

## 四、LLM 结构化评估

### 4.1 评估 Prompt 设计

```python
EVALUATION_SYSTEM_PROMPT = """你是一位行人流研究领域的资深评审专家，拥有丰富的实验设计和学术评审经验。

你需要对提交的实验方案进行系统化评估。评估基于以下五个维度：
1. 可行性 (25%) - 资源、技术、时间可行性
2. 完整性 (25%) - 实验设计要素是否齐备
3. 方法论 (25%) - 方法是否科学合理
4. 创新性 (15%) - 研究增量贡献
5. 可重复性 (10%) - 方法描述是否可复现

评分标准：
- 9-10 (优秀): 该维度表现卓越，无明显缺陷
- 7-8 (良好): 该维度表现良好，有小瑕疵
- 5-6 (合格): 基本满足要求，有明显不足
- 3-4 (不足): 存在重要缺陷，需要大幅修改
- 1-2 (差): 该维度严重不足，需要重新设计

请严格按照 JSON 格式输出评估结果。"""

EVALUATION_USER_PROMPT = """请评估以下行人流实验方案：

---
{plan_text}
---

相关文献参考：
{literature_context}

请对每个维度逐项评分，并给出具体的改进建议。"""
```

### 4.2 评估器实现

```python
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

class ExperimentEvaluator:
    """实验方案评估器"""
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.parser = PydanticOutputParser(pydantic_object=EvaluationResult)
    
    async def evaluate(self, plan_text: str) -> EvaluationResult:
        """执行完整评估流程"""
        
        # Step 1: 结构化解析方案
        plan = await self._parse_plan(plan_text)
        
        # Step 2: 检索相关文献支撑
        literature = await self._retrieve_methodology_literature(plan)
        
        # Step 3: LLM 多维度评估
        result = await self._llm_evaluate(plan_text, literature)
        
        # Step 4: 生成改进建议
        result.recommendations = await self._generate_recommendations(
            plan, result.dimension_scores, literature
        )
        
        return result
    
    async def _parse_plan(self, plan_text: str) -> ExperimentPlan:
        """用 LLM 将自由文本解析为结构化方案"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "从以下实验方案中提取结构化信息。"),
            ("user", "{plan_text}")
        ])
        
        chain = prompt | self.llm.with_structured_output(ExperimentPlan)
        return await chain.ainvoke({"plan_text": plan_text})
    
    async def _retrieve_methodology_literature(self, plan: ExperimentPlan) -> str:
        """检索方法论相关文献"""
        queries = [
            f"行人流实验 {plan.methodology}",
            f"pedestrian experiment methodology {plan.scenario_description}",
            f"行人流数据采集方法 最佳实践",
        ]
        
        docs = []
        for query in queries:
            results = await self.retriever.ainvoke(query)
            docs.extend(results[:3])
        
        return "\n\n".join([d.page_content for d in docs[:10]])
    
    async def _llm_evaluate(self, plan_text: str, literature: str) -> EvaluationResult:
        """LLM 执行评估"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", EVALUATION_SYSTEM_PROMPT),
            ("user", EVALUATION_USER_PROMPT)
        ])
        
        chain = prompt | self.llm.with_structured_output(EvaluationResult)
        
        result = await chain.ainvoke({
            "plan_text": plan_text,
            "literature_context": literature
        })
        
        # 计算加权总分
        total = sum(
            ds.score * next(c.weight for c in EVALUATION_CRITERIA if c.dimension == ds.dimension)
            for ds in result.dimension_scores
        )
        result.overall_score = round(total, 2)
        result.overall_level = self._score_to_level(total)
        
        return result
    
    async def _generate_recommendations(self, plan, scores, literature) -> List[str]:
        """基于评估结果生成具体改进建议"""
        weak_dimensions = [s for s in scores if s.score < 7]
        
        if not weak_dimensions:
            return ["方案整体质量良好，可考虑进一步细化数据分析计划。"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """基于评估结果，为实验方案的薄弱环节提供具体、可操作的改进建议。
            每条建议应包含：问题描述 + 改进方法 + 参考依据（如有）。"""),
            ("user", "薄弱维度：{weak_dims}\n\n原方案：{plan}\n\n文献参考：{literature}")
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "weak_dims": str([{"dim": s.dimension, "score": s.score, "issues": s.weaknesses} for s in weak_dimensions]),
            "plan": str(plan),
            "literature": literature
        })
        
        return response.content.split("\n")
    
    @staticmethod
    def _score_to_level(score: float) -> ScoreLevel:
        if score >= 9:
            return ScoreLevel.EXCELLENT
        elif score >= 7:
            return ScoreLevel.GOOD
        elif score >= 5:
            return ScoreLevel.ADEQUATE
        elif score >= 3:
            return ScoreLevel.INSUFFICIENT
        else:
            return ScoreLevel.POOR
```

---

## 五、评估报告模板

### 5.1 报告结构

```python
REPORT_TEMPLATE = """
# 实验方案评估报告

## 基本信息
- 方案名称：{title}
- 评估时间：{timestamp}
- 综合评分：{overall_score}/10 ({overall_level})

## 各维度评分

| 维度 | 得分 | 等级 | 权重 |
|------|------|------|------|
{dimension_table}

## 主要优势
{strengths}

## 主要问题
{issues}

## 改进建议
{recommendations}

## 文献支撑
{literature_references}

---
*评估置信度：{confidence}*
*本报告由 Ped-Agent 自动生成，建议结合专家意见使用。*
"""
```

### 5.2 报告生成器

```python
class ReportGenerator:
    """评估报告生成"""
    
    def generate_markdown(self, result: EvaluationResult) -> str:
        """生成 Markdown 格式报告"""
        dimension_table = "\n".join([
            f"| {s.dimension} | {s.score:.1f} | {s.level.value} | "
            f"{next(c.weight for c in EVALUATION_CRITERIA if c.dimension == s.dimension)} |"
            for s in result.dimension_scores
        ])
        
        strengths = "\n".join([f"- {s}" for ds in result.dimension_scores for s in ds.strengths])
        issues = "\n".join([f"- {i}" for i in result.key_issues])
        recommendations = "\n".join([f"{i+1}. {r}" for i, r in enumerate(result.recommendations)])
        
        return REPORT_TEMPLATE.format(
            title=result.plan_title,
            timestamp="auto",
            overall_score=result.overall_score,
            overall_level=result.overall_level.value,
            dimension_table=dimension_table,
            strengths=strengths,
            issues=issues,
            recommendations=recommendations,
            literature_references="\n".join([f"- {l['citation']}" for l in result.literature_support]),
            confidence=f"{result.confidence:.0%}"
        )
    
    def generate_json(self, result: EvaluationResult) -> str:
        """生成 JSON 格式报告 (供前端消费)"""
        return result.model_dump_json(indent=2)
```

---

## 六、评估质量保障

### 6.1 多轮评估 (Self-Consistency)

```python
async def evaluate_with_consistency(evaluator: ExperimentEvaluator, 
                                     plan_text: str,
                                     n_rounds: int = 3) -> EvaluationResult:
    """
    多轮评估取共识，提高评估稳定性
    
    当多轮评分标准差 > 1.5 时，触发人工复核
    """
    results = []
    for _ in range(n_rounds):
        result = await evaluator.evaluate(plan_text)
        results.append(result)
    
    # 取各维度中位数分数
    final_scores = {}
    for dim in ['feasibility', 'completeness', 'methodology', 'innovation', 'reproducibility']:
        scores = [
            next(s.score for s in r.dimension_scores if s.dimension == dim)
            for r in results
        ]
        final_scores[dim] = {
            'median': np.median(scores),
            'std': np.std(scores),
            'needs_review': np.std(scores) > 1.5
        }
    
    # 合并建议 (去重)
    all_recommendations = set()
    for r in results:
        all_recommendations.update(r.recommendations)
    
    # 构建最终结果
    final = results[0].model_copy()
    for ds in final.dimension_scores:
        ds.score = final_scores[ds.dimension]['median']
    
    final.confidence = 1.0 - np.mean([v['std'] for v in final_scores.values()]) / 10
    final.recommendations = list(all_recommendations)
    
    return final
```

### 6.2 LangSmith 评估集成

```python
# 评估器自身的质量评估
EVAL_DATASET_EXAMPLES = [
    {
        "inputs": {
            "plan_text": "我计划在地铁站观察行人流...(缺少对照组、样本量不明确)"
        },
        "outputs": {
            "expected_overall_range": [4, 6],
            "expected_issues": ["缺少对照组", "样本量"],
            "min_recommendations": 3
        }
    },
    # ... 更多标注数据
]

def eval_score_accuracy(run, example):
    """评估模型给出的分数是否在预期范围内"""
    predicted_score = run.outputs["overall_score"]
    expected_range = example.outputs["expected_overall_range"]
    return {"score": int(expected_range[0] <= predicted_score <= expected_range[1])}

def eval_issue_coverage(run, example):
    """评估是否覆盖了关键问题"""
    predicted_issues = " ".join(run.outputs.get("key_issues", []))
    expected_keywords = example.outputs["expected_issues"]
    coverage = sum(1 for kw in expected_keywords if kw in predicted_issues) / len(expected_keywords)
    return {"score": coverage}
```

---

## 七、LangChain Tool 封装

```python
from langchain.tools import tool

@tool
async def evaluate_experiment_plan(plan_text: str,
                                   evaluation_mode: str = "standard") -> str:
    """
    评估行人流实验方案的质量和可行性。
    
    Args:
        plan_text: 实验方案文本描述
        evaluation_mode: "quick" (单轮快速评估) | "standard" (3轮一致性评估)
    
    Returns:
        Markdown 格式的评估报告
    """
    evaluator = ExperimentEvaluator(llm=get_llm(), retriever=get_retriever())
    
    if evaluation_mode == "quick":
        result = await evaluator.evaluate(plan_text)
    else:
        result = await evaluate_with_consistency(evaluator, plan_text, n_rounds=3)
    
    report = ReportGenerator().generate_markdown(result)
    return report
```

---

## 八、推荐建议生成逻辑

### 8.1 建议类型

| 类型 | 触发条件 | 示例 |
|------|---------|------|
| **补充型** | 缺少必要要素 | "建议增加对照实验组..." |
| **改进型** | 方法存在缺陷 | "密度测量建议采用 Voronoi 方法替代网格法..." |
| **增强型** | 可提升空间 | "可考虑增加问卷调查收集主观感受数据..." |
| **参考型** | 有文献支撑 | "参考 Helbing (2005) 的实验设置..." |

### 8.2 建议优先级排序

```python
def prioritize_recommendations(recommendations: List[dict],
                               dimension_scores: List[DimensionScore]) -> List[dict]:
    """
    按紧迫程度排序建议：
    1. 致命缺陷 (得分 < 3 的维度) — 必须修改
    2. 重要缺陷 (得分 3-5) — 强烈建议修改
    3. 改进空间 (得分 5-7) — 建议改进
    4. 锦上添花 (得分 > 7) — 可选增强
    """
    priority_map = {
        ScoreLevel.POOR: 0,
        ScoreLevel.INSUFFICIENT: 1,
        ScoreLevel.ADEQUATE: 2,
        ScoreLevel.GOOD: 3,
        ScoreLevel.EXCELLENT: 4,
    }
    # recommendations: [{"text": "...", "level": ScoreLevel.ADEQUATE, "dimension": "..."}]
    # 按关联维度的得分排序 (得分低的建议排前面)
    return sorted(recommendations, key=lambda r: priority_map.get(r.get('level', ScoreLevel.ADEQUATE), 2))
```

---

## 九、模块文件结构

```
src/ped_agent/experiment/
├── __init__.py
├── evaluator.py          # ExperimentEvaluator 主类
├── criteria.py           # 评估标准定义 (EVALUATION_CRITERIA)
├── schemas.py            # Pydantic 数据模型
├── report.py             # ReportGenerator 报告生成
├── consistency.py        # 多轮一致性评估
└── prompts/
    ├── system.j2         # 系统提示模板
    ├── evaluation.j2     # 评估提示
    └── recommendation.j2 # 建议生成提示
```

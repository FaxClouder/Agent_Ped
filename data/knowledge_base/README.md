# 知识库目录说明

该目录会被 RAG 检索模块递归扫描，支持：

- `.md`
- `.txt`
- `.pdf`

推荐把资料按主题拆分：

```text
knowledge_base/
├─ papers/      # 疏散研究论文、综述、模型说明
├─ policies/    # 法规、标准、制度摘要或 PDF 原文
├─ scenarios/   # 典型场景、案例复盘、预案模板
└─ handbook/    # 自建领域手册、术语表、常识库
```

使用方式：

1. 直接把 PDF 放进对应目录
2. 运行 `python -m evac_agent.main --prepare-index`
3. 之后再运行问答命令

注意：

- PDF 会按页读取并参与检索
- 新增、删除、替换文件后，系统会自动检测并重建索引
- 对扫描版 PDF，建议先进行 OCR，否则可检索文本有限

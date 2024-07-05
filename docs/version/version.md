# 版本管理

## 开发计划

### ZLAI

> `zlai/llm`

优先级 [⭐]

- [ ] 增加天工API
- [ ] 增加MiniMax
- [ ] 增加ChatGPT
- [ ] 增加Azure OpenAI
- [ ] 增加Gimini
- [ ] 增加商汤大模型

> `zlai/agent`

优先级 [⭐⭐⭐]

- [ ] 完成期货、股票、基金全部tools
- [ ] 增加期货、股票、基金全部Agent Tools
- [ ] 增加测试、文档
- [ ] 增加对于多模态模型的支持，图片解读、文生图等
- [ ] 增加复杂任务的动态规划
- [ ] 增加Agent生成知识图谱
- [ ] 增加基金、股票、期货、期权等金融任务问答
  - [ ] 增加基金实时行情查询Agent
- [ ] 增加大模型风控相关内容，问答式分箱？
- [ ] 增加写代码然后执行问答
- [ ] 增加画图问答
- [ ] 增加其他Agent
- [ ] 增加React
- [ ] message prompt 的组织方式中的参数不能与 task completion 中的参数重名
- [ ] 对知识对话增加记忆机制，增加记忆机制在多个Agent之间的共享
  - [X] 完成`ChatAgent/Knowledge`

### DOC

- [ ] 增加对于`Agent`的文档

### Streamlit

- [ ] 增加对于`Agent`的Streamlit支持

## 已发版本

> [![Python package](https://img.shields.io/pypi/v/zlai)](https://pypi.org/project/zlai/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/zlai)](https://pypi.org/project/zlai/)

### 0.3.96

1. 增加百川、百度、豆包大模型

### 0.3.90

1. 清理了一些代码
2. 增加了[LLM-SiliconCloud](https://cloud.siliconflow.cn/)全系列大模型

### 0.3.83

1. 增加了Tools-call
2. 修复了一些内容，增加了几个测试。

### 0.3.76

1. 增加了对于GLM4-API的支持
2. 增加了对Ali-Qwen2-API的支持
3. 增加了`PretrainedEmbedding`，用于提供预训练的向量化模型
4. 修改了其他bug.

------

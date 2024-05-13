# 本地部署的开源大模型 API List

本小节罗列了可以本地部署的开源大模型 API List，如果您自己有自己的GPU计算资源，就可以部署自己本地化的模型服务，不依赖于线上的大模型API接口。在`zlai`中在线模型与本地模型使用方式完全一致，您可以根据需要选择使用。

## 模型列表

| 模型类型                                           | 模型名称                         | 配置类                                 | 备注 |
|------------------------------------------------|------------------------------|-------------------------------------|----|
| [chatglm](https://hf-mirror.com/THUDM)         | `chatglm3-6b`                | `ChatGLM6BGenerateConfig()`         |    |
| [chatglm](https://hf-mirror.com/THUDM)         | `chatglm3-6b-32k`            | -                                   |    |
| [chatglm](https://hf-mirror.com/THUDM)         | `chatglm3-6b-128k`           | `ChatGLM6B128KGenerateConfig()`     |    |
| [chatglm](https://hf-mirror.com/THUDM)         | `chatglm3-6b-base`           | -                                   |    |
| [baichuan](https://hf-mirror.com/baichuan-inc) | `baichuan2-7b-chat`          | -                                   |    |
| [baichuan](https://hf-mirror.com/baichuan-inc) | `baichuan2-7b-base`          | -                                   |    |
| [baichuan](https://hf-mirror.com/baichuan-inc) | `baichuan2-13b-chat`         | -                                   |    |
| [baichuan](https://hf-mirror.com/baichuan-inc) | `baichuan2-13b-base`         | -                                   |    |
| [qwen](https://hf-mirror.com/Qwen)             | `qwen-1_8b`                  | -                                   |    |
| [qwen](https://hf-mirror.com/Qwen)             | `qwen-1_8b-chat`             | -                                   |    |
| [qwen](https://hf-mirror.com/Qwen)             | `qwen-7b`                    | -                                   |    |
| [qwen](https://hf-mirror.com/Qwen)             | `qwen-7b-chat`               | -                                   |    |
| [qwen](https://hf-mirror.com/Qwen)             | `qwen-14b`                   | -                                   |    |
| [qwen](https://hf-mirror.com/Qwen)             | `qwen-14b-chat`              | -                                   |    |
| [qwen-vl](https://hf-mirror.com/Qwen)          | `qwen-vl-chat`               | -                                   |    |
| [qwen-1.5](https://hf-mirror.com/Qwen)         | `qwen1.5-7b-chat`            | `Qwen15Chat7BGenerateConfig()`      |    |
| [qwen-1.5](https://hf-mirror.com/Qwen)         | `qwen1.5-14b-chat`           | `Qwen15Chat14BGenerateConfig()`     |    |
| [qwen-1.5](https://hf-mirror.com/Qwen)         | `qwen1.5-72b-chat-awq`       | `Qwen15Chat17BAWQGenerateConfig()`  |    |
| [qwen-1.5](https://hf-mirror.com/Qwen)         | `qwen1.5-14b-chat-gptq-int4` | `Qwen15Chat72BInt4GenerateConfig()` |    |
| [qwen-1.5](https://hf-mirror.com/Qwen)         | `qwen1.5-14b-chat-gptq-int8` | `Qwen15Chat72BInt8GenerateConfig()` |    |
| [internlm](https://hf-mirror.com/internlm)     | `internlm-7b`                | -                                   |    |
| [internlm](https://hf-mirror.com/internlm)     | `internlm-chat-7b`           | -                                   |    |
| [internlm](https://hf-mirror.com/internlm)     | `internlm-20b`               | -                                   |    |
| [internlm](https://hf-mirror.com/internlm)     | `internlm-chat-20b`          | -                                   |    |
| [yi](https://hf-mirror.com/01-ai)              | `yi-34b`                     | -                                   |    |
| [yi](https://hf-mirror.com/01-ai)              | `yi-34b-chat`                | -                                   |    |
| [yi](https://hf-mirror.com/01-ai)              | `yi-34b-200k`                | -                                   |    |
| [yi](https://hf-mirror.com/01-ai)              | `yi-6b`                      | -                                   |    |
| [yi](https://hf-mirror.com/01-ai)              | `yi-6b-chat`                 | -                                   |    |
| [yi](https://hf-mirror.com/01-ai)              | `yi-6b-200k`                 | -                                   |    |
| [llama]()                                      | `llama-coder-7b-instruct`    | -                                   |    |
| [llama]()                                      | `llama-coder-33b-instruct`   | -                                   |    |
| [xverse](https://hf-mirror.com/xverse)         | `XVERSE-65B`                 | -                                   |    |
| [xverse](https://hf-mirror.com/xverse)         | `XVERSE-65B-2`               | -                                   |    |
| [xverse](https://hf-mirror.com/xverse)         | `XVERSE-65B-chat`            | -                                   |    |


## 模型调用

> 这里仅使用`chatglm3-6b`作为示例，其他模型使用方式完全一致。本地部署的开源模型统一使用`LocalLLMAPI`方法进行调用。

> `Query`推理示例

```python
from zlai.llms import LocalLLMAPI, ChatGLM6BGenerateConfig

# 创建模型推理配置
generate_config = ChatGLM6BGenerateConfig(
    max_tokens=1500,               # 用于指定模型在生成内容时token的最大数量
    top_p=0.8,                     # 生成过程中核采样方法概率阈值
    temperature=0.85,              # 用于控制随机性和多样性的程度
    stream=False,                  # 是否使用流式输出
)

# 创建大模型
llm = LocalLLMAPI(generate_config=generate_config)

completion = llm.generate(query="你好")
```

**Tips**: *其他方法如`Message`推理，模型输出数据类型、输出结果的解析都与前面线上调用模型的使用方式一致。这里不做更为细致的展开。详细的`Message`管理机制，请[参考](/doc/zlai-message-01.md)*

-----
@2024/04/28

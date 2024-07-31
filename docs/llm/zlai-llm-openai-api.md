# 开源大模型部署

> ZLAI在版本(0.3.104+)后会陆续支持本地化部署大模型，包括但不限于ChatGLM、Qwen、LLaMA、Baichuan等，并以`OpenAI`的接口形式提供，方便用户快速接入和使用，并且能够支持快速与ZLAI的其它功能进行整合，例如`ZLAI`的`Embedding`、`Agent`、`Prompt`、`LLM`、`Chat`等。

## 部署方式

> 部署方式

```bash
# 更新 ZLAI Package
pip install zlai["local"] -U

# 启动本地化部署
zlai_models --config_path "./models_config.yml" --reload=0 --port=8000 --host "0.0.0.0"
```

> 部署参数

- `config_path`: 模型配置文件路径，配置文件[参考](https://github.com/zlai-llm/zlai/blob/master/models_config.yml)
- `reload`: 在代码或文件更新后API是否热更新，默认为`0`，不自动加载更新
- `port`: API服务端口，默认为`8000`
- `host`: API服务地址，默认为`localhost`，指定为`0.0.0.0`后可以通过`IP`访问

> [配置文件说明](https://github.com/zlai-llm/zlai/blob/master/models_config.yml)

```yaml
models_config:
  - model_name: Qwen2-0.5B-Instruct                      # 模型名称
    model_path: /home/models/Qwen/Qwen2-0.5B-Instruct    # 模型文件路径
    model_type: completion                               # 模型类型，目前支持`completion`、`embedding`
    load_method: load_qwen2                              # 加载模型方法，目前支持`load_qwen2`、`load_glm4`
    max_memory:                                          # 模型最大显存
      0: "2GB"
  - model_name: Qwen2-1.5B-Instruct
    model_path: /home/models/Qwen/Qwen2-1.5B-Instruct
    model_type: completion
    load_method: load_qwen2
    max_memory:
      0: "4GB"
```

## API-Doc

> [http://localhost:8000/docs](http://localhost:8000/docs)

<center>
<img src="img/llm/zlai-llm-openai-api-doc.png" width="100%">
</center>

## 模型列表

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000")
models_list = client.models.list()
print(models_list.data)
```

*输出*

```text
[Model(id='Qwen2-0.5B-Instruct', created=None, object='model', owned_by='Open Source'),
 Model(id='Qwen2-1.5B-Instruct', created=None, object='model', owned_by='Open Source'),
 Model(id='Qwen2-7B-Instruct', created=None, object='model', owned_by='Open Source'),
 Model(id='Qwen2-57B-A14B-Instruct-GPTQ-Int4', created=None, object='model', owned_by='Open Source'),
 Model(id='glm-4-9b-chat', created=None, object='model', owned_by='Open Source'),
 Model(id='glm-4-9b-chat-1m', created=None, object='model', owned_by='Open Source'),
 Model(id='glm-4v-9b', created=None, object='model', owned_by='Open Source')]
```

## 模型请求

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000")
completion = client.chat.completions.create(
    model="glm-4-9b-chat",
    messages=[
        {"role": "user", "content": "你好"},
    ],
    stream=False
)
print(completion.choices[0].message.content)
```

*输出*

```text
你好👋！很高兴见到你，有什么可以帮助你的吗？
```

## Function Call

```python
from openai import OpenAI

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        }
    },
]

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000")
completion = client.chat.completions.create(
    model="glm-4-9b-chat",
    messages=[
        {"role": "user", "content": "What's the Celsius temperature in San Francisco?"},
    ],
    tools=tools,
    tool_choice="auto",
    stream=False,
)
print(completion.choices[0].message.tool_calls[0].function.name)
print(completion.choices[0].message.tool_calls[0].function.arguments)
```

*输出*

```text
get_current_weather

{"location": "San Francisco, CA", "format": "celsius"}
```

> 已经通过测试的模型列表

| 系列    | 模型                                | 普通推理 | 流式推理 | FunctionCall |
|-------|-----------------------------------|------|------|--------------|
| Qwen2 | Qwen2-0.5B-Instruct               | 通过   | 通过   | 暂不支持         |
| Qwen2 | Qwen2-1.5B-Instruct               | 通过   | 通过   | 暂不支持         |
| Qwen2 | Qwen2-7B-Instruct                 | 通过   | 通过   | 暂不支持         |
| Qwen2 | Qwen2-57B-A14B-Instruct-GPTQ-Int4 | 通过   | 通过   | 暂不支持         |
| GLM4  | glm-4-9b-chat                     | 通过   | 暂不支持 | 通过           |
| GLM4  | glm-4-9b-chat-1m                  | 通过   | 暂不支持 | 通过           |
| GLM4  | glm-4v-9b                         | 通过   | 暂不支持 | 通过           |

## End

------

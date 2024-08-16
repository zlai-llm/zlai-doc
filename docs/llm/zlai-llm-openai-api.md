# 开源大模型部署

> ZLAI在版本(0.3.104+)后会陆续支持本地化部署开源大模型，包括但不限于ChatGLM、Qwen、LLaMA、Baichuan等，并以`OpenAI`的接口形式提供，方便用户快速接入和使用，并且能够支持快速与ZLAI的其它功能进行整合，例如`ZLAI`的`Embedding`、`Agent`、`Prompt`、`LLM`、`Chat`等。

## 已经通过测试的模型列表

> [![Python package](https://img.shields.io/pypi/v/zlai)](https://pypi.org/project/zlai/)支持部署以下模型（[对话](llm/zlai-llm-openai-api?id=chat-completion)、[Function Call](llm/zlai-llm-openai-api?id=function-call)、[多模态](llm/zlai-llm-openai-api?id=多模态模型)、[文生图](llm/zlai-llm-openai-api?id=image-generate)、[图生图](llm/zlai-llm-openai-api?id=image-edit)、[文字转语音`TTS`](llm/zlai-llm-openai-api?id=audio)、[Embedding](llm/zlai-llm-openai-api?id=embedding)），并且已经通过`OpenAI-SDK`测试，可以正常使用。

| 类型         | 系列        | 模型                                                                                                 | 普通推理 | 流式推理 | FunctionCall |
|------------|-----------|----------------------------------------------------------------------------------------------------|------|------|--------------|
| 对话         | Qwen2     | [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)                             | ✔️   | ✔️   | 暂不支持         |
| 对话         | Qwen2     | [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct)                             | ✔️   | ✔️   | 暂不支持         |
| 对话         | Qwen2     | [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)                                 | ✔️   | ✔️   | 暂不支持         |
| 对话         | Qwen2     | [Qwen2-57B-A14B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4) | ✔️   | ✔️   | 暂不支持         |
| 对话         | GLM4      | [glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat)                                        | ✔️   | ✔️   | ✔️           |
| 对话         | GLM4      | [glm-4-9b-chat-1m](https://huggingface.co/THUDM/glm-4-9b-chat-1m)                                  | ✔️   | ✔️   | ✔️           |
| 多模态        | GLM4      | [glm-4v-9b](https://huggingface.co/THUDM/glm-4v-9b)                                                | ✔️   | ✔️   | ✔️           |
| 多模态        | MiniCPM   | [MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)                                      | ✔️   | ✔️   | ✔️           |
| 文生图        | Kolors    | [Kolors-diffusers](https://huggingface.co/Kwai-Kolors/Kolors-diffusers)                            | ✔️   | ❌    | ❌            |
| 图生图        | Kolors    | [Kolors-image2image](https://huggingface.co/Kwai-Kolors/Kolors-diffusers)                          | ✔️   | ❌    | ❌            |
| 文字转语音`TTS` | CosyVoice | [CosyVoice-300M](https://huggingface.co/FunAudioLLM/CosyVoice-300M)                                | ✔️   | ❌    | ❌            |
| Embedding  | BGE       | [bge-m3](https://huggingface.co/BAAI/bge-m3)                                                       | ✔️   | ❌    | ❌            |

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

## Model List

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
 Model(id='glm-4v-9b', created=None, object='model', owned_by='Open Source'),
 Model(id='bge-m3', created=None, object='model', owned_by='Open Source'),
 Model(id='kolors-diffusers', created=None, object='model', owned_by='Open Source'),
 Model(id='mini_cpm-v2_6', created=None, object='model', owned_by='Open Source')]
```

## Chat Completion

> 通用问答接口示例

### Chat Completion

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

### Stream Chat Completion

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000")
completion = client.chat.completions.create(
    model="glm-4-9b-chat",
    messages=[
        {"role": "user", "content": "你好"},
    ],
    stream=True
)

for chunk in completion:
    content = chunk.choices[0].delta.content
    print(content, flush=True, end='')
```

*输出*

```text
你好
你好👋！很高兴
你好👋！很高兴见到
你好👋！很高兴见到你
你好👋！很高兴见到你，有什么
你好👋！很高兴见到你，有什么可以帮助
你好👋！很高兴见到你，有什么可以帮助你的
你好👋！很高兴见到你，有什么可以帮助你的吗
你好👋！很高兴见到你，有什么可以帮助你的吗？ 
```

## Function Call

> 工具调用接口示例

> tools example

```python
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
```

### Normal Function Call

#### Function Call

> code

```python
from openai import OpenAI

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

#### Call Function

```python
from zlai.types.messages import ToolMessage

messages=[
    UserMessage(content="What's the Celsius temperature in San Francisco?"),
    completion.choices[0].message,
    ToolMessage(content="{San Francisco, CA: {celsius: 15}}"),
]

completion = client.chat.completions.create(
    model="glm-4-9b-chat", messages=messages,
    tools=tools, tool_choice="auto", stream=False,
)
```

*输出*

```text
The current Celsius temperature in San Francisco is 15 degrees.
```

### Stream Function Call

#### Function Call

> code

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000")
completion = client.chat.completions.create(
    model="glm-4-9b-chat",
    messages=[
        {"role": "user", "content": "What's the Celsius temperature in San Francisco?"},
    ],
    tools=tools,
    tool_choice="auto",
    stream=True,
)

answer = ""
for chunk in completion:
    content = chunk.choices[0].delta.content
    if content:
        answer += content
        print(answer)
```

*输出*

```text
 get_current_weather

 get_current_weather
{"location": 
 get_current_weather
{"location": "San 
 get_current_weather
{"location": "San Francisco, 
 get_current_weather
{"location": "San Francisco, CA", 
 get_current_weather
{"location": "San Francisco, CA", "format": 
 get_current_weather
{"location": "San Francisco, CA", "format": "celsius"} 
```

> Function Call Parameters

```python
print("content: ", chunk.choices[0].delta.content)
if chunk.choices[0].delta.tool_calls:
    print("function name: ", chunk.choices[0].delta.tool_calls[0].function.name)
    print("function arguments: ", chunk.choices[0].delta.tool_calls[0].function.arguments)
```

*输出*

```text
content:  None
function name:  get_current_weather
function arguments:  {"location": "San Francisco, CA", "format": "celsius"}
```

#### Call Function

```python
from zlai.types.messages import ToolMessage

messages=[
    UserMessage(content="What's the Celsius temperature in San Francisco?"),
    completion.choices[0].message,
    ToolMessage(content="{San Francisco, CA: {celsius: 15}}"),
]

completion = client.chat.completions.create(
    model="glm-4-9b-chat", messages=messages, tools=tools, 
    tool_choice="auto", stream=True, 
)

answer = ""
for chunk in completion:
    content = chunk.choices[0].delta.content
    if content:
        answer += content
        print(answer)
```

*输出*

```text
The 
The current 
The current Celsius 
The current Celsius temperature 
The current Celsius temperature in 
The current Celsius temperature in San 
The current Celsius temperature in San Francisco 
The current Celsius temperature in San Francisco is 
The current Celsius temperature in San Francisco is 15  
The current Celsius temperature in San Francisco is 15 degrees. 
```

## 多模态模型

**Example**: `glm-4v-9b`
**Image**: [Image](https://pic2.zhimg.com/v2-70ea697c0edec518b9d513a49228e489_b.jpg)

### Chat Completion

```python
from openai import OpenAI
from zlai.types.messages import ImageMessage, UserMessage, AssistantMessage

url = "https://pic2.zhimg.com/v2-70ea697c0edec518b9d513a49228e489_b.jpg"
messages = [
    ImageMessage(content="解析图片中的文字", images_url=[url]),
]
messages = [message.model_dump() for message in messages]

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000")
completion = client.chat.completions.create(
    model="glm-4v-9b",
    messages=messages,
)
print(completion.choices[0].message.content)
```

*输出*

```text
图片中的文字有“悦享先锋”、“B级先锋猎装SUV”、“宋L”和“知乎@七万钣金中里毅”。

- “悦享先锋”和“B级先锋猎装SUV”是宋L这款车的宣传语，突出了其作为一款B级先锋猎装SUV的特点。
- “宋L”是这款车的名称。
- “知乎@七万钣金中里毅”可能是图片来源或作者的标注。
```

### Stream Chat Completion

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000")
completion = client.chat.completions.create(
    model="glm-4v-9b",
    messages=messages,
    stream=True,
)

answer = ""
for chunk in completion:
    content = chunk.choices[0].delta.content
    answer += content
print(answer)
```

*输出*

```text
图片中的文字有“悦享先锋”、“B级先锋猎装SUV”、“宋L”和“知乎@七万钣金中里毅”。

其中，“悦享先锋”和“B级先锋猎装SUV”是宋L这款车的宣传语，突出了这款车的特点和定位；“宋L”是这款车的名称；“知乎@七万钣金中里毅”可能是图片来源或作者的标注。 
```

### 多图问答

> 对多个图片进行总结回答，当前仅`mini_cpm-v2_6`支持。

<div style="display: flex; justify-content: center;">
    <img src="img/llm/car-1.jpg" width="50%" style="margin-right: 10px;">
    <img src="img/llm/car-2.jpg" width="50%">
</div>

```python
from openai import OpenAI
from zlai.types.messages import ImageMessage, UserMessage, AssistantMessage

img_1 = "https://image11.m1905.cn/uploadfile/2008/1005/2006110205334520025.jpg"
img_2 = "https://pic2.zhimg.com/v2-70ea697c0edec518b9d513a49228e489_b.jpg"

messages = [
    UserMessage(content="你好"),
    AssistantMessage(content="你好"),
    ImageMessage(content="对比两张图的异同，给出结论。", images_url=[img_1, img_2]),
]
messages = [message.model_dump() for message in messages]

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000")
completion = client.chat.completions.create(
    model="mini_cpm-v2_6",
    messages=messages,
    stream=True,
)

for chunk in completion:
    content = chunk.choices[0].delta.content
    print(content, flush=True, end='')
```

*输出*

```text
这两张图片展示的是两种不同类型的车辆，一个是动画电影中的赛车，另一个是现实世界中的现代SUV。以下是它们的异同对比：

### 相同点：
1. **都是汽车**：两张图片中的主体都是汽车。
2. **车轮设计**：两辆车都配备了轮胎。

### 不同点：
1. **风格与类型**：
   - 第一张图（REF_1_FIG）展示的是一个动画角色，具体来说是一个绿色的赛车，具有拟人化的设计，有眼睛和嘴巴等面部特征。
   - 第二张图（REF_2_FIG）展示的是一辆现实世界的SUV，具有现代化的设计，没有拟人化的特征。

2. **颜色和外观**：
   - 第一张图的赛车主要是绿色，带有各种赞助商的标志和数字“86”。
   - 第二张图的SUV是银色或浅蓝色，设计更加简洁和现代。

3. **环境背景**：
   - 第一张图的背景是一个赛车场，周围有许多观众席。
   - 第二张图的背景是沙漠，显示了这辆车在户外的使用场景。

4. **功能用途**：
   - 第一张图中的赛车主要用于竞技比赛，具有高性能的特点。
   - 第二张图中的SUV主要用于日常驾驶，注重舒适性和实用性。

综上所述，这两张图片展示了不同类型的汽车，一个是用于娱乐和竞赛的动画角色，另一个是用于日常使用的现代SUV。它们在风格、颜色、环境和功能上都有显著的不同。
```

## Vision

> 图像生成与图像编辑模型

### Image Generate

> 文生图模型`Text2Image`：根据提供的`Prompt`渲染生成图片。

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000")

respose = client.images.generate(
    model="kolors_diffusers",
    prompt="长城下雪啦，写唯美实风格。",
    size="1792x1024",
    response_format="b64_json",
)

from zlai.utils.image import *
trans_bs64_to_image(respose.data[0].b64_json)
```

*输出*

<center>
<img src="img/llm/kolors-diffusers-01.png" width="100%">
</center>

### Image Edit

> 图生图模型`Image2Image`：根据提供的图片和`Prompt`对图片进行编辑。

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000")

respose = client.images.edit(
    model="kolors_image2image",
    prompt="宫崎骏卡通风格",
    image=open("./data/top-20-small-dog-breeds.jpeg", "rb"),
    size="1024x1024",
    response_format="b64_json",
)
```

*输出*

<div style="display: flex; justify-content: center;">
    <div>
        <center>
        <img src="img/llm/kolors-diffusers-02.png" width="80%" style="margin-right: 10px;">
        <br>
        原始图
        </center>
    </div>
    <div>
        <center>
        <img src="img/llm/kolors-diffusers-03.png" width="83%">
        <br>
        生成图
        </center>
    </div>
</div>

## Audio

> 文字转语音 `TTS(text to speech)`

### TTS

> `voice`: 音色有`['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']`

```python
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000")

text = """
有时,我会静静地坐在庭院里,看着落叶飘飞,感慨万千。青春无痕,日子如白驹过隙。
我们曾经多么明朗且无忧无虑,如今却逐渐变得沧桑和迷惘。生命像一朵花,明明灿烂地绽放,最终却也枯萎凋谢。
"""

speech_file_path = "./audio.wav"
response = client.audio.speech.create(model="CosyVoice-300M-SFT", voice="中文女", input=text)

response.stream_to_file(speech_file_path)
```

*输出*

<audio controls>
  <source src="audio/audio.wav" type="audio/mpeg">
</audio>

## Embedding

### Text Embedding

> 文本嵌入模型`Embedding`：将文本转换为向量。

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000")

response = client.embeddings.create(input=["你好"], model="bge-m3")
print(len(response.data[0].embedding))
print(response.usage)
print(response.data[0].embedding[:5])
```

*输出*

```text
1024
Usage(prompt_tokens=0, total_tokens=2)
[-0.037132877856492996, 0.006196103058755398, -0.06525908410549164, -0.02504667080938816, -0.016926351934671402]
```

## End

------

本小节罗列了可以本地部署的开源大模型 API List，如果您自己有自己的GPU计算资源，就可以部署自己本地化的模型服务，不依赖于线上的大模型API接口。在`zlai`中在线模型与本地模型使用方式完全一致，您可以根据需要选择使用。

**Tips**: *其他方法如`Message`推理，模型输出数据类型、输出结果的解析都与前面线上调用模型的使用方式一致。这里不做更为细致的展开。详细的`Message`管理机制，请[参考](/doc/zlai-message-01.md)*

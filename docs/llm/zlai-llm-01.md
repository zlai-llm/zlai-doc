# 在线大模型的简单调用

> 本节介绍`zlai`中大模型的调研方式。

目前，`zlai`支持多种在线大模型的调研，包括`ZhipuAI`、`Ali`、`Atom`、`MoonShot`、`Baichuan`、`Baidu`等。在`zlai`中你只需定义`模型类`与`模型配置类`两个实例即可完成模型的调用，不同的平台上的大模型我们都使用了相同的调用输出规范。

本章节介绍大模型的API调用方式，并提供一些示例代码。包括了本地部署的模型API调用方式，以及使用大模型API的示例代码。

`ZLAI`统一封装了各类平台的配置类与API调用方式，用户可以方便地调用大模型API。只需要指定好你需要的模型类与配置类，即可快速调用大模型API。

## API KEY

在调用在线大模型时，您需要申请相关平台的`API KEY`，并将其配置到`环境变量`中，相关模型的`API-Key`获取方式可以查阅下面所列举的官网地址。以下是各个平台`API KEY`的申请地址与`zlai`中对应的环境变量标准名称：

| 公司                | 模型            | 官网文档                                                         | 环境变量名称                                |
|-------------------|---------------|--------------------------------------------------------------|---------------------------------------|
| 智谱AI              | `GLM`         | [Doc](https://open.bigmodel.cn/dev/api)                      | ZHIPU_API_KEY                         |
| 阿里-灵积             | `Qwen`        | [Doc](https://dashscope.aliyun.com/)                         | ALI_API_KEY                           |
| Llama Family      | `Atom`        | [Doc](https://llama.family/docs/api)                         | ATOM_API_KEY                          |
| 硅基智能 Silicon Flow | `Qwen/GLM/Yi` | [Doc](https://docs.siliconflow.cn/docs)                      | SILICON_FLOW_API_KEY                  |
| 百川                | `Baichuan`    | [Doc](https://platform.baichuan-ai.com/docs/api)             | BAICHUAN_API_KEY                      |
| 深度求索              | `DeepSeek`    | [Doc](https://platform.deepseek.com/api-docs/zh-cn/)         | DEEPSEEK_API_KEY                      |
| 百度 千帆             | `Ernie`       | [Doc](https://cloud.baidu.com/doc/WENXINWORKSHOP/index.html) | QIANFAN_ACCESS_KEY/QIANFAN_SECRET_KEY |
| 字节跳动              | 豆包 `Doubao`   | [Doc](https://www.volcengine.com/docs/82379/1263512)         | DOUBAO_API_KEY                        |
| 科大讯飞              | 花火 `Spark`    | [Doc](https://xinghuo.xfyun.cn/sparkapi)                     | SPARK_API_KEY                         |
| 腾讯                | 混元 `HunYuan`  | [Doc](https://cloud.tencent.com/product/hunyuan)             | HUNYUAN_API_KEY                       |
| 零一万物              | `Yi`          | [Doc](https://platform.lingyiwanwu.com/)                     | YI_API_KEY                            |
| 月之暗面              | `moonshot`    | [Doc](https://platform.moonshot.cn)                          | MOONSHOT_API_KEY                      |

> 环境变量的配置方式

<center>
<img src="./img/zlai-llm-01.png" width="480px">
</center>

以下是一些大模型调用示例。

## ZhipuAI

> 配置API-Key

- API Key Name: `ZHIPU_API_KEY`
- Value: `API-Key`

> **模型方法**: `Zhipu`用于调用智谱AI的模型。

```python
from zlai.llms import Zhipu
```

**参数说明**

- `api_key`: 给定已经申请好的`API KEY`，默认为`None`，您也可以在环境变量中配置`ZHIPU_API_KEY`，对于已经配置好环境变量的情况下，可以不填写该参数。
- `generate_config`: 用于给定大模型生成内容时的参数配置，默认为`None`，后面会着重介绍这个参数，一般情况下只需要指定这一个参数就可以进行模型推理了。相关参数的理论介绍[参考](/doc/zlai-llm-04.md)。
- `output`: 指定输出信息的格式，枚举值 `["completion", "message", "str"]`。
    - `completion`: 按照在线API官方模式输出完整的返回信息。
    - `message`: 只输出`Message`信息，`role='Assistant' content='模型回答的内容...'`。
    - `str`: 只输出模型回答的文本信息。
- `verbose`: 是否展示模型的推理调用过程，默认为`False`。
- `api_key_name`: 默认为`ZHIPU_API_KEY`，你也可以指定其他的名称，前提是需要在环境变量中配置该名称与相应的变量值。
- `async_max_request_time`: 异步调用情况下的最大请求时间，默认为`600`s。

> **模型配置**

- `ZhipuGLM3Turbo`: 模型`glm-3-turbo`的默认参数类
- `ZhipuGLM4`: 模型`glm-4`的默认参数类

```python
# 导入智谱大模型配置类
from zlai.llms import (
    GLM3TurboGenerateConfig, GLM4GenerateConfig
)
```

**默认参数**

- `do_sample: Optional[bool]`: `do_sample` 为 `True` 时启用采样策略，`do_sample` 为 `False` 时采样策略 `temperature`、`top_p` 将不生效。
- `stream: Optional[bool]`: 此参数应当设置为 `Fasle` 或者省略。表示模型生成完所有内容后一次性返回所有内容。如果设置为 `True`，模型将通过标准 `Event Stream`，逐块返回模型生成内容。
- `temperature: Optional[float]`: 采样温度，控制输出的随机性，必须为正数取值范围是`(0.0, 1.0)`，不能等于 `0`，默认值为 `0.95`，值越大会使输出更随机，更具创造性；值越小输出会更加稳定或确定建议您根据应用场景调整 `top_p` 或 `temperature` 参数，但不要同时调整两个参数。
- `top_p: Optional[float]`: 用温度取样的另一种方法，称为核取样取值范围是 `(0.0, 1.0)` 开区间，不能等于 `0` 或 `1`，默认值为 `0.7` 模型考虑具有 `top_p` 概率质量 `tokens` 的结果。
- `max_tokens: Optional[int]`: 模型输出最大 `tokens`，最大输出为 `8192`，默认值为 `1024`

**完整调用示例**

```python
from zlai.llms import Zhipu, GLM3TurboGenerateConfig, GLM4GenerateConfig

# 创建模型推理配置
generate_config = GLM3TurboGenerateConfig(
    max_tokens=1500,               # 用于指定模型在生成内容时token的最大数量
    top_p=0.8,                     # 生成过程中核采样方法概率阈值
    temperature=0.85,              # 用于控制随机性和多样性的程度
)

# 创建大模型
llm = Zhipu(generate_config=generate_config)

# 如果您只需要使用默认的参数配置，可以直接使用以下方式创建模型
# llm = Zhipu(generate_config=ZhipuGLM3Turbo())
# llm = Zhipu(generate_config=ZhipuGLM4())
# query 为你需要问大模型的问题
completion = llm.generate(query="你好")

print(completion.choices[0].message.content)
```

**返回信息**

```text
>>> 你好👋！我是人工智能助手智谱清言（ChatGLM），很高兴见到你，欢迎问我任何问题。
```

*PS: 模型输出结构*

这里仅以智谱AI的输出结构作为示例，其他平台的大模型输出结构均与智谱AI的输出结构相似，如果有其他较大的差异点会在后续的介绍中进行单独表述。

```json
{
  "created": 1703487403,
  "id": "8239375684858666781",
  "model": "glm-4",
  "request_id": "8239375684858666781",
  "choices": [
      {
          "finish_reason": "stop",
          "index": 0,
          "message": {
              "content": "智绘蓝图，AI驱动 —— 智谱AI，让每一刻创新成为可能。",
              "role": "assistant"
          }
      }
  ],
  "usage": {
      "completion_tokens": 217,
      "prompt_tokens": 31,
      "total_tokens": 248
  }
}
```

模型会以`completion`对象的形式输出，我们需要关注`completion`对象的以下字段：

| 名称                        | 描述                                                                                        |
|---------------------------|-------------------------------------------------------------------------------------------|
| `model`                   | 模型名称                                                                                      |
| `choices`                 | 模型推理输出的结果，有一些模型可以根据随机采样的方式输出多个结果供选择，但目前大部分平台一般只输出一个结果，所以一般情况下`choices`是`List[Dict]`的数据形式。 |
| `choices/finish_reason`   | 模型推理结束的原因                                                                                 |
| `choices/message`         | 模型推理返回的消息                                                                                 |
| `choices/message/role`    | 模型推理返回的角色名称                                                                               |
| `choices/message/content` | 模型推理返回的消息内容                                                                               |
| `usage`                   | 记录了`token`的使用情况                                                                           |

> 模型价格

| 模型名称          | 配置类                | 备注               |
|---------------|--------------------|------------------|
| `glm-4`       | `ZhipuGLM4()`      | 0.1元 / 千tokens   |
| `glm-3-turbo` | `ZhipuGLM3Turbo()` | 0.005元 / 千tokens |

## Ali

> 配置API-Key

- API Key Name: `ALI_API_KEY`
- Value: `API-Key`

> **模型方法**: `Ali`用于调用阿里的线上模型。
 
您可以在[官网](https://dashscope.aliyun.com/)上申请`API KEY`，并将其配置到`环境变量`中，也可以在官网上查看最新的文档信息。

```python
from zlai.llms import Ali
```

**参数说明**: 参考`ZhipuAI`的参数说明。

> **模型配置**

`zlai`封装了几乎所有阿里的线上大模型，您只需要指定一个模型配置类即可完成模型的调用配置生成。相关参数的配置可以参考上文`Zhipu`部分，你也可以参考[官网的文档](https://help.aliyun.com/zh/dashscope/developer-reference/model-square/)。

| 模型名称                         | 配置类                                         | 备注    |
|------------------------------|---------------------------------------------|-------|
| `Qwen-Turbo`                 | `AliQwenTurboGenerateConfig()`              | 以官网为准 |
| `Qwen-Plus`                  | `AliQwenPlusGenerateConfig()`               | 以官网为准 |
| `Qwen-Max`                   | `AliQwenMaxGenerateConfig()`                | 以官网为准 |
| `Qwen-Max1201`               | `AliQwenMax1201GenerateConfig()`            | 以官网为准 |
| `Qwen-Max0428`               | `AliQwenMax0428GenerateConfig()`            | 以官网为准 |
| `Qwen-Max0403`               | `AliQwenMax0403GenerateConfig()`            | 以官网为准 |
| `Qwen-Max0107`               | `AliQwenMax0107GenerateConfig()`            | 以官网为准 |
| `Qwen-MaxLongContext`        | `AliQwenMaxLongContextGenerateConfig()`     | 以官网为准 |
| `qwen2-57b-a14b-instruct`    | `AliQwen2Instruct57BA14BGenerateConfig()`   | 以官网为准 |
| `qwen2-72b-instruct`         | `AliQwen2Instruct72BGenerateConfig()`       | 以官网为准 |
| `qwen2-7b-instruct`          | `AliQwenInstruct27BGenerateConfig()`        | 以官网为准 |
| `qwen2-1.5b-instruct`        | `AliQwen2Instruct15BGenerateConfig()`       | Free  |
| `qwen2-0.5b-instruct`        | `AliQwen2Instruct05BGenerateConfig()`       | Free  |
| `qwen1.5-110b-chat`          | `AliQwen15Chat110BGenerateConfig()`         | 以官网为准 |
| `Qwen1.5-Chat-72B`           | `AliQwen15Chat72BGenerateConfig()`          | 以官网为准 |
| `qwen1.5-32b-chat`           | `AliQwen15Chat32BGenerateConfig()`          | 以官网为准 |
| `Qwen1.5-Chat-14B`           | `AliQwen15Chat14BGenerateConfig()`          | 以官网为准 |
| `Qwen1.5-Chat-7B`            | `AliQwen15Chat7BGenerateConfig()`           | 以官网为准 |
| `qwen1.5-1.8b-chat`          | `AliQwen15Chat18BGenerateConfig()`          | Free  |
| `qwen1.5-0.5b-chat`          | `AliQwen15Chat05BGenerateConfig()`          | Free  |
| `codeqwen1.5-7b-chat`        | `AliQwen15Code7BGenerateConfig()`           | 以官网为准 |
| `Qwen-Chat-72B`              | `AliQwenChat72BGenerateConfig()`            | 以官网为准 |
| `Qwen-Chat-14B`              | `AliQwenChat14BGenerateConfig()`            | 以官网为准 |
| `Qwen-Chat-7B`               | `AliQwenChat7BGenerateConfig()`             | 以官网为准 |
| `Qwen-Chat-1.8B`             | `AliQwenChat18BGenerateConfig()`            | Free  |
| `Qwen-Chat-1.8B-LongContext` | `AliQwenChat18BLongContextGenerateConfig()` | -     |

```python
# 导入阿里大模型配置类
from zlai.llms import (
    AliQwenTurboGenerateConfig,
    AliQwenPlusGenerateConfig,
    AliQwenMaxGenerateConfig,
    AliQwenMax1201GenerateConfig,
    AliQwenMaxLongContextGenerateConfig,
    AliQwen15Chat72BGenerateConfig,
    AliQwen15Chat14BGenerateConfig,
    AliQwen15Chat7BGenerateConfig,
    AliQwenChat72BGenerateConfig,
    AliQwenChat14BGenerateConfig,
    AliQwenChat7BGenerateConfig,
    AliQwenChat18BGenerateConfig,
    AliQwenChat18BLongContextGenerateConfig,
)
```

**完整调用示例**

```python
from zlai.llms import Ali, AliQwenTurboGenerateConfig

# 创建模型推理配置
generate_config = AliQwenTurboGenerateConfig(
    max_tokens=1500,               # 用于指定模型在生成内容时token的最大数量
    top_p=0.8,                     # 生成过程中核采样方法概率阈值
    temperature=0.85,              # 用于控制随机性和多样性的程度
)

# 创建大模型
llm = Ali(generate_config=generate_config)

# 如果您只需要使用默认的参数配置，可以直接使用以下方式创建模型
# llm = Ali(generate_config=AliQwenTurboGenerateConfig())
# query 为你需要问大模型的问题
completion = llm.generate(query="你好")

# 相对于其他平台，阿里返回结构多一个`output`。
print(completion.output.choices[0].message.content)
```

**返回信息**

```text
你好！很高兴为你提供帮助。有什么我可以解答的问题吗？
```

## Atom

> 配置API-Key

- API Key Name: `ATOM_API_KEY`
- Value: `API-Key`

Atom是由[Llama中文社区](https://llama.family/)和AtomEcho（原子回声）联合研发的大模型，基于Llama2-7B采用大规模的中文数据进行了继续预训练，目前其官网上提供`Atom-1B/Atom-13B/Atom-7B`三个可免费调用的大模型。同时也提供了最新基于`Llama3`微调的`Llama3-Chinese-8B-Instruct`可供API调用。

> **模型方法**: `Ali`用于调用阿里的线上模型。

您可以在[Llama中文社区](https://llama.family/)上申请`API KEY`，并将其配置到`环境变量`中，也可以在官网上查看最新的文档信息。

```python
from zlai.llms import Atom
```

**参数说明**: 参考`ZhipuAI`的参数说明。

> **模型配置**

`zlai`封装了`Atom-13B/Atom-7B/Llama3-Chinese-8B-Instruct`，同样的您只需要指定一个模型配置类即可完成模型的调用配置生成。相关参数的配置可以参考上文`Zhipu`部分，你也可以参考[官网的文档](https://llama.family/docs/api)。

- `Atom1BGenerateConfig`,
- `Atom7BGenerateConfig`,
- `Atom13BGenerateConfig`,
- `Llama3Chinese8BInstruct`,

```python
# 导入Atom大模型配置类
from zlai.llms import (
    Atom1BGenerateConfig,
    Atom7BGenerateConfig,
    Atom13BGenerateConfig,
    Llama3Chinese8BInstruct,
)
```

**完整调用示例**

```python
from zlai.llms import Atom, Atom7BGenerateConfig

# 创建模型推理配置
generate_config = Atom7BGenerateConfig(
    max_tokens=1500,               # 用于指定模型在生成内容时token的最大数量
    top_p=0.8,                     # 生成过程中核采样方法概率阈值
    temperature=0.85,              # 用于控制随机性和多样性的程度
)

# 创建大模型
llm = Atom(generate_config=generate_config)

# 如果您只需要使用默认的参数配置，可以直接使用以下方式创建模型
# llm = Ali(generate_config=Atom7BGenerateConfig())
# query 为你需要问大模型的问题
completion = llm.generate(query="你好")

print(completion.choices[0].message.content)
```

**返回信息**

```text
你好！有什么我可以帮助你的吗？
```

> 模型价格: 目前免费。

## SiliconFlow

> 配置API-Key

- API Key Name: `SILICON_FLOW_API_KEY`
- Value: `API-Key`

> [SiliconFlow](https://siliconflow.cn/zh-cn/siliconcloud)提供一系列大模型API服务，对以下大模型提供免费调用。您可以访问其官网查阅模型细节。`zlai`封装了以下模型：

| model                                          | config                                | price                                              |
|------------------------------------------------|---------------------------------------|----------------------------------------------------|
| `Qwen/Qwen2-7B-Instruct (32K)`                 | Qwen2Instruct7BGenerateConfig         | free                                               |
| `Qwen/Qwen2-1.5B-Instruct (32K)`               | Qwen2Instruct15BGenerateConfig        | free                                               |
| `Qwen/Qwen1.5-7B-Chat (32K)`                   | Qwen15Chat7BGenerateConfig            | free                                               |
| `THUDM/glm-4-9b-chat (32K)`                    | GLM3Chat6BGenerateConfig              | free                                               |
| `THUDM/chatglm3-6b (32K)`                      | GLM4Chat9BGenerateConfig              | free                                               |
| `01-ai/Yi-1.5-9B-Chat-16K (16K)`               | Yi15Chat6BGenerateConfig              | free                                               |
| `01-ai/Yi-1.5-6B-Chat (4K)`                    | Yi15Chat9BGenerateConfig              | free                                               |
| `01-ai/Yi-1.5-6B-Chat (4K)`                    | Yi15Chat9BGenerateConfig              | free                                               |
| `Qwen/Qwen2-72B-Instruct (32K)`                | Qwen2Instruct72BGenerateConfig        | [价格表](https://docs.siliconflow.cn/docs/大语言模型-计费规则) |
| `Qwen/Qwen2-57B-A14B-Instruct (32K)`           | Qwen2Instruct57BA14BGenerateConfig    | [价格表](https://docs.siliconflow.cn/docs/大语言模型-计费规则) |
| `Qwen/Qwen1.5-110B-Chat (32K)`                 | Qwen15Chat110BGenerateConfig          | [价格表](https://docs.siliconflow.cn/docs/大语言模型-计费规则) |
| `Qwen/Qwen1.5-32B-Chat (32K)`                  | Qwen15Chat32BGenerateConfig           | [价格表](https://docs.siliconflow.cn/docs/大语言模型-计费规则) |
| `Qwen/Qwen1.5-14B-Chat (32K)`                  | Qwen15Chat14BGenerateConfig           | [价格表](https://docs.siliconflow.cn/docs/大语言模型-计费规则) |
| `deepseek-ai/DeepSeek-Coder-V2-Instruct (32K)` | Yi15Chat6BGenerateConfig              | [价格表](https://docs.siliconflow.cn/docs/大语言模型-计费规则) |
| `deepseek-ai/DeepSeek-V2-Chat (32K)`           | DeepSeekCoderV2InstructGenerateConfig | [价格表](https://docs.siliconflow.cn/docs/大语言模型-计费规则) |
| `deepseek-ai/deepseek-llm-67b-chat (32K)`      | DeepSeekV2ChatGenerateConfig          | [价格表](https://docs.siliconflow.cn/docs/大语言模型-计费规则) |
| `01-ai/Yi-1.5-34B-Chat-16K (16K)`              | DeepSeekLLM67BChatGenerateConfig      | [价格表](https://docs.siliconflow.cn/docs/大语言模型-计费规则) |

> 调用示例

```python
from zlai.llms.silicon_flow import *
from zlai.llms.generate_config.silicon_flow import *

config = [
    Qwen2Instruct7BGenerateConfig,
    Qwen2Instruct15BGenerateConfig,
    Qwen15Chat7BGenerateConfig,
    GLM3Chat6BGenerateConfig,
    GLM4Chat9BGenerateConfig,
    Yi15Chat6BGenerateConfig,
    Yi15Chat9BGenerateConfig,
]
for gen_config in config:
    llm = SiliconFlow(generate_config=gen_config())
    completion = llm.generate(query="你好")
    print(f"{gen_config.__name__.replace('GenerateConfig', '')}: {completion.choices[0].message.content}")
    print()
```

*输出*

```text
Qwen2Instruct7B: 你好！很高兴能为你提供帮助。有什么问题或需要我解答的吗？

Qwen2Instruct15B: 你好！有什么我可以帮助你的吗？

Qwen15Chat7B: 你好！有什么我能为你效劳的吗？

GLM3Chat6B: 你好！很高兴见到你，欢迎问我任何问题。

GLM4Chat9B: 你好👋！很高兴见到你，有什么可以帮助你的吗？

Yi15Chat6B: 你好！有什么我可以帮助你的吗？
```

## Baichuan

> 配置API-Key

- API Key Name: `BAICHUAN_API_KEY`
- Value: `API-Key`

> `zlai`封装了以下Baichuan模型：

| model                  | config                           | price                                           |
|------------------------|----------------------------------|-------------------------------------------------|
| `Baichuan4`            | Baichuan4GenerateConfig          | [price](https://platform.baichuan-ai.com/price) |
| `Baichuan3-Turbo`      | Baichuan3TurboGenerateConfig     | [price](https://platform.baichuan-ai.com/price) |
| `Baichuan3-Turbo-128k` | Baichuan3Turbo128kGenerateConfig | [price](https://platform.baichuan-ai.com/price) |
| `Baichuan2-Turbo`      | Baichuan2TurboGenerateConfig     | [price](https://platform.baichuan-ai.com/price) |
| `Baichuan2-Turbo-192k` | Baichuan2Turbo192kGenerateConfig | [price](https://platform.baichuan-ai.com/price) |


> 调用示例

```python
from zlai.llms.baichuan import *
from zlai.llms.generate_config.baichuan import *

config = [
    Baichuan4GenerateConfig,
    Baichuan3TurboGenerateConfig,
    Baichuan3Turbo128kGenerateConfig,
    Baichuan2TurboGenerateConfig,
    Baichuan2Turbo192kGenerateConfig,
]
for gen_config in config:
    llm = Baichuan(generate_config=gen_config())
    completion = llm.generate(query="你好")
    print(f"{gen_config.__name__.replace('GenerateConfig', '')}: {completion.choices[0].message.content}")
    print()
```

*输出*

```text
Baichuan4: 你好！有什么我可以帮助你的吗？

Baichuan3Turbo: 你好！有什么我可以帮助你的吗？

Baichuan3Turbo128k: 你好！有什么可以帮助你的吗？

Baichuan2Turbo: 你好，有什么我可以帮助你的吗？

Baichuan2Turbo192k: 你好，有什么我可以帮助你的吗？
```

## DeepSeek

> 配置API-Key

- API Key Name: `DEEPSEEK_API_KEY`
- Value: `API-Key`

> `zlai`封装了以下DeepSeek模型：

| model            | config                      | price                                                          |
|------------------|-----------------------------|----------------------------------------------------------------|
| `deepseek-chat`  | DeepSeekChatGenerateConfig  | [price](https://platform.deepseek.com/api-docs/zh-cn/pricing/) |
| `deepseek-coder` | DeepSeekCoderGenerateConfig | [price](https://platform.deepseek.com/api-docs/zh-cn/pricing/) |

> 调用示例

```python
from zlai.llms.deepseek import *
from zlai.llms.generate_config.deepseek import *

config = [
    DeepSeekChatGenerateConfig,
    DeepSeekCoderGenerateConfig,
]
for gen_config in config:
    llm = DeepSeek(generate_config=gen_config())
    completion = llm.generate(query="你好")
    print(f"{gen_config.__name__.replace('GenerateConfig', '')}: {completion.choices[0].message.content}")
    print()
```

*输出*

```text
DeepSeekChatGenerateConfig: 你好！有什么我可以帮助你的吗？

DeepSeekCoderGenerateConfig: 你好！有什么我可以帮助你的吗？
```

## Baidu

> 配置API-Key

- API Key Name: `QIANFAN_ACCESS_KEY/QIANFAN_SECRET_KEY`
- Value: `QIANFAN_ACCESS_KEY/QIANFAN_SECRET_KEY`

> 百度[千帆大模型](https://cloud.baidu.com/doc/WENXINWORKSHOP/index.html)，下面是百度的几款免费的大模型。

| series        | name                   | config                               | price |
|---------------|------------------------|--------------------------------------|-------|
| ERNIE Speed系列 | ERNIE-Speed-8K         | `ErnieSpeed8KGenerateConfig`         | free  |
| ERNIE Speed系列 | ERNIE-Speed-128K       | `ErnieSpeed128KGenerateConfig`       | free  |
| ERNIE Speed系列 | ERNIE Speed-AppBuilder | `ErnieSpeedAppBuilderGenerateConfig` | free  |
| ERNIE Lite系列  | ERNIE-Lite-8K          | `ErnieLite8KGenerateConfig`          | free  |
| ERNIE Lite系列  | ERNIE-Lite-8K-0922     | `ErnieLite8K0922GenerateConfig`      | free  |
| ERNIE Tiny系列  | ERNIE-Tiny-8K          | `ErnieTiny8KGenerateConfig`          | free  |

> 调用示例

```python
from zlai.llms import Baidu
from zlai.llms.generate_config.baidu import *

config = [
    ErnieSpeed8KGenerateConfig,
    ErnieSpeed128KGenerateConfig,
    ErnieSpeedAppBuilderGenerateConfig,
    ErnieLite8KGenerateConfig,
    ErnieLite8K0922GenerateConfig,
    ErnieTiny8KGenerateConfig,
]
for gen_config in config:
    llm = Baidu(generate_config=gen_config())
    completion = llm.generate(query="1+1=")
    print(f"{gen_config.__name__.replace('GenerateConfig', '')}: {completion.choices[0].message.content}")
    print()
```

*输出*

```text
ErnieSpeed8K: 您好，这是一个简单的数学问题，1+1等于2。
ErnieSpeed128K: 您好，这是一个简单的数学问题，1+1等于2。
ErnieSpeedAppBuilder: 因为算式的每个数相加的结果都是$0$，所以最终结果就是$0$。
ErnieLite8K: 计算结果为：$1+1 = 2$
ErnieLite8K0922: 1+1=2。这是一个基本的数学等式，代表一加一等于二。
ErnieTiny8K: “1+1=”是一个简单的数学表达式，表示两个相同的数字相加的结果。在数学中，这是基本的加法运算。因此，这个答案应该是“等于”。
```

## Doubao

> 配置API-Key

- API Key Name: `DOUBAO_API_KEY`
- Value: `API-Key`

> 字节跳动[豆包`Doubao`大模型](https://www.volcengine.com/docs/82379/1263512)

| 模型名称             | 简介                                                                      |
|:-----------------|:------------------------------------------------------------------------|
| Doubao-lite-4k   | Doubao-lite拥有极致的响应速度，更好的性价比，为客户不同场景提供更灵活的选择。支持4k上下文窗口的推理和精调。            |
| Doubao-lite-32k  | Doubao-lite拥有极致的响应速度，更好的性价比，为客户不同场景提供更灵活的选择。支持32k上下文窗口的推理和精调。           |
| Doubao-lite-128k | Doubao-lite 拥有极致的响应速度，更好的性价比，为客户不同场景提供更灵活的选择。支持128k上下文窗口的推理和精调。         |
| Doubao-pro-4k    | 效果最好的主力模型，适合处理复杂任务，在参考问答、总结摘要、创作、文本分类、角色扮演等场景都有很好的效果。支持4k上下文窗口的推理和精调。   |
| Doubao-pro-32k   | 效果最好的主力模型，适合处理复杂任务，在参考问答、总结摘要、创作、文本分类、角色扮演等场景都有很好的效果。支持32k上下文窗口的推理和精调。  |
| Doubao-pro-128k  | 效果最好的主力模型，适合处理复杂任务，在参考问答、总结摘要、创作、文本分类、角色扮演等场景都有很好的效果。支持128k上下文窗口的推理和精调。 |

> 调用示例

```python
from zlai.llms import DouBao, OpenAIGenerateConfig

models = [
    "ep-20240630100146-r89zj",
    "ep-20240630100032-29bh9",
    "ep-20240630100000-xvz2b",
    "ep-20240630095928-f9fl7",
    "ep-20240630095857-p9kbd",
    "ep-20240630095730-6c7sc",
]

for model in models:
    llm = DouBao(generate_config=OpenAIGenerateConfig(model=model))
    completion = llm.generate("1+1=")
    print(f"{model}: {completion.choices[0].message.content}")
    print()
```

*输出*

```text
ep-20240630100146-r89zj : 在常规的数学运算中，1+1=2。但在一些情况下，答案可能会有所不同。例如，在计算机的二进制中，1+1=10。如果是脑筋急转弯的话，答案可能会更加多样。
ep-20240630100032-29bh9 : 在常规的数学运算中，1+1=2。但在一些情况下，答案可能会有所不同。例如，在计算机的二进制中，1+1=10。如果是脑筋急转弯的话，答案可能会更加多样。
ep-20240630100000-xvz2b : 2
ep-20240630095928-f9fl7 : 1 + 1 = 2 
ep-20240630095857-p9kbd : 1+1=2
ep-20240630095730-6c7sc : 在常规的数学运算中，1+1=2。但在一些情况下，答案可能会有所不同。例如，在计算机的二进制中，1+1=10。如果是脑筋急转弯的话，答案可能会更加多样。
```

## Spark

> 配置API-Key

- API Key Name: `SPARK_API_KEY`
- Value: `API-Key`

> 科大讯飞[Spark大模型](https://xinghuo.xfyun.cn/sparkapi)

| model           | config                    | price                                      |
|-----------------|---------------------------|--------------------------------------------|
| `Spark-Lite`    | SparkLiteGenerateConfig   | [price](https://xinghuo.xfyun.cn/sparkapi) |
| `Spark-V2`      | SparkV2GenerateConfig     | [price](https://xinghuo.xfyun.cn/sparkapi) |
| `Spark-Pro`     | SparkProGenerateConfig    | [price](https://xinghuo.xfyun.cn/sparkapi) |
| `Spark-Max`     | SparkMaxGenerateConfig    | [price](https://xinghuo.xfyun.cn/sparkapi) |
| `Spark-4-Ultra` | Spark4UltraGenerateConfig | [price](https://xinghuo.xfyun.cn/sparkapi) |

*Tips:*

```text
# 控制台获取key和secret拼接，假使APIKey是key123456，APISecret是secret123456
api_key="key123456:secret123456", 
```

> 调用示例

```python
from zlai.llms import Spark
from zlai.llms.generate_config.spark import *

config = [
    SparkLiteGenerateConfig,
    SparkV2GenerateConfig,
    SparkProGenerateConfig,
    SparkMaxGenerateConfig,
    Spark4UltraGenerateConfig,
]
for gen_config in config:
    llm = Spark(generate_config=gen_config())
    data = llm.generate(query="1+1=")
    print(f"{gen_config.__name__.replace('GenerateConfig', '')}: {data.choices[0].message.content}")
    print()
```

*输出*

```text
SparkLite: 1+1的解答过程很简单，因为这是一个基本的算术加法。将两个数相加，即：\n\n$1+1$ =2\n\n所以，$1+1$ =2。
SparkV2: 2
SparkPro: $1+1$ =2
SparkMax: $1+1$ =2
Spark4Ultra: $1+1$ =2
```

## HunYuan

> 配置API-Key

- API Key Name: `HUNYUAN_API_KEY`
- Value: `SecretId:SecretKey`

> 腾讯混元[HunYuan](https://hunyuan.tencent.com/)大模型

| model                   | config                            | price                                 |
|-------------------------|-----------------------------------|---------------------------------------|
| `hunyuan-lite`          | HunYuanLiteGenerateConfig         | [price](https://hunyuan.tencent.com/) |
| `hunyuan-standard`      | HunYuanStandardGenerateConfig     | [price](https://hunyuan.tencent.com/) |
| `hunyuan-standard-256k` | HunYuanStandard256KGenerateConfig | [price](https://hunyuan.tencent.com/) |
| `hunyuan-pro`           | HunYuanProGenerateConfig          | [price](https://hunyuan.tencent.com/) |

> 调用示例

```python
from zlai.llms import HunYuan
from zlai.llms.generate_config.hunyuan import *

models = [
    HunYuanLiteGenerateConfig,
    HunYuanStandardGenerateConfig,
    HunYuanStandard256KGenerateConfig,
    HunYuanProGenerateConfig,
]

for gen_config in models:
    llm = HunYuan(generate_config=gen_config())
    data = llm.generate(query="1+1=")
    print(f"{gen_config.__name__.replace('GenerateConfig', '')}: {data.choices[0].message.content}")
    print()
```

*输出*

```text
HunYuanLite: 1加1等于1+1 = 2。
HunYuanStandard: 在数学中，加法是一种基本的算术运算。当我们进行两个数的加法时，我们实际上是在计算这两个数相加的和。在这个例子中，我们将数字1与另一个数字1相加。根据加法的定义，1 + 1 的结果是2。所以，1 + 1 = 2。
HunYuanStandard256K: 计算结果为:1 + 1 = 2
HunYuanPro: 这是一个非常基础的数学问题，涉及到加法运算。在这个问题中，我们需要将两个相同的数相加，这两个数都是1。\n\n当我们进行加法运算时，我们简单地将这两个数相加。在这种情况下，1加1等于2。\n\n所以，1+1的答案是2。
```

## Yi

> 配置AP-Key

- API Key Name: `YI_API_KEY`
- Value: `API-Key`

> 零一万物[Yi大模型](https://platform.lingyiwanwu.com/docs)

| model              | config                       | price                                          |
|--------------------|------------------------------|------------------------------------------------|
| `yi-large`         | YiLargeGenerateConfig        | [price](https://platform.lingyiwanwu.com/docs) |
| `yi-medium`        | YiMediumGenerateConfig       | [price](https://platform.lingyiwanwu.com/docs) |
| `yi-medium-200k`   | YiMedium200KGenerateConfig   | [price](https://platform.lingyiwanwu.com/docs) |
| `yi-spark`         | YiSparkGenerateConfig        | [price](https://platform.lingyiwanwu.com/docs) |
| `yi-large-rag`     | YiLargeRAGGenerateConfig     | [price](https://platform.lingyiwanwu.com/docs) |
| `yi-large-turbo`   | YiLargeTurboGenerateConfig   | [price](https://platform.lingyiwanwu.com/docs) |
| `yi-large-preview` | YiLargePreviewGenerateConfig | [price](https://platform.lingyiwanwu.com/docs) |

> 调用示例

```python
from zlai.llms import Yi
from zlai.llms.generate_config.yi import *

models = [
    YiLargeGenerateConfig,
    YiMediumGenerateConfig,
    YiMedium200KGenerateConfig,
    YiSparkGenerateConfig,
    YiLargeRAGGenerateConfig,
    YiLargeTurboGenerateConfig,
    YiLargePreviewGenerateConfig,
]

for gen_config in models:
    llm = Yi(generate_config=gen_config())
    data = llm.generate(query="1+1=")
    print(f"{gen_config.__name__.replace('GenerateConfig', '')}: {data.choices[0].message.content}")
    print()
```

*输出*

```text
YiLarge: 1 + 1 = 2
YiMedium: 1+1 equals 2.
YiMedium200K: I'm sorry, but I can only provide assistance and information. It seems like you might have encountered an error or a glitch in your input. If you need help with something specific, please let me know how I can assist you!
YiSpark: 1+1 equals 2.
YiLargeRAG: 1 + 1 = 2
YiLargeTurbo: 1 + 1 = 2
YiLargePreview: 1 + 1 = 2<issue_start>
```

## MoonShot

> 配置API-Key

- API Key Name: `MOONSHOT_API_KEY`
- Value: `API-Key`

> 月之暗面[MoonShot大模型](https://platform.moonshot.cn/docs)

| model              | config                  | price                                                    |
|--------------------|-------------------------|----------------------------------------------------------|
| `moonshot-v1-8k`   | SparkLiteGenerateConfig | [price](https://platform.moonshot.cn/docs/price/pricing) |
| `moonshot-v1-32k`  | SparkLiteGenerateConfig | [price](https://platform.moonshot.cn/docs/price/pricing) |
| `moonshot-v1-128k` | SparkLiteGenerateConfig | [price](https://platform.moonshot.cn/docs/price/pricing) |

> 调用示例

```python
from zlai.llms import MoonShot
from zlai.llms.generate_config.moonshot import *

models = [
    MoonShot8KV1GenerateConfig,
    MoonShot32KV1GenerateConfig,
    MoonShot128KV1GenerateConfig,
]

for gen_config in models:
    llm = MoonShot(generate_config=gen_config())
    data = llm.generate(query="1+1=")
    print(f"{gen_config.__name__.replace('GenerateConfig', '')}: {data.choices[0].message.content}")
    print()
```

*输出*

```text
MoonShot8KV1: 1+1等于2。这是一个基本的数学加法运算。
MoonShot32KV1: 1+1 equals 2. This is a basic arithmetic operation where you add the two numbers together.
MoonShot128KV1: 1+1等于2。这是基本的数学加法运算。
```

## End

-----
@2024/07/01

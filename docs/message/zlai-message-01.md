# Message

> 在许多问答应用程序中，我们希望大模型可以与用户进行多轮的问答，这意味着应用程序需要某种“记忆”，记住过去的问题和答案以及一些逻辑。此外，在另一些场景中，我们希望通过模型对过去的记忆总结回答当前的问题。在`zlai`中有是否方便的消息(`Message`)机制可供选择。

## 相关概念

- **Prompt**：提示词，是生成模型在生成内容时所遵循的规则和模式。
- **Message**: 消息，是生成模型接收的输入与输出。
- **Few-shot**: 少样本学习，是指在模型推理时使用少量的样本作为示例。

> `Message`类型

| Message                | 类别   | 备注 |
|------------------------|------|----|
| `SystemMessage()`      | 系统消息 |    |
| `UserMessage()`        | 用户消息 |    |
| `AssistantMessage()`   | AI消息 |    |
| `ObservationMessage()` | 观测消息 |    |

> 调用方式

```python
from zlai.schema import (
    Message,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ObservationMessage,
)
```

您可以使用上面的`Message`类来创建消息，下面将简单的介绍`Message`将如何使用。

### `SystemMessage`

> `SystemMessage`用于指定大模型的角色，`role="system"`已经被内置，你只需要指定该类的`content`即可。下面是一个示例：

```python
from zlai.schema import SystemMessage

system_message = SystemMessage(content="you are a helpful assistant.")
print(system_message)
```

*打印输出*

```text
role='system' content='you are a helpful assistant.'
```

> `SystemMessage`的作用

`SystemMessage`的作用是告诉大模型，你是一个什么角色，以及你对他的要求，这样大模型就可以根据你的角色与要求进行相应的处理。下面是一个示例，比如我想让大模型当做一个计算器：

**不加`SystemMessage`**

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo

llm = Zhipu(generate_config=ZhipuGLM3Turbo())
completion = llm.generate(query="76-8=")

print(completion.choices[0].message.content)
```

*打印输出*

```text
76-8 的计算结果是 68。
```

**加入`SystemMessage`**

```python
from zlai.schema import SystemMessage, UserMessage
from zlai.llms import Zhipu, ZhipuGLM3Turbo

messages = [
    SystemMessage(content="你是一个计算器。请直接输出计算后的结果，不要输出其他内容。"),
]
messages.append(UserMessage(content="76-8="))

llm = Zhipu(generate_config=ZhipuGLM3Turbo())
completion = llm.generate(messages=messages)
print(completion.choices[0].message.content)
```

*打印输出*

```text
68
```

可以从上面的实验中看出，在不使用`SystemMessage`的情况下，大模型会根据问题进行回答，结果正确但结果并不接简洁。而加入`SystemMessage`的提示之后，会根据`SystemMessage`的要求进行相应的处理，从而得到我们期望的输出结果。

### `UserMessage`

> `UserMessage`用于指定用户提出的问题，`role="user"`已经被内置，你只需要指定该类的`content`即可。下面是一个示例：

```python
from zlai.schema import UserMessage

user_message = UserMessage(content="76-8=")
print(user_message)
```

*打印输出*

```text
role='user' content='76-8='
```

### `AssistantMessage`

> `AssistantMessage`用于指定用户提出的问题，`role="assistant"`已经被内置，你只需要指定该类的`content`即可。下面是一个示例：

```python
from zlai.schema import AssistantMessage

assistant_message = AssistantMessage(content="76-8 的计算结果是 68。")
print(assistant_message)
```

*打印输出*

```text
content='76-8 的计算结果是 68。', role='assistant'
```

## `Message`用于管理大模型的记忆

这个例子用于介绍`Message`用于管理大模型的记忆。下面的例子中，我们将提供给模型多组账单信息，模型将根据这些信息进行推理，并给出相应的回答。

> 示例代码

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo
from zlai.schema import SystemMessage, UserMessage, AssistantMessage

messages = [
    SystemMessage(content="你是一个记账机器人。"),
    UserMessage(content="""时间：2024年4月15日\\n地点：超级市场\n金额：¥250.00\n事项：杂货购物"""),
    AssistantMessage(content="好的，已记录。"),
    UserMessage(content="""时间：2024年4月18日\n地点：餐厅A\n金额：¥80.50\n事项：晚餐"""),
    AssistantMessage(content="好的，已记录。"),
    UserMessage(content="""时间：2024年4月20日\n地点：加油站B\n金额：¥45.20\n事项：汽油"""),
    AssistantMessage(content="好的，已记录。"),
    UserMessage(content="""时间：2024年4月22日\n地点：电影院C\n金额：¥120.00\n事项：电影票"""),
    AssistantMessage(content="好的，已记录。"),
    UserMessage(content="""时间：2024年4月25日\n地点：购物中心D\n金额：¥350.00\n事项：衣物购物"""),
    AssistantMessage(content="好的，已记录。"),
]

messages.append(UserMessage(content="请告诉我最近我看电影花了多少钱？"))

llm = Zhipu(generate_config=ZhipuGLM3Turbo())
completion = llm.generate(messages=messages)

print(completion.choices[0].message.content)
```

> 模型输出

```text
您最近在电影院C购买电影票的费用是¥120.00。
```

*看，模型正确的找出了相关的费用信息，您可以询问其他已经记录的内容。*

## `Message`用于`Few-shot`

> 什么是`Few-shot`?

`Few-shot learning`（少样本学习）是一种机器学习的方法，旨在通过使用非常有限数量的样本来训练模型，以便在面对新的、相似但未见过的任务时能够进行有效的泛化。在某些情况下直接让大模型回答问题可能难以得到相对正确的结果，但给定几个样例即`few-shot`之后，往往会起到非常令人惊喜的效果。

假设需要让大模型提取出下面语句中的农作物，并以`List`表示。

```text
样本1: "我们种植了小麦和玉米。"
样本2: "农田里有大豆和稻谷。"
样本3: "我们正在研究水稻和小麦的生长条件。"
样本4: "这个地区主要种植的作物有玉米和大豆。"
样本5: "农民们关注的是小麦和水稻的收成情况。"
样本6: "我们进行了小麦、玉米和大豆的种植实验。"
样本7: "稻田里有稻谷和水稻苗。"
样本8: "我正在研究大豆和小麦的生长周期。"
样本9: "这片土地适合种植玉米和稻谷。"
```

### 直接问答的方式

我们看一下，直接让大模型回答问题，会得到什么结果。

> 示例代码

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo
from zlai.schema import SystemMessage, UserMessage

messages = [
    SystemMessage(content="你是一个实体识别机器人，你需要提取后续提供给你文本中涉及到的作物，并输出为List，注意仅输出List信息，不输出其他内容。"),
    UserMessage(content="农民们关注的是小麦和水稻的收成情况。"),
]

llm = Zhipu(generate_config=ZhipuGLM3Turbo())
completion = llm.generate(messages=messages)

print(completion.choices[0].message.content)
```

> 模型输出

我们运行多次上面的代码，看生成的结果。

```text
第1次："小麦, 水稻"
第2次："```python\ncrops = ['小麦', '水稻']\n```"
```

可以发现，第一次输出的内容并不是标准的List(我们期望输出`['小麦', '水稻']`)，第二次输出的像是Python代码，但并不是我们期望的。

### `Few-shot`的方式

> 示例代码

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo
from zlai.schema import SystemMessage, UserMessage, AssistantMessage

messages = [
    SystemMessage(content="你是一个实体识别机器人，你需要提取后续提供给你文本中涉及到的作物，并输出为List，注意仅输出List信息，不输出其他内容。")
]

few_shot = [    
    UserMessage(content="我们种植了小麦和玉米。"),
    AssistantMessage(content=str(["小麦", "玉米"])),
    UserMessage(content="农田里有大豆和稻谷。"),
    AssistantMessage(content=str(["大豆", "稻谷"])),
    UserMessage(content="我们正在研究水稻和小麦的生长条件。"),
    AssistantMessage(content=str(["水稻", "小麦"]))
]

messages.extend(few_shot)
messages.append(UserMessage(content="农民们关注的是小麦和水稻的收成情况。"))

llm = Zhipu(generate_config=ZhipuGLM3Turbo())
completion = llm.generate(messages=messages)

print(completion.choices[0].message.content)
```

> 模型输出

```text
['小麦', '水稻']
```

运行多次后我们发现，模型始终输出我们期望的结果。上面例子中，我们通过`few-shot`的方式，让模型先学习一些样本，然后通过这些样本来推理，从而得到我们期望的结果。

## `MessagePrompt`

> 什么是`MessagePrompt`?

`MessagePrompt`是`zlai`中更为方便的消息管理机制。正如上面的例子中，即使简单的例子中也涉及到了多种的消息`Message`机制，如果在实际使用中，需要编写大量的代码来管理这些消息，这显然是不方便的。因此，`zlai`中提供了`MessagePrompt`来管理这些消息。

> `MessagePrompt`的结构逻辑

下面的图片展示了`MessagePrompt`的结构逻辑。

<center>
<img src="img/zlai-message-01.png" width="680px">
<h6>图：MessagePrompt</h6>
</center>

简单来说，`MessagePrompt`集成了以下信息：

1. 系统消息: 即`SystemMessage`。
2. `Few-shot`: 即用户给定的样本样例。
3. 问题模板: 有时我们希望问题能够封装至固定的问题模板中，以便让大模型更清晰的识别哪些文本时给定的`参考文本`哪些文本时`用户问题`哪些是`需要特殊注意的消息`。这里我们在`zlai`中封装了`langchain`的`PromptTemplate`。
4. 用户问题：即用户提出的最新的问题。
5. 向量化模型: 即`Embedding`，用于对`few-shot`进行重排。向量化模型[参考](/doc/zlai-embedding-01.md)
6. 其他参数: 如指定是否进行重排，指定几个`few-shot`样例等。

> `few-shot`重排: 由于自回归模型是依据上下文进行下一个词的预测，文本的长度是十分关键的，模型对于较远的文本记忆并不深刻，我们希望在`few-shot`中，`few-shot`样例始终能够根据用户给定的问题进行相似度倒序排列，使得与最新问题最相似的样本能够展示在与最新问题更为接近的位置中，这样模型就可以更容易的抓取到相关的处理逻辑。

**详细参数说明**

- `system_message[SystemMessage]`: 系统消息`SystemMessage`。
- `few_shot[List[Message]]`: `Few-shot`样例。
- `n_shot[int]`: 相似度最高的 `n-shot`，`Few-shot`中可能有多个`n-shot`，这里指定的是需要给到大模型学习的`n-shot`数量。
- `rerank[bool]`: 是否进行rerank，即是否进行`Few-shot`的重排。
- `embedding[Embedding]`: 进行rerank排序的向量化模型。
- `verbose[bool]`: 是否显示中间过程。

> 使用示例

下面是一个简单的`MessagePrompt`示例。我们建设让大模型充当一个计算器，让他输出计算结果。

```python
from zlai.prompt import MessagesPrompt
from zlai.embedding import Embedding
from zlai.schema import *

# 设定系统Message
system_message = SystemMessage(content="你是一个计算器")

# 设定Few-shot
few_shot = [
    UserMessage(content="1+1="),
    AssistantMessage(content="2"),
    UserMessage(content="1+2="),
    AssistantMessage(content="3"),
    UserMessage(content="1+3="),
    AssistantMessage(content="4"),
]

# 设定Embedding
embedding = Embedding(model_path="/home/models/BAAI/bge-m3/")

# 创建 messages prompt
messages_prompt = MessagesPrompt(
    system_message=system_message,
    few_shot=few_shot,
    n_shot=2,                    # 相似度最高的 n-shot
    rerank=True,                 # 是否进行rerank
    embedding=embedding,         # 指定rerank排序模型
    verbose=True,                # 是否显示中间过程
)

messages = messages_prompt.prompt_format(content="1+1=")

print(messages)
```

> 打印输出内容

```text
[SystemMessage(role='system', content='你是一个计算器'),
 UserMessage(role='user', content='1+2='),
 AssistantMessage(role='assistant', content='3'),
 UserMessage(role='user', content='1+1='),
 AssistantMessage(role='assistant', content='2'),
 UserMessage(role='user', content='1+1=')]
```

在上面的示例中，我们首先定义了系统消息对象`system_message`，内容为"你是一个计算器"，表示系统的角色是一个计算器。然后定义了`few_shot`包含几个用户消息和助手回复的列表，用于在对话开始时提供一些示例对话。`embedding`是一个嵌入模型对象，用于将文本转换为向量表示，这里我使用了[`beg-m3`](https://huggingface.co/BAAI/bge-m3)模型。接着，创建了一个 `MessagesPrompt` 对象，指定了`n_shot=2`表示，我们只关心与当前问题最相似的2个样例。

最后，代码调用了 `messages_prompt.prompt_format(content="1+1=")` 方法，生成了针对`1+1=`的消息内容。可以观察到，第一条信息是系统信息，接着是两条问答对，最后是用户最新的问题，其中与`1+1=`最相似的内容排在了最新问题最近的位置，这样模型可以更容易得捕捉样例信息。

----
@2024/04/28

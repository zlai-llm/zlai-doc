# 大模型调用其他方法

> 这里展示的模型方法适用于`zlai`封装的全部在线大模型或本地大模型，但由于篇幅所限这里仅以`智谱AI`的`ZhipuGLM3Turbo`作为示例。

## 模型输出数据类型

上一节已经简单介绍了模型的回答输出数据类型，这里再简单总结一下。

### 全量`Completion`

> 默认情况下，全量输出`completion`信息

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo

llm = Zhipu(generate_config=ZhipuGLM3Turbo())
completion = llm.generate(query="你好")
print(completion)
```

> 输出`Completion`结果

```text
model='glm-3-turbo' created=1714271348 choices=[CompletionChoice(index=0, finish_reason='stop', message=CompletionMessage(content='你好👋！我是人工智能助手智谱清言（ChatGLM），很高兴见到你，欢迎问我任何问题。', role='assistant', tool_calls=None))] request_id='8605594695711311367' id='8605594695711311367' usage=CompletionUsage(prompt_tokens=6, completion_tokens=30, total_tokens=36)
```

### 返回消息`Message`

> 指定`output="message"`，返回`Message`消息

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo

llm = Zhipu(
    output="message",                # 以message消息形式输出
    generate_config=ZhipuGLM3Turbo()
)
message = llm.generate(query="你好")
print(message)
```

> 输出结果

```text
content='你好👋！我是人工智能助手智谱清言（ChatGLM），很高兴见到你，欢迎问我任何问题。' role='assistant' tool_calls=None
```

**Tips**: *`tool_calls`是`function call`模式下模型调用信息，这里先不做介绍，在后续的`Agent`部分将会做详细的介绍。*

### 仅返回回答的文本信息

> 指定`output="str"`，返回回答的文本信息

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo

llm = Zhipu(
    output="str",                    # 以str形式输出
    generate_config=ZhipuGLM3Turbo()
)
content = llm.generate(query="你好")
print(content)
```

> 输出结果

```text
你好👋！我是人工智能助手智谱清言（ChatGLM），很高兴见到你，欢迎问我任何问题。
```

## 不同的模型回答生成方式`Generate`

> 本小节介绍`zlai`中不同的生成回答方式及其适用的场景，并展示如何使用这些方式。

### `Query`推理

`Query`推理的方式是直接指定一个`query`问题参数，然后模型会根据这个`query`生成回答。模型会在完成全部的模型推理后返回最终的回答。

> 代码示例

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo
# 创建大模型
llm = Zhipu(generate_config=ZhipuGLM3Turbo())
completion = llm.generate(query="你好")

print(completion.choices[0].message.content)
```

> 回答输出

```text
你好👋！我是人工智能助手智谱清言（ChatGLM），很高兴见到你，欢迎问我任何问题。
```

### `Message`推理

`Message`推理的方式是指定一个`Message`的`List`的形式(`List[Message]`)模型会根据多个消息（`Message`）进行问题的回答。模型会在完成全部的模型推理后返回最终的回答。

`Message`的一般包含两个内容`role`和`content`：

- 角色`role`: 对话的角色，一般包含用户角色`user`、AI角色`Assistant`、系统角色`System`。
  - 用户角色`user`: 指的是用户，一般由用户提出问题。
  - AI角色`Assistant`: 指的是AI助手，一般由AI助手回答问题。
  - 系统角色`System`: 指的是系统，一般是`List[Message]`的第一条信息是提供给大模型的系统消息内容。
- 内容`content`: 对话的内容，一般是`str`格式的文本数据。

> 如何指定`Message`？

在`zlai`中您可以方便的组织`Message`信息。下面是一个示例：

```python
from zlai.schema import SystemMessage, UserMessage, AssistantMessage

messages = [
    SystemMessage(content="你是一个人工智能助手"),
    UserMessage(content="你好"),
    AssistantMessage(content="你好！我是人工智能助手智谱清言（ChatGLM），很高兴见到你，欢迎问我任何问题。"),
    UserMessage(content="你叫什么名字？"),
    ...,
]
```

其中，

- `SystemMessage`: 用于指定系统消息。
- `UserMessage`: 用于指定用户消息。
- `AssistantMessage`: 用于指定AI助手消息。
- `Message`: 是以上三个消息的父类，您也可以使用`Message`来指定任意类型的消息。下面是个例子:

```python
from zlai.schema import Message

messages = [
    Message(role="system", content="你是一个人工智能助手"),
    Message(role="user", content="你好"),
    Message(role="assistant", content="你好！我是人工智能助手智谱清言（ChatGLM），很高兴见到你，欢迎问我任何问题。"),
    Message(role="user", content="你叫什么名字？"),
]
```

> 在构造了`Messages`后，如何进行`Message`推理？

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo
from zlai.schema import SystemMessage, UserMessage, AssistantMessage

# 创建大模型
llm = Zhipu(generate_config=ZhipuGLM3Turbo())

messages = [
    SystemMessage(content="你是一个人工智能助手"),
    UserMessage(content="你好"),
    AssistantMessage(content="你好！我是人工智能助手智谱清言（ChatGLM），很高兴见到你，欢迎问我任何问题。"),
    UserMessage(content="你叫什么名字？"),
]

completion = llm.generate(messages=messages)

print(completion.choices[0].message.content)
```

```text
我是一个人工智能助手，我的名字是智谱清言（ChatGLM）。
```

**Tips**: *我们推荐您更多的使用*Message*的推理方式，这可以为您方便的组织历史消息信息，构造更为灵活的问答机制。以上仅是一个简单的示例，在后面的介绍中我们将介绍更为复杂的`messages`的组织逻辑。*

### `Stream`推理

`stream`推理是一种逐块返回回答内容的方式，能够实现类似打字机的效果，构建聊天程序的时候常用，能够给使用者更为快捷的消息反馈，有更好的使用体验。

在实际使用时，您只需要在模型配置类中指定`stream=True`即可，其他参数与上文的内容并无区别。

> 使用示例

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo
from zlai.schema import UserMessage

# 创建模型推理配置，并指定stream=True
generate_config = ZhipuGLM3Turbo(stream=True)

# 创建大模型
llm = Zhipu(generate_config=generate_config)
messages = [UserMessage(content="你好")]
responses = llm.generate(messages=messages)

answer = ''
for response in responses:
    answer += response.choices[0].delta.content
    print(answer, end='\n\n')
```

**Tips**: *在上面的代码中，我们将每次回答的输出`delta`拼接到`answer`回答中，并逐行进行了打印。下面是输出结果。*

> 回答输出

```text
你好

你好👋！

你好👋！我是

你好👋！我是人工智能

你好👋！我是人工智能助手

你好👋！我是人工智能助手智

你好👋！我是人工智能助手智谱

你好👋！我是人工智能助手智谱清

你好👋！我是人工智能助手智谱清言

你好👋！我是人工智能助手智谱清言（

你好👋！我是人工智能助手智谱清言（C

你好👋！我是人工智能助手智谱清言（Chat

你好👋！我是人工智能助手智谱清言（ChatGL

你好👋！我是人工智能助手智谱清言（ChatGLM

你好👋！我是人工智能助手智谱清言（ChatGLM），

你好👋！我是人工智能助手智谱清言（ChatGLM），很高兴

你好👋！我是人工智能助手智谱清言（ChatGLM），很高兴见到

你好👋！我是人工智能助手智谱清言（ChatGLM），很高兴见到你

你好👋！我是人工智能助手智谱清言（ChatGLM），很高兴见到你，

你好👋！我是人工智能助手智谱清言（ChatGLM），很高兴见到你，欢迎

你好👋！我是人工智能助手智谱清言（ChatGLM），很高兴见到你，欢迎问我

你好👋！我是人工智能助手智谱清言（ChatGLM），很高兴见到你，欢迎问我任何

你好👋！我是人工智能助手智谱清言（ChatGLM），很高兴见到你，欢迎问我任何问题

你好👋！我是人工智能助手智谱清言（ChatGLM），很高兴见到你，欢迎问我任何问题。

你好👋！我是人工智能助手智谱清言（ChatGLM），很高兴见到你，欢迎问我任何问题。
```



----

## 输出结果的解析

为了方便输出内容的结构化，`zlai`封装的大模型都支持对模型输出的结果进行解析，解析的结果可以以`dict/list/script`的形式返回。这样的设计可以为后续的文本数据解析以及从回答内容中解析出程序代码`script`提供方便。

使用`generate_with_parse`方法，可以实现对于模型输出结果的解析。

> `generate_with_parse`参数说明

| 参数名          | 参数类型          | 参数说明                                                                                                                               |
|--------------|---------------|------------------------------------------------------------------------------------------------------------------------------------|
| `query`      | str           | 需要大模型回答的问题，与`messages`二选一给定即可                                                                                                      |
| `messages`   | List[Message] | 消息，与`query`二选一给定即可                                                                                                                 |
| `parse_fun`  | Callable      | 自定义的解析函数，与下面`parse_dict`二选一给定即可                                                                                                    |
| `parse_dict` | Literal[str]  | 枚举值，有三个值可以给定`["eval", "greedy", "nested"]`, `eval`适用于输出内容中只有一个`Dict`的情况；`greedy`适用于输出内容中有多个`Dict`的情况；`nested`适用于输出内容中有嵌套`Dict`的情况； |

> 使用方式-1: 给定`parse_dict`

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo

llm = Zhipu(generate_config=ZhipuGLM3Turbo())

question = """
文本：张三在杭州吃了一笼小笼包。
问题：请解析出文本中的人名、地点，以Dict的格式输出。
格式示例：{'name': ..., 'place': ...,}
你只需要输出解析后的Dict，不需要输出其他内容。"""

output = llm.generate_with_parse(query=question, parse_dict='eval',)

print(llm.parse_info)
print(f"解析后数据类型: {type(output[0])}")
print(f"解析结果: {output[0]}")
```

> 输出结果

```text
[ParseInfo(content="{'name': '张三', 'place': '杭州'}", parsed_data=[{'name': '张三', 'place': '杭州'}], error_message=None)]
解析后数据类型: <class 'dict'>
解析结果: {'name': '张三', 'place': '杭州'}
```

其中，`llm.parse_info`中存储了解析的历史信息，`content`是模型回答输出的内容数据类型为`str`，`parsed_data`是解析后的数据，`error_message`是解析错误的信息，如果解析正确则为`None`。

> 使用方式-2: 自定义解析函数。

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo

def udf_parse(string):
  """自定义解析函数"""
  return eval(string)

llm = Zhipu(generate_config=ZhipuGLM3Turbo())

question = """
文本：张三在杭州吃了一笼小笼包。
问题：请解析出文本中的人名、地点，以Dict的格式输出。
格式示例：{'name': ..., 'place': ...,}
你只需要输出解析后的Dict，不需要输出其他内容。"""

output = llm.generate_with_parse(query=question, parse_fun=udf_parse,)

print(llm.parse_info)
print(f"解析后数据类型: {type(output[0])}")
print(f"解析结果: {output[0]}")
```

> 输出结果

```text
[ParseInfo(content="{'name': '张三', 'place': '杭州'}", parsed_data=[{'name': '张三', 'place': '杭州'}], error_message=None)]
解析后数据类型: <class 'dict'>
解析结果: {'name': '张三', 'place': '杭州'}
```

**Tips-1**: *`stream=True`的流式输出模式下禁用了结果解析*
**Tips-2**: *其他解析方式请参考`parse`相关文档内容*

## 异步生成回答结果

在数据量较多的时候，也许我们希望可以分布式的调用模型，让模型同时可以处理多条问答。

### 异步调用`async_generate`

> 特殊参数：

- `async_max_request_time`: 异步调用时`async_generate`会每隔`1s`查询回答是否完成，默认为`600`s，如果回答超时，则返回`None`。您可以根据自己的实际情况设定超时时间。

> 调用示例

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo

llm = Zhipu(generate_config=ZhipuGLM3Turbo())

question_1 = """
文本：张三在杭州吃了一笼小笼包。
问题：请解析出文本中的人名、地点，以Dict的格式输出。
格式示例：{'name': ..., 'place': ...,}
你只需要输出解析后的Dict，不需要输出其他内容。"""

question_2 = """
文本：李四在上海吃了一笼小煎包。
问题：请解析出文本中的人名、地点，以Dict的格式输出。
格式示例：{'name': ..., 'place': ...,}
你只需要输出解析后的Dict，不需要输出其他内容。"""

output = llm.async_generate(query_list=[question_1, question_2])

for completion in output:
    content = completion.choices[0].message.content
    print(type(content), content)
```

> 输出结果

```text
<class 'str'> {'name': '张三', 'place': '杭州'}
<class 'str'> {'name': '李四', 'place': '上海'}
```

### 异步调用并解析输出内容`async_generate_with_parse`

> 使用方式-1: 给定`parse_dict`

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo

llm = Zhipu(generate_config=ZhipuGLM3Turbo())

question_1 = """
文本：张三在杭州吃了一笼小笼包。
问题：请解析出文本中的人名、地点，以Dict的格式输出。
格式示例：{'name': ..., 'place': ...,}
你只需要输出解析后的Dict，不需要输出其他内容。"""

question_2 = """
文本：李四在上海吃了一笼小煎包。
问题：请解析出文本中的人名、地点，以Dict的格式输出。
格式示例：{'name': ..., 'place': ...,}
你只需要输出解析后的Dict，不需要输出其他内容。"""

outputs = llm.async_generate_with_parse(
    query_list=[question_1, question_2],
    parse_dict="eval",
)

for output in outputs:
    print(type(output), output)
```

> 输出结果

```text
<class 'list'> [{'name': '张三', 'place': '杭州'}]
<class 'list'> [{'name': '李四', 'place': '上海'}]
```

> 使用方式-2: 自定义解析函数。

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo

def udf_parse(string):
  """自定义解析函数"""
  return eval(string)

llm = Zhipu(generate_config=ZhipuGLM3Turbo())

question_1 = """
文本：张三在杭州吃了一笼小笼包。
问题：请解析出文本中的人名、地点，以Dict的格式输出。
格式示例：{'name': ..., 'place': ...,}
你只需要输出解析后的Dict，不需要输出其他内容。"""

question_2 = """
文本：李四在上海吃了一笼小煎包。
问题：请解析出文本中的人名、地点，以Dict的格式输出。
格式示例：{'name': ..., 'place': ...,}
你只需要输出解析后的Dict，不需要输出其他内容。"""

outputs = llm.async_generate_with_parse(
    query_list=[question_1, question_2],
    parse_fun=udf_parse,
)

for output in outputs:
    print(type(output), output)
```

> 输出结果

```text
[ParseInfo(content="{'name': '张三', 'place': '杭州'}", parsed_data=[{'name': '张三', 'place': '杭州'}], error_message=None)]
解析后数据类型: <class 'dict'>
解析结果: {'name': '张三', 'place': '杭州'}
```

**Tips**: *目前，仅`智谱AI`支持异步调用模式。*

## 4. 其他

**注意：**

1. 在做解析类任务时，`top_p`和`temperature`可以设置相对较低的值，以减少生成结果的随机性。
2. 将设置`do_sample=True`，以获取可重复的结果。在设置`do_sample=False`后，`top_p`和`temperature`将不起作用。

# End

@ 2024/04/28

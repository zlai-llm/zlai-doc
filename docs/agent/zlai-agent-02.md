# 天气查询

> 在之前的介绍中，我们详细介绍了`zlai-agent`的实现原理，并用天气查询这个例子做了详细的说明，这里我们将再次简单介绍一下这个例子，更多的是补充一些前文未关注到的细节。

## AddressAgent

> 这是一个地址解析Agent，从文本中提取出地址信息，并将其转换为标准行政区划。

这是一个示例：

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo
from zlai.agent import AddressAgent

llm = Zhipu(generate_config=ZhipuGLM3Turbo())
agent = AddressAgent(llm=llm, agent_name="Address Agent", verbose=True)
task_completion = agent("北京市朝阳区")
print(task_completion)
```

*日志信息*：

```text
[Address Agent] User Question: 北京市朝阳区
==================== [Address Agent] Messages Start ====================

user [6]: [北京市朝阳区]

==================== [Address Agent] Messages End    ====================

[Address Agent] Final Answer:
{'province': '北京市', 'city': '北京市', 'district': '朝阳区'}
[Address Agent] Parsed address:
[{'province': '北京市', 'city': '北京市', 'district': '朝阳区'}]
```

*最终推理结果`task_completion`信息*：

```text
TaskCompletion(
    query='北京市朝阳区', 
    content="{'province': '北京市', 'city': '北京市', 'district': '朝阳区'}", 
    parsed_data=StandardAddress(province='北京市', city='北京市', district='朝阳区'), 
)
```

**可以看到，上面的`AddressAgent`将地址信息解析为省市县三个级别，在`TaskCompletion`中记录了`用户问题`，大模型的回答`content`，以及解析后的数据`parsed_data`。**

## 消息模板`message prompt`

关于消息机制你可以在[这里](message/zlai-message-01)找到详细的介绍，在绝大多数的`Agent`场景中我们都可以设置不同的消息机制来让大模型完成不同的任务。比如，下面是`AddressAgent`中所用到的消息机制：

```python
from zlai.prompt import MessagesPrompt
from zlai.schema import SystemMessage, UserMessage, AssistantMessage

system_message = SystemMessage(content="""你是一个地址命名实体识别机器人，你需要解析出文本中出现的省、市、区，并以Dict输出。\
文本中没有省、市、区，则返回空Dict。""")

few_shot = [
    UserMessage(content="上海市浦东新区张江高科技园区"),
    AssistantMessage(content=str({'province': '上海市', 'city': '上海市', 'district': '浦东新区'})),
    UserMessage(content="广东省深圳市南山区深南大道1001号"),
    AssistantMessage(content=str({'province': '广东省', 'city': '深圳市', 'district': '南山区'})),
    UserMessage(content="河北省衡水市景县的天气情况"),
    AssistantMessage(content=str({'province': '河北省', 'city': '衡水市', 'district': '景县'})),
    UserMessage(content="你好呀"),
    AssistantMessage(content=str({})),
]

messages_prompt: MessagesPrompt = MessagesPrompt(
    system_message=system_message,
    few_shot=few_shot,
    n_shot=5,
)

query = "北京市朝阳区"
messages = messages_prompt.prompt_format(content=query)
print(messages)
```

*打印输出：*

```text
[
    SystemMessage(role='system', content='你是一个地址命名实体识别机器人，你需要解析出文本中出现的省、市、区，并以Dict输出。文本中没有省、市、区，则返回空Dict。'), 
    UserMessage(role='user', content='上海市浦东新区张江高科技园区'), 
    AssistantMessage(role='assistant', content="{'province': '上海市', 'city': '上海市', 'district': '浦东新区'}"), 
    UserMessage(role='user', content='广东省深圳市南山区深南大道1001号'), 
    AssistantMessage(role='assistant', content="{'province': '广东省', 'city': '深圳市', 'district': '南山区'}"), 
    UserMessage(role='user', content='河北省衡水市景县的天气情况'), 
    AssistantMessage(role='assistant', content="{'province': '河北省', 'city': '衡水市', 'district': '景县'}"), 
    UserMessage(role='user', content='你好呀'),
    AssistantMessage(role='assistant', content='{}'), 
    UserMessage(role='user', content='北京市朝阳区')
]
```

**通过制定消息模版的方式，可以让大模型知道当前你需要他帮助你完成怎样的任务，通过给定`few-shot`可以让大模型了解你期望它输出的格式与内容。通常情况下给定`5-shot`会比不给定样例有更稳定的输出，但给定过多的样例也会占用过多的推理`tokens`的消耗，你需要根据自己实际的任务合理指定模型应该了解哪些样例。**

**此外，你也可以[参考](message/zlai-message-01?id=messageprompt)对`MessagesPrompt`增加`few-shot`的重排功能，每次选出与当前问题最为接近的几个样例供大模型进行参考。这些细节的调整都会改善大模型输出结果的稳定性与准确性。**

## WeatherAgent

> `WeatherAgent`会接受标准的上文中提到的`AddressAgent`输出的`TaskCompletion`格式的结果作为输入（其实就是标准的地址格式输入），并输出一个`TaskCompletion`格式的结果，即模型对[天气接口](agent/zlai-agent-02?id=参考)信息的总结回答。下面是示例：

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo
from zlai.agent import WeatherAgent

llm = Zhipu(generate_config=ZhipuGLM3Turbo())
agent = WeatherAgent(llm=llm, verbose=True)
task_completion = agent(query=task_completion)
print(task_completion)
```

*日志信息：*

```text
[Weather Agent] City Name: 朝阳区
==================== [Weather Agent] Messages Start ====================

user [189]: [请根据以下天气信息回答问题。

信息：{'current_condition': {'temp_C': '18', 'FeelsLikeC': '18', 'humidity': '60',...]

==================== [Weather Agent] Messages End    ====================

[Weather Agent] Final Answer:
当前北京市朝阳区的天气情况是：轻度降雨，温度为18摄氏度，体感温度也是18摄氏度，湿度为60%。这个信息是基于最近的观测时间，即上午9点40分。请注意随时关注实时天气变化，合理安排您的日常活动。
```

**Tips:**

* 前文所描述的[`TaskSequence`](agent/zlai-agent-01?id=顺序执行)是对于`AddressAgent`与`WeatherAgent`的集成。
* `zlai`中已经将`AddressAgent`与`WeatherAgent`集成，你只需要调用`Weather`即可。上面讲这么多只是为了解释工程流程。

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo
from zlai.agent import Weather

llm = Zhipu(generate_config=ZhipuGLM3Turbo())
agent = Weather(llm=llm, verbose=True)
task_completion = agent(query="XX天气怎么样？")
```

-----

## 参考链接

* [wttr天气接口](https://wttr.in/)
* [wttr-github](https://github.com/chubin/wttr.in)

@2024/05/23

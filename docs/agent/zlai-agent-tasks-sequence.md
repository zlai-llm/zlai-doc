# Agent Tasks Sequence

> `Agent`任务的序列执行，用于对多个任务的顺序执行，前一个任务的结果是后一个任务的输出。

## 序列任务执行

绝大部分的大模型应用场景都需要补充给大模型相关参考信息，`Tasks Sequence`提供了一种顺序
执行的模式，清晰切分任务的执行内容，尽可能使得模型执行简单的小任务，一步一步完成大任务，将
大大提升大模型的执行效率与应用的稳定性。

<center>
<img src="./img/zlai-tasks-sequence.png" width="880px">
<h5>Fig. Tasks Sequence</h5>
</center>

**思考：如果构建一个天气查询播报机器人。**

这里可以将`天气查询播报机器人`拆分为两个主要的`Agent`动作。

1. 查询天气：绝大多数的天气查询接口，只支持标准的行政区划地址，然而用户问题一般不是标准地址名称。
2. 播报天气：需要查询相关的天气信息。

<center>
<img src="./img/zlai-tasks-weather.png" width="780px">
<h5>Fig. Tasks Weather</h5>
</center>

## `Tasks Sequence`示例：天气查询播报

> step 1: `Agent`配置`List`

```python
from zlai.llms import *
from zlai.agent import *
from zlai.agent.prompt.tasks import *  # 0.3.45 后的版本不需要引入

task_weather = [
    TaskDescription(
        task=AddressAgent, task_id=0, task_name="地址解析机器人",
        task_description="""可以帮助你解析文本中的地址信息，并返回标准地址字段信息。""",
        task_parameters=TaskParameters(
            llm=Zhipu(generate_config=ZhipuGLM3Turbo()),
            verbose=True,
        )
    ),
    TaskDescription(
        task=WeatherAgent, task_id=1, task_name="天气播报机器人",
        task_description="""提供具体的地址信息后可以帮助你查询当地的天气情况，必须提供标准地址。""",
        task_parameters=TaskParameters(
            llm=Zhipu(generate_config=ZhipuGLM3Turbo()),
            verbose=True,
        )
    ),
]
```

> step 2: 构建`Task Sequence`

**Method-2**: *使用`TaskSequence`*

```python
from zlai.agent import *

task_seq = TaskSequence(task_list=task_weather, verbose=True)
```

**Method-2**: *继承`TaskSequence`*

```python
from typing import *
from zlai.llms import *
from zlai.agent import *
from zlai.agent.prompt.tasks import *  # 0.3.45 后的版本不需要引入

class Weather(TaskSequence):
    """"""
    llm: Optional[TypeLLM]
    task_list: Optional[List[TaskDescription]]

    def __init__(
            self,
            llm: Optional[TypeLLM] = None,
            task_list: Optional[List[TaskDescription]] = None,
            *args: Any,
            **kwargs: Any,
    ):
        super().__init__(llm=llm, task_list=task_list, *args, **kwargs)
        if task_list is None:
            self.task_list = task_weather

task_seq = Weather(task_list=task_weather, verbose=True)
```

> step 3: 执行任务

```python
query = "杭州今天天气怎么样？"
answer = task_seq(query=query)
```

*以下为开启了`verbose=True`后输出的执行信息，详细记录了消息流转状态。*

```text
[Task Sequence] Running 2 tasks...

[Task Sequence @ Address] Start ...
[Address] User Question: 余杭今天天气怎么样？
==================== [Address] Messages Start ====================
user: [余杭今天天气怎么样？]

==================== [Address] Messages End    ====================
[Address] Final Answer:
{'province': '浙江省', 'city': '杭州市', 'district': '余杭区'}
[Address] Parsed address:
[{'province': '浙江省', 'city': '杭州市', 'district': '余杭区'}]
[Task Sequence @ Address] End.

[Task Sequence @ Weather Agent] Start ...
[Weather Agent] City Name: 余杭区
==================== [Weather Agent] Messages Start ====================
user: [请根据以下天气信息回答问题。

信息：
"""
{'current_condition': {'temp_C': '24', 'FeelsLikeC': '26', 'humidity': '83', 'weatherDesc': [{'value': 'Partly cloudy'}], 'observation_time': '07:50 AM'}}
"""
问题：余杭今天天气怎么样？]

==================== [Weather Agent] Messages End    ====================

[Weather Agent] Final Answer:
余杭今天的天气情况是：部分多云，当前温度为24摄氏度，体感温度为26摄氏度，湿度为83%，观测时间为早上7点50分。
[Task Sequence @ Weather Agent] End.

Task Sequence End.
余杭今天的天气情况是：部分多云，当前温度为24摄氏度，体感温度为26摄氏度，湿度为83%，观测时间为早上7点50分。
```

-----
@2024/04/02
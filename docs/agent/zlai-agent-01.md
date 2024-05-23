# ZLAI-Agent架构介绍

> Agent是什么?

智能体（Intelligent Agent，IA）是计算机科学和人工智能领域中的一个关键概念，指的是一种能够感知其环境并采取行动以实现特定目标的自主实体。智能体可以应用在各种场景中，从简单的自动化任务到复杂的决策系统。借助大模型等技术，智能体可以实现更复杂的任务和决策。

> 智能体的基本特性

* **自主性（Autonomy）**： 智能体能够在很大程度上自主地运行，减少对人为干预的依赖。
* **感知（Perception）**： 智能体通过传感器（或输入接口）感知周围环境的信息。例如，机器人通过摄像头、麦克风等感知环境。
* **行动（Action）**： 智能体通过执行器（或输出接口）对环境进行操作，以实现目标。例如，机器人通过移动、抓取等动作与环境互动。
* **目标导向（Goal-oriented）**： 智能体被设计为完成特定任务或达到某些目标。它们通常具有内部的目标或任务规划机制。
* **反应性与主动性（Reactive and Proactive）**： 智能体能够对环境的变化做出快速反应（反应性），并且能够基于预期计划和目标主动采取行动（主动性）。

> 智能体的应用

智能体在各种领域都有广泛的应用，包括:

- **机器人技术**：智能体用于自主机器人，包括家庭机器人（如扫地机器人）、工业机器人（如自动化生产线机器人）和无人机。
- **自动驾驶**：自动驾驶汽车使用智能体技术感知环境、规划路径和控制车辆。
- **智能助理**：智能语音助理（如Siri、Alexa）是典型的智能体应用，能够理解用户指令并执行相应任务。
- **游戏AI**：在视频游戏中，智能体用于控制NPC，使其能够表现出智能行为和决策。
- **金融领域**：智能体用于自动交易系统，通过分析市场数据和趋势进行买卖决策。
- **医疗领域**：智能诊断系统通过分析患者数据辅助医生进行诊断和治疗决策。

## ZLAI-Agent架构思路

尽管目前有非常多种的Agent实现方式与路径，如：`FunctionCall`、思维树、思维链等，由于当前大模型对于任务推理的智能能力所限，这些方法往往在智能程度较高的模型上效果出众，但在一般的大模型上往往难以取得较好的效果，就像在`langchain`框架下对于`ChatGPT`上运行良好的示例，但在参数量较小的开源模型上，或在国内的某些大模型API上就难以取得让人满意的效果。`langchain`是一个极为优秀的开源Agent框架，但有时候我们难以直接切换到国内的大模型或开源模型上，以及我们希望更为自由的定义任务的流程、任务流程的prompt、任务流程的执行方式等等。这都需要一个更为灵活的Agent框架来支持。

因此，在 `ZLAI-Agent` 的实现中，我们思考是否有一种更为简单并且清晰的工程结构，能够使得不够智能的大模型也能够完成任务的拆分与自主的任务推理；当然，使用更为聪明的大模型能够给出更好的结果。在此基础上，我们思考这套架构有这样的优势或者能够带来什么样的价值：

* **更小的推理模型**: 使用小型的模型，或不那么智能的模型，可以使得相关开发与应用API-tokens消耗更为便宜，而且推理速度更快，切换为更为智能的模型后效果往往会更好。
* **工程结构清晰**: 在多轮的Agent任务流转中，我们希望能够清晰的监控到模型在处理什么事情，这样去保证Agent执行操作的可解释性、可靠性与安全性。
* **Agent节点的灵活拼接**: 在Agent任务执行过程中，我们希望Agent能够灵活的拼接，以适应不同的任务场景，并且能够适应不同的任务执行方式。
 
> 为此，我们没有采用`FunctionCall`的方式，仅依靠大模型自回归输出文本的能力，在 `zlai` 框架中将智能体任务（`Agent`）分为以下几种类型：

* **意图判断**: 给定大模型多个Agent选项，让大模型选择符合用户意图的Agent。
* **智能解析**: 发挥大模型对于文本内容的解析能力，如，用户要查询某地的天气，我们就需要让大模型解析出要查询的标准行政区划。
* **代码生成**: 用于生成多种代码，如`Python`、`SQL`、`JavaScript`、`C++`等。
* **执行观察**: 用于执行观察，如执行`Python`、`SQL`代码，并观察其执行结果。
* **总结回答**: 用于总结回答，根据历史消息，以及历史生成的脚本代码与执行日志总结回答最终的答案，如总结回答用户的问题。

在后面的讨论中，我们想详细讨论，以上概念的实现流程与相应的示例。

## Agent Tasks

<center>
<img src="./img/zlai-tasks-switch.png" width="780px">
<h5>Fig. Agent Tasks</h5>
</center>

> `ZLAI` 中重要的 Agent 任务概念






## Agent Tasks Sequence

> `Agent`任务的序列执行，用于对多个任务的顺序执行，前一个任务的结果是后一个任务的输出。

## 序列任务执行

绝大部分的大模型应用场景都需要补充给大模型相关参考信息，`Tasks Sequence`提供了一种顺序执行的模式，清晰切分任务的执行内容，尽可能使得模型执行简单的小任务，一步一步完成大任务，将大大提升大模型的执行效率与应用的稳定性。

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

# Task Switch

> 任务切换



-----
@2024/05/23

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

## Agent Task

<center>
<img src="./img/zlai-tasks-switch.png" width="780px">
<h5>Fig. Agent Tasks</h5>
</center>

在 `ZLAI-Agent` 中我们希望大模型总是执行一步最简单的操作以获取最稳定的回答，其中`Task`是重要的概念，`Task`定义了大模型处理任务的最细单元。一个次简单的聊天对话即可视为一个`Task`节点。

> `Task`是如何被定义的？

`TaskDescription`是`ZLAI-Agent`中定义`Task`的类，其定义了`Task`的名称、描述、参数、执行方式等。下面是一个最简单的`TaskDescription`示例：

```python
from zlai.agent import TaskDescription, ChatAgent

task_chat = TaskDescription(
    task=ChatAgent, task_name="聊天机器人",
    task_description="""提供普通对话聊天，不涉及专业知识与即时讯息。""",
)

print(task_chat)
```

*输出*: 

```text
TaskDescription(
    task=<class 'zlai.agent.chat.ChatAgent'>, 
    task_id=None, 
    task_name='聊天机器人', 
    task_description='提供普通对话聊天，不涉及专业知识与即时讯息。', 
    task_parameters=TaskParameters(
        llm=None, 
        embedding=None, 
        db=None, 
        db_path=None, 
        system_message=None, 
        system_template=None, 
        prompt_template=None, 
        few_shot=None, 
        messages_prompt=None, 
        logger=None, 
        verbose=None, 
        index_name=None, 
        elasticsearch_host=None, 
        kwargs=None,
    )
)
```

> `TaskDescription`的参数

- `task`: 任务`class`定义了任务执行的逻辑，在这个示例中，这个任务仅提供普通的对话聊天，在后续的其他`agent task`介绍中我们将介绍更为复杂的任务，如代码生成、代码执行、api调用等等。
- `task_id`: 任务的唯一ID标识，如果存在多种任务，大模型在意图识别阶段会依照此ID选择任务进行执行。
- `task_name`: 任务的名称，大模型会参考用户的问题、任务名称、任务描述选择相应的`task`。
- `task_description`: 任务的描述，大模型会参考用户的问题、任务名称、任务描述选择相应的`task`。

以上是`TaskDescription`的描述性参数，`task_parameters`是大模型执行这个任务需要的信息。

> `TaskParameters`的参数

- `llm`: 任务执行的大模型，可以是`ChatGPT`、`Zhipu`、`Baichuan`等。
- `embedding`: 任务执行的大模型需要使用的`embedding`模型，比如某些`RAG`任务会用到。
- `db`: 数据库链接，如果你需要大模型直接访问数据库，可以在这里定义数据库链接。
- `db_path`: 数据库文件路径，如果你需要大模型直接访问数据库，可以在这里定义数据库文件路径。
- `system_message`: 系统消息，任务执行的大模型需要使用的`system_message`。
- `system_template`: 对系统消息增加的消息模板。
- `prompt_template`: 对于当前用户对话或当前任务需要执行的消息模板。
- `few_shot`: 给定大模型的样例。
- `messages_prompt`: 任务执行的大模型需要使用的消息组织方式，如果你需要对`few-shot`进行重排，您可以考虑指定该参数。
- `logger`: 用于记录任务执行日志的`logger`。
- `verbose`: 布尔值，是否显示当前执行的任务的详细信息。
- `index_name`: 向量数据库的索引名称。
- `elasticsearch_host`: 任务执行的大模型需要使用的`elasticsearch_host`。
- `kwargs`: 其他参数。

下面是一个`TaskParameters`的示例：

```python
from zlai.schema import SystemMessage
from zlai.llms import Zhipu, ZhipuGLM3Turbo
from zlai.agent import TaskParameters

system_message = SystemMessage(content="""You are a helpful assistant.""")

task_parameters = TaskParameters(
    llm=Zhipu(generate_config=ZhipuGLM3Turbo()),
    system_message=system_message,
)

print(task_parameters)
```

*输出*: 

```text
TaskParameters(
    llm=<zlai.llms.zhipu.Zhipu object at 0x0000019AA6C169E0>, 
    embedding=None, 
    db=None, 
    db_path=None, 
    system_message=SystemMessage(
        role='system', 
        content='You are a helpful assistant.'
    ), 
    system_template=None, 
    prompt_template=None, 
    few_shot=None, 
    messages_prompt=None, 
    logger=None, 
    verbose=None, 
    index_name=None, 
    elasticsearch_host=None, 
    kwargs=None
)
```

*Tips: 您可以按照需要组织`TaskParameters`参数，在全部参数都不指定的情况下，框架会给定一般合适的值，保证任务的执行。*

## Task 组织方式

> 在`zlai`中有两种组织`Task`的方式，一种是顺序执行，按照给定的`Task`顺序依次执行，另一种是判断意图后依据给定的`Agent`有选择的执行任务。并且顺序执行与随机执行可以随意的依据实际情况进行嵌套组合，这样可以构建庞大的Agent网络，以短小的任务组织成庞大的Agent流程。

<center>
<img src="./img/tasks-01.png" width="880px">
<h5>Fig. 任务组织方式</h5>
</center>

### 顺序执行

> 顺序执行Agent任务(`Agent Tasks Sequence`)，顾名思义就是让大模型按照指定的顺序执行任务。`Agent`执行任务序列时，前一个任务的结果是后一个任务的输出。

<center>
<img src="./img/zlai-tasks-sequence.png" width="880px">
<h5>Fig. Tasks Sequence</h5>
</center>

绝大部分的大模型应用场景都需要补充给大模型相关参考信息，`Tasks Sequence`提供了一种顺序执行的模式，清晰切分任务的执行内容，尽可能使得模型执行简单的小任务，一步一步完成大任务，将大大提升大模型的执行效率与应用的稳定性。

**思考一下：我们该如果构建一个天气查询播报机器人，通过对话的方式让大模型播报当前的天气信息。**

或许这里可以将`天气查询播报机器人`拆分为几个个主要的`Agent`动作。

1. 接受用户的问题，分析并解析用户所要查询的地址信息。这是因为绝大多数的天气查询接口，只支持标准的行政区划地址，然而用户问题一般不是标准地址名称。
2. 查询天气，调用相关的天气查询接口，获取当前的天气信息。
3. 将天气信息返回的`json`内容提供给大模型，让其总结当前天气的概况。

<center>
<img src="./img/zlai-tasks-weather.png" width="780px">
<h5>Fig. Tasks Weather</h5>
</center>

### `Tasks Sequence`示例：天气查询播报

> step 1: `Agent`配置`List`

**在第一个步骤中，我们仅定义了两个agent，用于执行解析地址，与天气查询**

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

**这里有两种方式可以定义顺序执行对象**

**Method-1**: *使用`TaskSequence`给定`task_weather`任务列表即可，我们推荐使用这种方式定义任务流程。*

```python
from zlai.agent import *

task_seq = TaskSequence(task_list=task_weather, verbose=True)
```

**Method-2**: *第二种方式是自定义一个类，继承`TaskSequence`。这种方式相对繁琐，但您可以增加更多自定义的执行逻辑。*

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

执行任务就比较简单了，只需要将相应的问题给到`task_seq`对象即可。

```python
query = "杭州今天天气怎么样？"
answer = task_seq(query=query)
```

以下为开启了`verbose=True`后输出的执行信息，详细记录了消息流转状态。 我们这里对消息内容做简单的解释：

*1. 第一部分，接受到两个任务，并开始任务的执行，并且打印出了用户的问题。中括号内标记的是当前日志来自于哪个`Task`。如， *

* 第一条信息中消息来自`Task Sequence`。
* 第二条消息由`Task Sequence`转发至`Address`进行地址的解析。
* 第三条消息来自`Address`，该任务首先打印出了用户问题。

```text
[Task Sequence] Running 2 tasks...

[Task Sequence @ Address] Start ...
[Address] User Question: 余杭今天天气怎么样？
```

*2. 第二部分，在`Address Agent`中执行，这个Agent解析了用户传入的问题，将`余杭`解析为标准的行政区划json(`{'province': '浙江省', 'city': '杭州市', 'district': '余杭区'}`)*

```text
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
```

*3. 第三部分，首先调用了天气查询的接口，返回了天气信息，并将其与用户最开始的问题一同提供给大模型，让大模型依据天气json信息回答问题，并最终给出总结答案。*

```text
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
```

**下面是完整的执行信息：**

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

*嘿嘿，以上是不是更加直观的展示了Agent的执行流程呢。*

### 任务切换

在实际的Agent应用场景中，我们更多的想让大模型自主的选择相应的工具完成各类复杂的任务，比如我希望有一个Agent，可以帮我做`天气查询、股票查询、基金查询、代码编写、数据库查询`等多种的任务，这个时候紧紧定义一种Agent是很难应对这种多样化的任务场景的。任务切换(`Task Switch`)即是为这种场景设计的一种解决方案，通过大模型对不同问题场景的判断，自动选择合适当前问题的Agent并执行，这样就赋予了大模型Agent更强的通用性，以使用即为复杂的任务场景。

> 任务切换(`Task Switch`): 在执行过程中，大模型根据当前任务的状态，自主选择下一个步骤该选择哪一个`Agent`任务，并开始后续的流程。

<center>
<img src="./img/tasks-01.png" width="880px">
<h5>Fig. 任务组织方式</h5>
</center>

> 那么如何实现任务切换呢？

在`zlai`中，您可以使用`TaskSwitch`来完成任务切换，它是一个`Task`的子类，可以被`zlai.TaskSequence`使用。假设有这样的一个简单场景，你需要让大模型同时具备天气查询的功能与闲聊的功能。在这种需求下，你可以使用`TaskSwitch`来完成任务切换。

**与`Task Sequence`一样，这里同样有两种方式可以定义`Task Switch`对象**

**Method-1**: **使用`TaskSwitch`给定`tasks`任务列表即可，我们推荐使用这种更简洁的方式定义任务流程。** 下面是定义示例：

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo
from zlai.agent import TaskSwitch, TaskDescription, ChatAgent, Weather

task_switch_list = [
    TaskDescription(
        task=ChatAgent, task_id=0, task_name="闲聊机器人",
        task_description="""解答用户的各类闲聊问题""",
    ),
    TaskDescription(
        task=Weather, task_id=1, task_name="天气播报机器人",
        task_description="""查询当前的天气数据，并为用户播报当前的天气信息""",
    ),
]

llm = Zhipu(generate_config=ZhipuGLM3Turbo())
chat = TaskSwitch(llm=llm, task_list=task_switch_list, verbose=True)
```

**Method-2**: *第二种方式是自定义一个类，继承`TaskSwitch`。这种方式相对繁琐，但您可以增加更多自定义的执行逻辑。*

```python
from typing import Optional, Any
from zlai.agent import TaskSwitch, TaskDescription, ChatAgent, Weather

class Chat(TaskSwitch):
    """"""
    def __init__(
            self,
            task_name: Optional[str] = "Task SQLite",
            *args: Any,
            **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.task_list = [
            TaskDescription(
                task=ChatAgent, task_id=0, task_name="闲聊机器人",
                task_description="""解答用户的各类闲聊问题""",
            ),
            TaskDescription(
                task=Weather, task_id=1, task_name="天气播报机器人",
                task_description="""查询当前的天气数据，并为用户播报当前的天气信息""",
            ),
        ]

chat = Chat()
```

> **Task Switch** 对象定义好之后，就可以使用它来执行任务了。第一个任务，我们先测试简单的闲聊。

*代码如下：*

```python
task_completion = chat(query="你好")
```

*输出：*

```text
[Task Switch] Start ...

[Task Switch] Question: 你好
==================== [Task] Messages Start ====================

user [2]: [你好]

==================== [Task] Messages End    ====================

[Task Switch] matched task id: ['0']
[Task Switch] task id: 0, task name: 闲聊机器人, content: [task_id: 0]
[Chat Agent] Start ...

[Chat Agent] User Question: 你好

==================== [Chat Agent] Messages Start ====================

user [2]: [你好]

==================== [Chat Agent] Messages End    ====================

[Chat Agent] Final Answer: 你好👋！我是人工智能助手智谱清言（ChatGLM），可以叫我小智🤖，很高兴见到你，欢迎问我任何问题。

[Task Switch] End ...
```

**上面的例子中，我们可以看到大模型首先判断出我们的意图是`闲聊`，并且调用了`闲聊机器人`回答了后面的问题。**

> 接下来，我们来测试一下天气播报机器人。

*代码如下：*

```python
task_completion = chat(query="杭州今天天气怎么样？")
```

*输出：*

```text
[Task Switch] Start ...

[Task Switch] Question: 杭州今天天气怎么样？
==================== [Task] Messages Start ====================

user [10]: [杭州今天天气怎么样？]

==================== [Task] Messages End    ====================

[Task Switch] matched task id: ['1']
[Task Switch] task id: 1, task name: 天气播报机器人, content: [根据用户的问题，需要查询杭州今天的天气情况并进行回答。这符合天气播报机器人的任务描述。

task_id: 1]
[Weather] Running 2 tasks...

[Weather @ Address Agent] Start ...
[Address Agent] User Question: 杭州今天天气怎么样？
==================== [Address Agent] Messages Start ====================

user [10]: [杭州今天天气怎么样？]

==================== [Address Agent] Messages End    ====================

[Address Agent] Final Answer:
{'province': '浙江省', 'city': '杭州市'}
[Address Agent] Parsed address:
[{'province': '浙江省', 'city': '杭州市'}]
[Weather @ Address Agent] End.

[Weather @ Weather Agent] Start ...
[Weather Agent] City Name: 杭州市
==================== [Weather Agent] Messages Start ====================

user [196]: [请根据以下天气信息回答问题。

信息：
'''
{'current_condition': {'temp_C': '31', 'FeelsLikeC': '34', 'humidity': '59',...]

==================== [Weather Agent] Messages End    ====================

[Weather Agent] Final Answer:
根据最新的天气信息，杭州今天的天气状况是部分多云，当前温度为31摄氏度，体感温度为34摄氏度，湿度为59%。观测时间为早上6点。请注意，由于天气状况可能会随时间变化，建议您及时关注天气预报。
[Weather @ Weather Agent] End.

[Weather] End.
[Task Switch] End ...
```

**上面的例子中，我们可以看到大模型首先判断出我们的意图是`天气播报`，并且调用了`天气播报机器人`回答了后面的问题。**

**虽然，上面的例子看起来些许简单，甚至有些幼稚，但是他确实初步具备了执行复杂流程的基础。让我们从这里出发，构建更复杂的智能体。**

最后，我们简单介绍一下`TaskCompletion`

## TaskCompletion

> `TaskCompletion`是对任务执行结果的封装。包含了最终的模型输出，与任务执行过程中的任何数据信息。方便查询、分析、管理、可视化Agent的中间节点。

```python
task_completion = chat(query="杭州今天天气怎么样？")

print(task_completion)
print(chat.task_completions)
```

**`task_completion`是最终的结果信息，数据类型为[TaskCompletion]**

```text
TaskCompletion(
    query='杭州今天天气怎么样？',
    task_id=None, 
    task_name=None, 
    content=None, 
    stream=None, 
    delta=None, 
    script=None, 
    parsed_data=None, 
    observation=None, 
    data=None, 
    task_description=None
)
```

*参数说明*

- `query`: 用户最开始的问题
- `task_id`: Agent 任务 ID
- `task_name`: Agent 任务名称
- `content`: 大模型回答的文本内容
- `stream`: 若为流式输出，则返回流式输出的全量内容
- `delta`: 若为流式输出，则返回流式输出的增量内容
- `script`: 如果是写脚本的任务，该位置存储脚本文本
- `parsed_data`: 从大模型输出中的解析后的数据
- `observation`: 一般是执行了代码或执行了调用API后返回的内容
- `data`: 存储从数据库或其他数据源中获取的数据对象，比如一个`DataFrame`对象
- `task_description`: 任务描述，数据类型为`TaskDescription`，记录了该任务的`prompt/few-shot/大模型`等相关信息

**`chat.task_completions`记录了每一次操作的结果信息数据类型为[List[TaskCompletion]]**

```text
[
    TaskCompletion(query='杭州今天天气怎么样？', task_id=None, task_name=None, content=None, stream=None, delta=None, script=None, parsed_data=None, observation=None, data=None, task_description=None), 
    TaskCompletion(query='杭州今天天气怎么样？', task_id=None, task_name=None, content=None, stream=None, delta=None, script=None, parsed_data=None, observation=None, data=None, task_description=TaskDescription(task=<class 'zlai.agent.weather.Weather'>, task_id=1, task_name='天气播报机器人', task_description='查询当前的天气数据，并为用户播报当前的天气信息', task_parameters=TaskParameters(llm=None, embedding=None, db=None, db_path=None, system_message=None, system_template=None, prompt_template=None, few_shot=None, messages_prompt=None, logger=None, verbose=None, index_name=None, elasticsearch_host=None, kwargs=None))), 
    TaskCompletion(query='杭州今天天气怎么样？', task_id=None, task_name=None, content='杭州今天的天气状况是阴天，当前温度为20摄氏度，湿度为72%，体感温度也是20摄氏度。这是凌晨4:54的观测数据。', stream=None, delta=None, script=None, parsed_data={'province': '浙江省', 'city': '杭州市', 'district': ''}, observation=None, data=None, task_description=None)
]
```

> **好嘞，我们希望通过以上繁琐的介绍，您能了解`zlai`中处理Agent任务的逻辑与机制，这对后续的任务有着非常大的帮助。并且以上流程机制几乎适用于整个`zlai`中的Agent设计逻辑。** 在后面的介绍中我们将介绍更多的使用示例，并会逐步介绍`zlai`中的Agent任务设计机制。希望这些对您有帮助。

-----
@2024/05/23

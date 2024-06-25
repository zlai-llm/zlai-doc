# ToolsAgent

> `ToolsAgent`是基于 Function Calling 的 Agent 实现方式，它允许用户使用自定义的函数来完成 Agent 的构建与运行。您只需要指定要大模型调用的函数，即可创建一个可用的 Agent。 下面是一个使用示例：

这个示例中我们给大模型两个可以使用的函数。

- `get_weather`: 可以查询天气的函数
- `random_number_generator`: 给定值域范围，生成随机数的函数

```python
from zlai.llms import Zhipu, GLM4AirGenerateConfig
from zlai.agent.tools import *

# 创建大模型实例
llm = Zhipu(generate_config=GLM4AirGenerateConfig())

# 将定义好的工具 get_weather, random_number_generator 添加到工具集Tools中
tools = Tools(function_list=[get_weather, random_number_generator])

# 创建一个ToolsAgent实例，并指定大模型llm和工具集tools
agent = ToolsAgent(llm=llm, tools=tools, verbose=True)
```

```text
[Tools Agent] Registered 2 Tools: ['get_weather', 'random_number_generator']
```

*上面代码运行后，会显示已经注册了两个工具，等待大模型的调用。*

**测试-1: 让大模型调用随机数生成函数，生成随机数字**

```python
task_completion = agent("输出一个0-6的随机数？")
print(task_completion.content)
```

*输出Log*

```text
[Tools Agent] Start ...

[Tools Agent] User Question: 输出一个0-6的随机数？

==================== [Tools Agent] Messages Start ====================

system [28]: [You are a helpful assistant.]

user [12]: [输出一个0-6的随机数？]

==================== [Tools Agent] Messages End    ====================

[Tools Agent] Answer Content: 
[Tools Agent] Call Tool: random_number_generator
[Tools Agent] Tool Params: {'range': (0, 7), 'seed': 1}
[Tools Agent] Tool Data: 2
[Tools Agent] Final Answer: 生成的0-6之间的随机数是2。

生成的0-6之间的随机数是2。
```

**测试-2: 让大模型调用天气查询API查询某地天气**

```python
task_completion = agent("余杭区的天气如何？")
print(task_completion.content)
```

*输出Log*

```text
[Tools Agent] Start ...

[Tools Agent] User Question: 余杭区的天气如何？

==================== [Tools Agent] Messages Start ====================

system [28]: [You are a helpful assistant.]

user [9]: [余杭区的天气如何？]

==================== [Tools Agent] Messages End    ====================

[Tools Agent] Answer Content: 
[Tools Agent] Call Tool: get_weather
[Tools Agent] Tool Params: {'city_name': '余杭区'}
[Tools Agent] Tool Data: {'current_condition': {'temp_C': '21', 'FeelsLikeC': '21', 'humidity': '96', 'weatherDesc': [{'value': 'Light rain'}], 'observation_time': '07:07 AM'}}
[Tools Agent] Final Answer: 余杭区当前的天气情况是：轻微降雨，温度为21摄氏度，体感温度也是21摄氏度，湿度为96%，观测时间为早上7点07分。

余杭区当前的天气情况是：轻微降雨，温度为21摄氏度，体感温度也是21摄氏度，湿度为96%，观测时间为早上7点07分。
```

-----

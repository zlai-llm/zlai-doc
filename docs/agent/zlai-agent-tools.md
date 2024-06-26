# ToolsAgent

## ToolsAgent 介绍

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

## 自定义ToolsAgent

### 自定义工具函数

> 下面自定义了一个基金当前净值查询函数，可以依据基金代码查询当前净值数据。

```python
from typing import Annotated, Union, Dict
from zlai.tools import get_url, CurrentFundData
from zlai.parse import ParseDict, ParseList

def get_current_fund(
        fund_code: Annotated[str, '基金代码', True] = '000001',  # Annotated 用于描述函数参数的类型、描述、是否必填 
) -> Union[Dict, str]:
    """
    依据用户传入的基金代码，获取当前基金的净值、涨幅等信息。
    :param fund_code: 基金代码
    :return: 基金当前数据
    """
    url = f"http://fundgz.1234567.com.cn/js/{fund_code}.js"
    text = get_url(url)
    data = ParseDict.eval_dict(text)
    if len(data) > 0:
        return CurrentFundData(**data[0]).map_dict()
    else:
        return f"未查询到 {fund_code} 的数据。"

data = get_current_fund(fund_code='008888')
print(data)
```

*输出*

```text
{'基金代码': '008888', '基金名称': '华夏国证半导体芯片ETF联接C', '上一交易日': '', '基金净值（截止上一交易日）': '0.7613', '估算净值（实时）': '0.7567', '估算涨幅（实时）': '-0.60', '更新时间（实时）': '2024-06-26 11:12'}
```

### ToolsAgent调用自定义工具

1. 使用`Tools`封装工具集
2. 使用`ToolsAgent`集成大模型与工具函数

```python
from zlai.llms import Zhipu, GLM4AirGenerateConfig
from zlai.agent import Tools, ToolsAgent

llm = Zhipu(generate_config=GLM4AirGenerateConfig())

tools = Tools(function_list=[get_current_fund])
agent = ToolsAgent(llm=llm, tools=tools, verbose=True)
task_completion = agent("帮忙查询基金代码为008888的当前行情数据。")
print(task_completion.content)
```

*输出Log*

```text
[Tools Agent] Registered 1 Tools: ['get_current_fund']

[Tools Agent] Start ...

[Tools Agent] User Question: 帮忙查询基金代码为008888的当前行情数据。

==================== [Tools Agent] Messages Start ====================

system [28]: [You are a helpful assistant.]

user [23]: [帮忙查询基金代码为008888的当前行情数据。]

==================== [Tools Agent] Messages End    ====================

[Tools Agent] Answer Content: 
[Tools Agent] Call Tool: get_current_fund
[Tools Agent] Tool Params: {'fund_code': '008888'}
[Tools Agent] Tool Data: {'基金代码': '008888', '基金名称': '华夏国证半导体芯片ETF联接C', '上一交易日': '', '基金净值（截止上一交易日）': '0.7613', '估算净值（实时）': '0.7570', '估算涨幅（实时）': '-0.57', '更新时间（实时）': '2024-06-26 11:16'}

[Tools Agent] Final Answer: 基金代码008888的当前行情数据如下：
- 基金名称：华夏国证半导体芯片ETF联接C
- 上一交易日：无数据
- 基金净值（截止上一交易日）：0.7613
- 估算净值（实时）：0.7570
- 估算涨幅（实时）：-0.57%
- 更新时间（实时）：2024-06-26 11:16
```

-----
@2024/06/27

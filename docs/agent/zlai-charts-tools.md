# Tools-Agent绘图

## Line chart

> 代码

```python
from zlai.tools import *
from zlai.schema import SystemMessage
from zlai.agent import ToolsAgent, Tools
from zlai.llms import Zhipu, GLM4GenerateConfig

llm = Zhipu(generate_config=GLM4GenerateConfig())

system_message = SystemMessage(content="You are a data drawing robot.")
tools = Tools(tool_list=[base_chart, pie_chart], params_fun=transform_tool_params)
agent = ToolsAgent(llm=llm, tools=tools, system_message=system_message, verbose=True)

task_completion = agent("汉东省2024年一季度的GDP数据为1.2万亿、2.3万亿、4.5万亿，请依据这些数据，绘制一个折线图，数据来源。")
```

> 效果

<center>
<iframe width="400px" height="520px" src="charts/line.html"></iframe>
<h6> Line chart </h6>
</center>

## Pie chart

> 代码

```python
# 前置代码同 Line chart
task_completion = agent("比亚迪、特斯拉、蔚来、理想的4月份销量数据为100、200、300、400，请依据这些数据，绘制一个饼图，数据来源（大黄蜂汽车资讯）。")
```

> 效果

<center>
<iframe width="400px" height="520px" src="charts/pie.html"></iframe>
<h6> Pie chart </h6>
</center>

------

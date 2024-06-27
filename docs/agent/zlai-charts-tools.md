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
tools = Tools(tool_list=[base_chart, pie_chart, radar_chart, scatter_chart, map_chart], params_fun=transform_tool_params)
agent = ToolsAgent(llm=llm, tools=tools, system_message=system_message, verbose=True)

task_completion = agent("汉东省2024年一季度的GDP数据为1.2万亿、2.3万亿、4.5万亿，请依据这些数据，绘制一个折线图，数据来源。")
```

> 效果

<center>
<iframe width="400px" height="520px" src="charts/line.html"></iframe>
<h6> Line chart </h6>
</center>

## Bar chart

> 代码

```python
# 前置代码同 Line chart
task_completion = agent("汉东省2024年一季度的GDP数据为1.2万亿、2.3万亿、4.5万亿，请依据这些数据，绘制一个柱状图，数据来源。")
```

> 效果

<center>
<iframe width="400px" height="520px" src="charts/bar.html"></iframe>
<h6> Bar chart </h6>
</center>

## Scatter chart

> 代码

```python
# 前置代码同 Line chart
import numpy as np

data = np.random.randint(high=10, low=0, size=(5, 2)).tolist()
task_completion = agent(f"请绘制一个散点图展示，数据为{data}。")
```

> 效果

<center>
<iframe width="400px" height="520px" src="charts/scatter.html"></iframe>
<h6> Scatter chart </h6>
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

## Radar chart

> 代码

```python
# 前置代码同 Line chart
task_completion = agent("""三国武将的谋略、武力、道德、攻速、领导力数据分别是，
吕布（56, 99, 40, 67, 47）、
关羽（89, 92, 99, 82, 95）、
张飞（47, 89, 85, 67, 73）、
赵云（86, 91, 87, 98, 69），
请绘制雷达图展示。
""")
```

> 效果

<center>
<iframe width="400px" height="520px" src="charts/radar.html"></iframe>
<h6> Radar chart </h6>
</center>

## Map chart

> 代码

```python
# 前置代码同 Line chart
task_completion = agent("""2024年一季度的河北省、河南省、浙江省、广东省粮食收获数据为5.2亿吨、7.3亿吨、2.5亿吨、3.5亿吨，\
绘制一个地图清晰展示该数据，数据来源（农业部）。""")
```

> 效果

<center>
<iframe width="400px" height="520px" src="charts/map.html"></iframe>
<h6> Map chart </h6>
</center>

------

# 公募基金问答

## 查询基金代码

> 依据`基金名称`或者`基金代码`，获取基金代码、基金拼音简写、基金类型、基金拼音全称等信息

**下面是代码示例**

*示例-1*

```python
from zlai.tools import *
from zlai.agent import *
from zlai.llms import Zhipu, GLM4GenerateConfig

# 初始化智谱大模型
llm = Zhipu(generate_config=GLM4GenerateConfig())
# 基金查询相关工具集
tool_list = [
    search_fund, get_fund_basic_info, get_current_fund,
    get_current_fund, get_fund_company, get_fund_history
]
tools = Tools(tool_list=tool_list)

# 创建基金查询问答Agent 
agent = ToolsAgent(llm=llm, tools=tools, verbose=True)

# 第一个问题
task_completion = agent("帮忙查询基金代码为008888的的名称、基金类型、基金拼音全称。")
```

*输出*

```text
[Tools Agent] Call Tool: search_fund
[Tools Agent] Tool Params: {'fund_code': '008888'}
[Tools Agent] Tool Data: [{'基金代码': '008888', '基金拼音简写': 'HXGZBDTXPETFLJC', '基金名称': '华夏国证半导体芯片ETF联接C', '基金类型': '指数型-股票', '基金拼音全称': 'HUAXIAGUOZHENGBANDAOTIXINPIANETFLIANJIEC'}]
[Tools Agent] Final Answer: 基金代码008888对应的基金名称是“华夏国证半导体芯片ETF联接C”，基金类型是“指数型-股票”，基金拼音全称是“HUAXIAGUOZHENGBANDAOTIXINPIANETFLIANJIEC”。
```

*示例-2*

```python
# 前置代码参考示例-1
task_completion = agent("帮忙查询基金名称为东方新能源汽车混合的基金代码、基金类型、基金拼音简写")
```

*输出*

```text
[Tools Agent] Call Tool: search_fund
[Tools Agent] Tool Params: {'fund_name': '东方新能源汽车混合'}
[Tools Agent] Tool Data: [{'基金代码': '400015', '基金拼音简写': 'DFXNYQCHH', '基金名称': '东方新能源汽车混合', '基金类型': '混合型-偏股', '基金拼音全称': 'DONGFANGXINNENGYUANQICHEHUNHE'}]
[Tools Agent] Final Answer: 根据您的查询，我已经帮您找到了基金名称为“东方新能源汽车混合”的基金代码、基金类型和基金拼音简写。该基金的基金代码为“400015”，基金类型为“混合型-偏股”，基金拼音简写为“DFXNYQCHH”。
```

## 查询基金基本信息

> 基本信息包括以下字段

```text
[
    '基金全称', '基金简称', '基金代码', '基金类型', '发行日期', '成立日期/规模', '资产规模', '份额规模', '基金管理人',
    '基金托管人', '基金经理人', '成立来分红', '管理费率', '托管费率',  '销售服务费率', '最高认购费率', '最高申购费率',
    '最高赎回费率', '业绩比较基准', '跟踪标的'
]
```

> 示例

```python
# 前置代码参考示例-1
task_completion = agent("帮忙查询基金代码为008888的基金基本信息，并总结这只基金的基本信息。")
```

*输出*

```text
[Tools Agent] Call Tool: get_fund_basic_info
[Tools Agent] Tool Params: {'fund_code': '008888'}
[Tools Agent] Tool Data: {'基金全称': '华夏国证半导体芯片交易型开放式指数证券投资基金发起式联接基金', '基金简称': '华夏国证半导体芯片ETF联接C', '基金代码': '008888（前端）', '基金类型': '指数型-股票', '发行日期': '2020年05月08日', '成立日期/规模': '2020年06月02日 / 10.280亿份', '资产规模': '56.95亿元（截止至：2024年03月31日）', '份额规模': '74.2915亿份（截止至：2024年03月31日）', '基金管理人': '华夏基金', '基金托管人': '建设银行', '基金经理人': '赵宗庭', '成立来分红': '每份累计0.00元（0次）', '管理费率': '0.50%（每年）', '托管费率': '0.10%（每年）', '销售服务费率': '0.30%（每年）', '最高认购费率': '0.00%（前端）', '最高申购费率': '0.00%（前端）', '最高赎回费率': '1.50%（前端）', '业绩比较基准': '国证半导体芯片指数收益率*95%+人民币活期存款税后利率*5%', '跟踪标的': '国证半导体芯片指数'}
[Tools Agent] Final Answer: 基金代码为008888的基金是一只指数型-股票基金，全称为“华夏国证半导体芯片交易型开放式指数证券投资基金发起式联接基金”，简称“华夏国证半导体芯片ETF联接C”。该基金由华夏基金管理，建设银行托管，赵宗庭担任基金经理。基金成立于2020年06月02日，初始规模为10.280亿份，截止至2024年03月31日，资产规模为56.95亿元，份额规模为74.2915亿份。基金自成立以来未进行过分红，管理费率为0.50%，托管费率为0.10%，销售服务费率为0.30%，最高认购费率和最高申购费率为0.00%，最高赎回费率为1.50%。基金的业绩比较基准为国证半导体芯片指数收益率*95%+人民币活期存款税后利率*5%，跟踪标的为国证半导体芯片指数。
```

## 查询基金公司

> 给定基金公司代码查询基金公司名称，或给定基金公司名称查询基金公司代码

> 示例

*示例-1*

```python
# 前置代码参考示例-1
task_completion = agent("帮忙查询基金公司为建信基金的公司代码")
```

*输出*

```text
[Tools Agent] Call Tool: get_fund_company
[Tools Agent] Tool Params: {'company_code': '', 'company_name': '建信基金'}
[Tools Agent] Tool Data: [{'基金公司代码': '80065990', '基金公司名称': '建信基金'}]
[Tools Agent] Final Answer: 建信基金的公司代码是80065990。
```

*示例-2*

```python
task_completion = agent("帮忙查询基金公司代码80404011的基金公司名称。")
```

*输出*

```text
[Tools Agent] Call Tool: get_fund_company
[Tools Agent] Tool Params: {'company_code': '80404011', 'company_name': ''}
[Tools Agent] Tool Data: [{'基金公司代码': '80404011', '基金公司名称': '申万宏源证券'}]
[Tools Agent] Final Answer: 基金公司代码80404011对应的基金公司名称是申万宏源证券。
```

## 查询当前行情

> 依据基金代码，获取当前基金的净值、涨幅等信息。

**示例**

```python
# 前置代码参考示例-1
task_completion = agent("帮忙查询基金代码为008888的当前行情。")
```

*输出*

```text
[Tools Agent] Call Tool: get_current_fund
[Tools Agent] Tool Params: {'fund_code': '008888'}
[Tools Agent] Tool Data: {'基金代码': '008888', '基金名称': '华夏国证半导体芯片ETF联接C', '上一交易日': '', '基金净值（截止上一交易日）': '0.7680', '估算净值（实时）': '0.7762', '估算涨幅（实时）': '1.07', '更新时间（实时）': '2024-06-28 13:04'}
[Tools Agent] Final Answer: 基金代码008888的当前行情如下：
- 基金名称：华夏国证半导体芯片ETF联接C
- 上一交易日：未提供
- 基金净值（截止上一交易日）：0.7680
- 估算净值（实时）：0.7762
- 估算涨幅（实时）：1.07%
- 更新时间（实时）：2024-06-28 13:04

请注意，以上信息是根据最新的数据更新，但市场价格可能会有所变动。
```

## 查询历史行情

> 给定基金代码、查询开始与结束日期，查询日期内该基金的历史行情信息。

```python
# 前置代码参考示例-1
task_completion = agent("帮忙查询基金代码为008888在2024年5月10日至5月17日的历史行情，并做分析。")
```

*输出*

```text
[Tools Agent] Call Tool: get_fund_history
[Tools Agent] Tool Params: {'end_date': '2024-05-17', 'fund_code': '008888', 'start_date': '2024-05-10'}
[Tools Agent] Tool Data: | 净值日期   |   单位净值 |   累计净值 |   日增长率 | 申购状态   | 赎回状态   |
|:-----------|-----------:|-----------:|-----------:|:-----------|:-----------|
| 2024-05-17 |     0.7512 |     0.7512 |       1.05 | 开放申购   | 开放赎回   |
| 2024-05-16 |     0.7434 |     0.7434 |       0.09 | 开放申购   | 开放赎回   |
| 2024-05-15 |     0.7427 |     0.7427 |      -1.2  | 开放申购   | 开放赎回   |
| 2024-05-14 |     0.7517 |     0.7517 |       0.07 | 开放申购   | 开放赎回   |
| 2024-05-13 |     0.7512 |     0.7512 |      -0.81 | 开放申购   | 开放赎回   |
| 2024-05-10 |     0.7573 |     0.7573 |      -1.94 | 开放申购   | 开放赎回   |

[Tools Agent] Final Answer: 基金代码008888在2024年5月10日至5月17日的历史行情如下：

- 2024年5月10日，单位净值为0.7573，累计净值为0.7573，日增长率为-1.94%，申购状态为开放申购，赎回状态为开放赎回。
- 2024年5月13日，单位净值为0.7512，累计净值为0.7512，日增长率为-0.81%，申购状态为开放申购，赎回状态为开放赎回。
- 2024年5月14日，单位净值为0.7517，累计净值为0.7517，日增长率为0.07%，申购状态为开放申购，赎回状态为开放赎回。
- 2024年5月15日，单位净值为0.7427，累计净值为0.7427，日增长率为-1.2%，申购状态为开放申购，赎回状态为开放赎回。
- 2024年5月16日，单位净值为0.7434，累计净值为0.7434，日增长率为0.09%，申购状态为开放申购，赎回状态为开放赎回。
- 2024年5月17日，单位净值为0.7512，累计净值为0.7512，日增长率为1.05%，申购状态为开放申购，赎回状态为开放赎回。

在这段时间内，基金的净值波动较大，尤其是在5月10日和5月15日，分别出现了-1.94%和-1.2%的较大跌幅。不过，在5月17日，基金净值又出现了1.05%的增长。总体来看，这段期间基金的净值表现不稳定，投资者可能需要关注基金的风险。
```

-----
@2024/06/28
# Agent对CSV进行数据分析

> 在这一小节中介绍如何使用大模型Agent对CSV数据处理。

## 数据

> 我们使用titanic数据集。

```python
import pandas as pd
from zlai.llms import *
from zlai.embedding import *
from zlai.agent import CSV, CSVQA, Tasks, task_list
from IPython.display import display, Markdown

csv_path = "./data/titanic.csv"
df = pd.read_csv(csv_path)
df.head()
```

*数据实例：*

```text
|    |   Survived |   Pclass | Name                                               | Sex    |   Age |   Siblings/Spouses Aboard |   Parents/Children Aboard |    Fare |
|---:|-----------:|---------:|:---------------------------------------------------|:-------|------:|--------------------------:|--------------------------:|--------:|
|  0 |          0 |        3 | Mr. Owen Harris Braund                             | male   |    22 |                         1 |                         0 |  7.25   |
|  1 |          1 |        1 | Mrs. John Bradley (Florence Briggs Thayer) Cumings | female |    38 |                         1 |                         0 | 71.2833 |
|  2 |          1 |        3 | Miss. Laina Heikkinen                              | female |    26 |                         0 |                         0 |  7.925  |
|  3 |          1 |        1 | Mrs. Jacques Heath (Lily May Peel) Futrelle        | female |    35 |                         1 |                         0 | 53.1    |
|  4 |          0 |        3 | Mr. William Henry Allen                            | male   |    35 |                         0 |                         0 |  8.05   |
```

## Table-QA

### Prompt

> 这里只使用了简单的Prompt，以及数据的前5行。

```text
The following is the top 5 lines of a table, \
please answer the user’s question according to these contents.

{df_head_markdown}
```

### Example

> 让大模型理解这份数据，并回答一些问题。这里只是让模型根据提供给他的示例数据做简单的数据介绍，并不需要进行复杂的分析，也没有进行复杂数据的计算与索引。

```python
llm = Zhipu(generate_config=ZhipuGLM3Turbo(temperature=0.01))
csv = CSVQA(csv_path=csv_path, llm=llm, verbose=True)
query = "请使用中文介绍这个表格"
answer = csv.generate(query=query)

display(Markdown(answer))
```

*输出*: 

```text
[CSV-QA] Start ...

[CSV-QA]Question: 请使用中文介绍这个表格

==================== [CSV-QA] Messages Start ====================
user: [请使用中文介绍这个表]

==================== [CSV-QA] Messages End    ====================
[CSV-QA]Answer: 当然可以。这个表格是关于一些人的基本信息，这些人可能与一次特定的历史事件有关，比如泰坦尼克号沉船事故。表格列出了五个主要的信息类别：

1. **序号** (`|    |`): 这可能是指每个人的记录编号。
2. **是否生存** (`|   Survived |`): 这一列表明了这些人在事件中是否存活下来。`0` 可能表示未生存，而 `1` 可能表示生存。
3. **船票等级** (`|   Pclass |`): 这个等级可能指的是他们在船上的舱位等级，例如 `1` 代表一等舱，`3` 代表三等舱。
4. **姓名** (`| Name                                               |`): 这一列是这些人的姓名。
5. **性别** (`| Sex    |`): 表明这些人的性别，`male` 代表男性，`female` 代表女性。
6. **年龄** (`|   Age |`): 这些人的年龄。
7. **兄弟姐妹/配偶是否同行** (`|   Siblings/Spouses Aboard |`): 这可能指的是这些人在船上是否有兄弟姐妹或配偶同行。
8. **父母/孩子是否同行** (`|   Parents/Children Aboard |`): 这可能指的是这些人在船上是否有父母或孩子同行。
9. **票价** (`|    Fare |`): 这些人支付的船票费用。

根据表格中的数据，我们可以看到：

- Owen Harris Braund 是一位22岁的男性，乘坐三等舱，存活了下来，他的票价是7.25。
- Florence Briggs Thayer Cumings 是一位38岁的女性，乘坐一等舱，存活了下来，她的票价是71.2833。
- Laina Heikkinen 是一位26岁的女性，乘坐三等舱，存活了下来，她的票价是7.925。
- Lily May Peel Futrelle 是一位35岁的女性，乘坐一等舱，存活了下来，她的票价是53.1。
- William Henry Allen 是一位35岁的男性，乘坐三等舱，没有存活下来，他的票价是8.05。

这个表格可能是用来分析泰坦尼克号上不同人群的生存情况，包括他们的性别、舱位等级和年龄等因素。

[CSV-QA] End ...
```

## Table-query

> 对于复杂的提问，我们需要准确的计算数据，并给出相应的答案。具体有三个步骤：

1. 根据用户的问题、示例数据生成可执行代码。
2. 执行代码，并获取结果。
3. 将数据结果与问题提供给模型，让模型总结输出答案。

### Prompt

> system message

```text
Given you a question and the execution result of the relevant code, please answer the user's question concisely based on the observation.
```

> 生成代码 generate code

```text
You have access to a pandas dataframe `df`. \
Here is the output of `df.head().to_markdown()`:

{df_head_markdown}

Given a user question, write the Python code to answer it. \
Return ONLY the valid Python code and nothing else. \
Don't assume you have access to any libraries other than built-in Python ones and pandas.
```

> 总结回答

```text
Question: {question}
Script: {script}
Observation：```\n{observation}\n```
Answer：
```

### Example

> 问题-1: 表中有多少数据？

```python
llm = Zhipu(generate_config=ZhipuGLM3Turbo(temperature=0.01))
csv = CSV(csv_path=csv_path, llm=llm, verbose=True)

query = "表中有多少数据？"
answer = csv.generate(query=query)
display(Markdown(answer))
```

*输出*:

```text
[CSV-Script] Start ...

[CSV-Script]Question: 表中有多少数据？

[CSV-Script]Assistant: """python
len(df)
"""

[CSV-Script]Parsed code: """
len(df)
"""

[CSV-Script]Tools invoke: 887

==================== [CSV-Script] Messages Start ====================
user: [表中有多少数据]

assistant: ["""python
len(df)
"""]

user: [
Question: 表中有多少数据？
Observation："""
887
"""
Answer：]

==================== [CSV-Script] Messages End    ====================
[CSV-Script]Final Answer: 根据您提供的执行结果，表中的数据量是887条记录。.

[CSV-Script] End ...
```

> 问题-2: 生存率是多少？

```python
query = "生存率是多少？"
csv.generate(query=query)
```

*输出*:

```text
[CSV-Script] Start ...

[CSV-Script]Question: 生存率是多少？

[CSV-Script]Assistant: """python
survival_rate = df['Survived'].mean()
print(f"生存率为: {survival_rate:.2f}")
"""

[CSV-Script]Parsed code: """
survival_rate = df['Survived'].mean()
print(f"生存率为: {survival_rate:.2f}")

"""
[CSV-Script]Tools invoke: 生存率为: 0.39


==================== [CSV-Script] Messages Start ====================
user: [生存率是多少]

assistant: ["""python
survival_rate = df['Survived'].mean()
print(f"生存率为: {survival_rate:.2f}")
``]

user: [
Question: 生存率是多少？
Observation："""
生存率为: 0.39

"""
Answer：]

==================== [CSV-Script] Messages End    ====================
[CSV-Script]Final Answer: 根据您提供的代码和执行结果，生存率是39%，具体到两位小数是0.39。.

[CSV-Script] End ...
```

## Agent-Task

在之前的[Sequence-Tasks/Switch-Tasks](agent/zlai-agent-01?id=task-组织方式)的介绍中，我们介绍了如何将多个任务进行组合，现在您可以将使用`Task`来组织`Agent`，使其可以应对各类`CSV`问答场景。

### Task-List

> 现在我们思考以下三个`CSV`数据问答场景：

1. 对`CSV`数据进行基本介绍，例如表中数据的基本信息等，这种情况下只需要提供表的示例数据，而不需要执行代码。
2. 对`CSV`数据进行数据提取，进行某种形式的加工后获得最终的答案，例如查询表中数据的具体信息，计算平均值等，这种情况下需要提供表的示例数据，并让大模型写出需要执行的代码，并执行代码，最终依据查询计算到的具体值回答问题。
3. 提供一般的聊天回答，因为在杂糅了多个Agent之后，我们需要保留一个单独的聊天Agent通道以回答其他的各类问题。

> 下面是这三个场景的`Task`列表：

```python
from zlai.agent import TaskDescription, CSVQA, CSVScriptWithObservation

csv_task_list = [
    TaskDescription(
        task=CSVScriptWithObservation, task_id=0, task_name="数据提取与计算机器人",
        task_description="""可以帮你写一段`DataFrame`脚本代码查询表中数据的具体信息""",
    ),
    TaskDescription(
        task=CSVQA, task_id=1, task_name="数据表介绍机器人",
        task_description="""可以介绍并回答数据表的基本信息，但不能够查询真实的数据，只能做一般性的介绍。""",
    ),
]
```

```text
[TaskDescription(task=<class 'zlai.agent.csv.CSV'>, task_id=0, task_name='数据提取与计算机器人', task_description='可以帮你写一段`DataFrame`脚本代码查询表中数据的具体信息'),
 TaskDescription(task=<class 'zlai.agent.csv.CSVQA'>, task_id=1, task_name='数据表介绍机器人', task_description='可以介绍并回答数据表的基本信息，但不能够查询真实的数据，只能做一般性的介绍。')]
```



```python
llm = Zhipu(generate_config=ZhipuGLM3Turbo(temperature=0.01))
csv = Tasks(llm=llm, task_list=task_list, verbose=True)

query = "请使用中文介绍这个表格"
answer = csv.generate(query=query, csv_path=csv_path)

display(Markdown(answer))
```


```text
[Task] Start ...

[Task]Question: 请使用中文介绍这个表格
==================== [Task] Messages Start ====================
user: [请使用中文介绍这个表]

==================== [Task] Messages End    ====================
[Task] matched task id: ['1']
[Task] task id: 1, task name: 数据表介绍机器人, content: [task_id: 1]
[CSV-QA] Start ...

[CSV-QA]Question: 请使用中文介绍这个表格

==================== [CSV-QA] Messages Start ====================
user: [请使用中文介绍这个表]

==================== [CSV-QA] Messages End    ====================
[CSV-QA]Answer: 当然可以。这个表格是关于一些人的基本信息，这些人可能与一次特定的历史事件有关，例如泰坦尼克号沉船事件。表格列出了五个主要的信息类别：

1. **序号** (`|    |`): 这可能是指每个人的记录编号。
2. **是否生存** (`|   Survived |`): 这一列表明了这些人在事件中是否存活下来。`0` 可能表示未生存，而 `1` 表示生存。
3. **船票等级** (`|   Pclass |`): 表示这些乘客购买的船票等级，这里列出的有 `1` 级、`2` 级和 `3` 级，通常船票等级越高，船舱越舒适，费用也越高。
4. **姓名** (`| Name                                               |`): 乘客的名字。
5. **性别** (`| Sex    |`): 乘客的性别。
6. **年龄** (`|   Age |`): 乘客的年龄。
7. **兄弟姐妹/配偶是否同行** (`|   Siblings/Spouses Aboard |`): 这一列表明了是否有兄弟姐妹或配偶在同一艘船上。
8. **父母/孩子是否同行** (`|   Parents/Children Aboard |`): 这一列表明了是否有父母或孩子在同一艘船上。
9. **票价** (`|    Fare |`): 乘客支付的船票费用。

根据表格的前五行数据，我们可以看到：

-  Owen Harris Braund 先生，22岁，是 `3` 级乘客，他的船票费用是 `7.25` 英镑，他独自旅行，没有家人同行。
- Mrs. John Bradley (Florence Briggs Thayer) Cumings 夫人，38岁，是 `1` 级乘客，她的船票费用是 `71.2833` 英镑，她独自旅行，没有家人同行。
- Miss. Laina Heikkinen 女士，26岁，是 `3` 级乘客，她的船票费用是 `7.925` 英镑，她独自旅行，没有家人同行。
- Mrs. Jacques Heath (Lily May Peel) Futrelle 夫人，35岁，是 `1` 级乘客，她的船票费用是 `53.1` 英镑，她有配偶同行，没有孩子或父母同行。
- William Henry Allen 先生，35岁，是 `3` 级乘客，他的船票费用是 `8.05` 英镑，他独自旅行，没有家人同行。

这个表格可能是用来分析乘客的背景、旅行情况以及他们在特定事件中的生存概率等因素的。

[CSV-QA] End ...


[Task] End ...

```


```python
query = "生存率是多少？"
answer = csv.generate(query=query, csv_path=csv_path)

display(Markdown(answer))
```


```python
query = "计算出表格中的性别比例是多少？"
answer = csv.generate(query=query, csv_path=csv_path)

display(Markdown(answer))
```


```text
[Task] Start ...

[Task]Question: 统计表格中船票的不同船票等级的生存率？
==================== [Task] Messages Start ====================
user: [统计表格中船票的不同船票等级的生存率]

==================== [Task] Messages End    ====================
[Task] matched task id: ['0']
[Task] task id: 0, task name: 数据提取与计算机器人, content: [task_id: 0, task_name: 数据提取与计算机器人, task_description: 可以帮你写一段DataFrame脚本代码查询表中数据的具体信息
python
tool_call(task='统计表格中船票的不同船票等级的生存率')
]
[CSV-Script] Start ...

[CSV-Script]Question: 统计表格中船票的不同船票等级的生存率？

[CSV-Script]Assistant: python
# 统计不同船票等级的生存率
survival_rates_by_class = df.groupby('Pclass')['Survived'].mean()
print(survival_rates_by_class)

[CSV-Script]Parsed code:
# 统计不同船票等级的生存率
survival_rates_by_class = df.groupby('Pclass')['Survived'].mean()
print(survival_rates_by_class)

[CSV-Script]Tools invoke: Pclass
1    0.629630
2    0.472826
3    0.244353
Name: Survived, dtype: float64


==================== [CSV-Script] Messages Start ====================
user: [统计表格中船票的不同船票等级的生存率]

assistant: [python
# 统计不同船票等级的生存率
survival_rates_by_class = df.groupby('Pclass')['Survived'].mean()
print(survival_rates_by_class)
]

user: [
Question: 统计表格中船票的不同船票等级的生存率？
Observation：
Pclass
1    0.629630
2    0.472826
3    0.244353
Name: Survived, dtype: float64

Answer：]

==================== [CSV-Script] Messages End    ====================
[CSV-Script]Final Answer: 根据提供的执行结果，我们可以得出以下结论：

- 1等舱的生存率为62.96%，
- 2等舱的生存率为47.28%，
- 3等舱的生存率为24.44%。

这意味着在泰坦尼克号上，一等舱的乘客生存率最高，而三等舱的乘客生存率最低。.

[CSV-Script] End ...


[Task] End ...
```

-------

@start: 2024/05/28
@  end: 


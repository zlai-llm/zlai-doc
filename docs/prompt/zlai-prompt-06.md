# 思维链

> 链式提示

为了提高大语言模型的性能使其更可靠，一个重要的提示工程技术是将任务分解为许多子任务。确定子任务后，将子任务的提示词提供给语言模型，得到的结果作为新的提示词的一部分。这就是所谓的链式提示（prompt chaining），一个任务被分解为多个子任务，根据子任务创建一系列提示操作。链式提示可以完成很复杂的任务。LLM 可能无法仅用一个非常详细的提示完成这些任务。在链式提示中，提示链对生成的回应执行转换或其他处理，直到达到期望结果。

除了提高回答的准确性，链式提示还有助于提高 LLM 应用的透明度，增加控制性和可靠性。这意味着您可以更容易地定位模型中的问题，分析并改进需要提高的不同阶段的性能。链式提示在构建 LLM 驱动的对话助手和提高应用程序的个性化用户体验方面非常有用。

> 文档问答中的链式提示

提示链可以用于不同的场景，这些场景可能涉及多个操作或转换。例如，LLM 的一个常见用途是根据大型文本文档回答问题。想要更好阅读大文本文档，可以设计两个不同的提示，第一个提示负责参考给定问题`提取相关引文`，第二个提示则以`引文和问题`为输入来回答给定的问题。例如：

如果我想知道“谁是美国目前的总统”，我可以使用以下提示链：

下面的第一个提示根据问题从文档中提取相关引文。

*提示 1:*

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo

query = """
你是一个很有帮助的助手。你的任务是根据文档回答问题。第一步是从文档中提取与问题相关的引文。如果没有找到相关引文，请回应“未找到相关引文！”。

####
小约瑟夫·罗比内特·拜登（英语：Joseph Robinette Biden Jr.；1942年11月20日—），通称“乔·拜登”（Joe Biden），美国爱尔兰裔民主党籍政治家，曾任美国特拉华州联邦参议员、美国副总统（第47任），现任美国总统（第46任）。 [41]
拜登毕业于特拉华大学和雪城大学，当过律师，1970年踏入政界，1973年担任特拉华州联邦参议员并六次连任至2009年。期间担任参议院司法委员会主席 [8]及高级成员16年、对外关系委员会主席及高级成员12年 [41]。
1987年和2007年，拜登两度参与民主党的党内总统初选，但均以失败告终。2008、2012年两度成为奥巴马的竞选搭档。 [175]2009—2017年任副总统。 [2]2019年4月25日宣布参选总统。 [3]2021年1月7日正式确认当选美国总统。 [4]1月20日宣誓就任。 [5] [173]
拜登内政和外交经验丰富。 [174]在专业外交领域，他是熟谙外交政策的专家 [185]，以通晓国外政策以及国际安全事务而知名 [183]，关注全球各地人权、种族歧视、减贫、裁军等议题，并在多个关键外交决策中发挥重要作用。拜登在竞选时提出的执政政策则总体温和：政治上推动修复、重振国内民主制度，强化民主联盟。经济上，反对贸易战，维护多边贸易协定；大力推行经济复苏计划，在教育就业、住房收入等领域加大投入 [171]；加大科研投资，推动高新领域建设。 [173]他还提出实行强制口罩令等主张以应对疫情；致力于解决美国“系统性种族主义”问题，维护少数族裔权利。 [171]
####

问题：请提取与问题“谁是美国目前的总统”相关的引文。
"""

llm = Zhipu(generate_config=ZhipuGLM3Turbo())
completion = llm.generate(query=query)
print(f"Assistant: {completion.choices[0].message.content}")
```

*提示 1 的输出：*

```text
Assistant: 美国目前的总统是：“小约瑟夫·罗比内特·拜登（英语：Joseph Robinette Biden Jr.；1942年11月20日—），通称“乔·拜登”（Joe Biden），现任美国总统（第46任）。” [41]
```

在第一个提示中返回的引文现在可以用作下面第二个提示的输入。您可以对这些引文进行清洗，比如移除引用标志。然后，第二个提示接收由第一个提示提取的相关引文，并根据文档和这些提取的引文生成回答。第二个提示可以是以下内容：

*提示 2：*

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo

query = """
请依据引文内容，回答问题

####
美国目前的总统是：“小约瑟夫·罗比内特·拜登（英语：Joseph Robinette Biden Jr.；1942年11月20日—），通称“乔·拜登”（Joe Biden），现任美国总统（第46任）。” [41]
####

问题：谁是美国目前的总统？
"""

llm = Zhipu(generate_config=ZhipuGLM3Turbo())
completion = llm.generate(query=query)
print(f"Assistant: {completion.choices[0].message.content}")
```

*提示 2 的输出：*

```text
Assistant: 美国目前的总统是乔·拜登（Joe Biden）。
```

以上通过两个步骤来依据事实信息回答问题，有效的规避了大模型可能出现幻觉、缺少当前事实信息等问题，这在增强检索任务`RAG`中十分有用，您还可以在这份[文档](https://docs.anthropic.com/claude/docs/prompt-chaining)中找到更多关于提示链的示例。

## 思维链

> 链式思考（CoT）提示(`Chain-of-Thought Prompting`)

<center>
<img src="https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fcot.1933d9fe.png&w=1080&q=75" width="580px">

图片来源：[Wei等人（2022）](https://arxiv.org/abs/2201.11903)
</center>

在 [Wei等人（2022）](https://arxiv.org/abs/2201.11903) 的研究中引入了链式思考（CoT），核心观点是探索了一种称为“思维链提示”（`chain-of-thought prompting`）的方法，这种方法通过生成一系列中间推理步骤来显著提高大型语言模型执行复杂推理的能力。研究表明，通过在提示中提供少量的思维链示例，可以在足够大的语言模型中自然地激发出这种推理能力。您可以将其与少样本提示`Few-shot`相结合，以获得更好的结果。下面是一个示例：

*提示：*

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo

query = """这组数中的奇数加起来是偶数：4、8、9、15、12、2、1。
A：将所有奇数相加（9、15、1）得到25。答案为False。
这组数中的奇数加起来是偶数：17、10、19、4、8、12、24。
A：将所有奇数相加（17、19）得到36。答案为True。
这组数中的奇数加起来是偶数：16、11、14、4、8、13、24。
A：将所有奇数相加（11、13）得到24。答案为True。
这组数中的奇数加起来是偶数：17、9、10、12、13、4、2。
A：将所有奇数相加（17、9、13）得到39。答案为False。
这组数中的奇数加起来是偶数：15、32、5、13、82、7、1。
A："""

llm = Zhipu(generate_config=ZhipuGLM3Turbo())
completion = llm.generate(query=query)
print(f"Assistant: {completion.choices[0].message.content}")
```

*输出：*

```text
Assistant: 将所有奇数相加（15、5、13、7、1）得到41。答案为False。
```

我们可以看到在提供了推理步骤后得到了非常正确的输出结果。

## 零样本 COT 提示

<center>
<img src="https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fzero-cot.79793bee.png&w=1080&q=75" width="580px">

Zero-shot COT 图片来源：[Kojima等人（2022）](https://arxiv.org/abs/2205.11916)
</center>

零样本CoT([Kojima 等人，2022年](https://arxiv.org/abs/2205.11916))，简单来说就是将“让我们逐步思考”添加到原始提示中。让我们尝试一个简单的问题，看看模型的表现如何：

*提示：*

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo

query = """4、8、9、15、12、2、1。
逐步思考，上面的数字中的奇数加起来是否是偶数？："""

llm = Zhipu(generate_config=ZhipuGLM3Turbo())
completion = llm.generate(query=query)
print(f"Assistant: {completion.choices[0].message.content}")
```

*输出：*

```text
Assistant: 首先，我们将给定的数字列出来：4、8、9、15、12、2、1。

然后，我们找出其中的奇数：9、15、1。

接下来，我们将这些奇数相加：9 + 15 + 1 = 25。

最后，我们来判断25是不是偶数。由于25不能被2整除，所以它不是偶数。

因此，上面的数字中的奇数加起来不是偶数。
```

从上面的回答中我们可以看出来，模型对每一个分析步骤都进行了详细的解析，并最终获得了正确的答案。

## 自动思维链

当使用思维链提示时，这个过程需要手工制作有效且多样化的例子。这种手动工作可能会导致次优解决方案。[Zhang et al. （2022）](https://arxiv.org/abs/2210.03493)提出了一种消除人工的方法，即利用 LLMs “让我们一步一步地思考”提示来生成一个接一个的推理链。这种自动过程仍然可能在生成的链中出现错误。为了减轻错误的影响，证明的样例多样性很重要。这项工作提出了自动思维链`Auto-CoT`，它对具有多样性的问题进行采样，并生成推理链来构建演示。

`Auto-CoT` 主要由两个阶段组成：

- 阶段1：**问题聚类**：将给定问题划分为几个聚类
- 阶段2：**演示抽样**：从每组数组中选择一个具有代表性的问题，并使用带有简单启发式的 `Zero-Shot-CoT` 生成其推理链

该过程如下图所示：

<center>
<img src="https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fauto-cot.642d9bad.png&w=1200&q=75">

图片来源：[Zhang等人（2022）](https://arxiv.org/abs/2210.03493)
</center>

[Auto-CoT 的代码Github](https://github.com/amazon-science/auto-cot)。

## 思维树

对于需要探索或预判战略的复杂任务来说，传统或简单的提示技巧是不够的。[Yao et el. (2023)](https://arxiv.org/abs/2305.10601) 提出了思维树（Tree of Thoughts，ToT）框架，该框架基于思维链提示进行了总结，引导语言模型探索把思维作为中间步骤来解决通用问题。

ToT 维护着一棵思维树，思维由连贯的语言序列表示，这个序列就是解决问题的中间步骤。使用这种方法，LLM 能够自己对严谨推理过程的中间思维进行评估。LM 将生成及评估思维的能力与搜索算法（如广度优先搜索和深度优先搜索）相结合，在系统性探索思维的时候可以向前验证和回溯。

ToT 框架原理如下：
<center>
<img src="https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2FTOT.3b13bc5e.png&w=1200&q=75" width="680px">

图片援引自：[Yao et el. (2023)](https://arxiv.org/abs/2305.10601)
</center>

`ToT` 需要针对不同的任务定义思维/步骤的数量以及每步的候选项数量。例如，论文中的“算 24 点游戏”是一种数学推理任务，需要分成 3 个思维步骤，每一步都需要一个中间方程。而每个步骤保留最优的（best） 5 个候选项。

<center>
<img src="https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2FTOT2.9eb8f0f9.png&w=1080&q=75">

图片援引自：[Yao et el. (2023)](https://arxiv.org/abs/2305.10601)
</center>

ToT 完成算 24 点的游戏任务要执行广度优先搜索（BFS），每步思维的候选项都要求 LLM 给出能否得到 24 的评估：“sure/maybe/impossible”（一定能/可能/不可能） 。目的是得到经过少量向前尝试就可以验证正确（sure）的局部解，基于‘太大/太小’的常识消除那些不可能（impossible）的局部解，其余的局部解作为‘maybe’保留。每步思维都要抽样得到 3 个评估结果。

从下图中报告的结果来看，ToT 的表现大大超过了其他提示方法：

<center>
<img src="https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2FTOT3.bf83699e.png&w=828&q=75">

图片援引自：[Yao et el. (2023)](https://arxiv.org/abs/2305.10601)
</center>

[代码示例-1](https://github.com/princeton-nlp/tree-of-thought-llm)

[代码示例-2](https://github.com/jieyilong/tree-of-thought-puzzle-solver)

从大方向上来看，[Yao et el. (2023)](https://arxiv.org/abs/2305.10601) 和 [Long (2023)](https://arxiv.org/abs/2305.08291) 的核心思路是类似的。两种方法都是以多轮对话搜索树的形式来增强 LLM 解决复杂问题的能力。主要区别在于 [Yao et el. (2023)](https://arxiv.org/abs/2305.10601) 采用了深度优先（DFS）/广度优先（BFS）/集束（beam）搜索，而 [Long (2023)](https://arxiv.org/abs/2305.08291) 则提出由强化学习（Reinforcement Learning）训练出的 “ToT 控制器”（ToT Controller）来驱动树的搜索策略。深度优先/广度优先/集束搜索是通用搜索策略，并不针对具体问题。相比之下，由强化学习训练出的 ToT 控制器有可能从新的数据集学习，或是在自对弈（AlphaGo vs. 蛮力搜索）的过程中学习。

[Hulbert (2023)](https://github.com/dave1010/tree-of-thought-prompting) 将 ToT 框架的主要概念概括成了一段简短的提示词，指导 LLM 在一次提示中对中间思维做出评估。ToT 提示词的例子如下：

```text
假设三位不同的专家来回答这个问题。
所有专家都写下他们思考这个问题的第一个步骤，然后与大家分享。
然后，所有专家都写下他们思考的下一个步骤并分享。
以此类推，直到所有专家写完他们思考的所有步骤。
只要大家发现有专家的步骤出错了，就让这位专家离开。
请问...
```

-----
@2024/05/13

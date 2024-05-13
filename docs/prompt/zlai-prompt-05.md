# 事实知识提示

<center>
<img src="https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fgen-knowledge.055b8d37.png&w=828&q=75" width="580px">

图片来源：[Liu 等人 2022](https://arxiv.org/pdf/2110.08387.pdf)
</center>

在LLM的Prompt研究中，一种流行的技术是能够融合知识或信息帮助模型做出更准确的预测。[Liu 等人 2022](https://arxiv.org/pdf/2110.08387.pdf)尝试使语言模型生成知识，然后将这些知识作为额外的输入提供给回答系统，从而让模型得到更为准确的回答；文章强调了语言模型具有常识推理的灵活和高质量知识源的潜力，通过在推理时简单地插入知识，而不需要对模型进行额外的微调，就可以实现知识的整合。论文中介绍了两种知识的生成方式：

> NumerSense任务：

```text
指令：生成关于对象的一些数值事实。示例：
输入：企鹅有<mask>个翅膀。
知识：鸟类有两个翅膀，企鹅是一种鸟类。
```

> 对于QASC(`Question Answering via Sentence Composition`)任务：

```text
指令：生成关于输入的一些知识。示例：
输入：云朵是由哪种类型的水构成的？
知识：云由水蒸气构成。
```

假设你想写一篇关于某种动物的小博客文章，比如中华小熊猫。你可以让大模型首先生成一些关于中华小熊猫的事实。这将有助于`LLM`写一个更翔实文章。下面是一个示例：

> 知识生成

*提示：*

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo

query = """生成5条关于中华小熊猫的事实。"""

llm = Zhipu(generate_config=ZhipuGLM3Turbo())
completion = llm.generate(query=query)
print(f"Assistant: {completion.choices[0].message.content}")
```

*输出：*

```text
Assistant: 1. 中华小熊猫，学名Ailurus styani，是一种小型的小猫熊科哺乳动物，主要分布在中国云南和四川地区。
2. 中华小熊猫体型较小，体长一般在50-70厘米之间，体重在5-10公斤左右，外貌与大熊猫相似，但颜色稍浅，毛发更加柔软。
3. 中华小熊猫主要栖息在海拔1500-3500米的山地森林中，它们以竹叶、果实、昆虫等为食，具有较强的适应能力。
4. 中华小熊猫是一种孤独的动物，通常喜欢独自生活，只在交配季节才会寻找伴侣。它们的繁殖能力较弱，繁殖周期较长，通常每胎只产下一只幼崽。
5. 由于栖息地破坏、食物资源减少等原因，中华小熊猫的数量一度急剧下降，目前已被列为国家一级保护动物。在我国，相关部门已经采取了一系列措施来保护这一珍稀物种，包括设立自然保护区、开展科学研究等。
```

> 知识整合

*提示：*

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo

query = """
1. 中华小熊猫，学名Ailurus styani，是一种小型的小猫熊科哺乳动物，主要分布在中国云南和四川地区。
2. 中华小熊猫体型较小，体长一般在50-70厘米之间，体重在5-10公斤左右，外貌与大熊猫相似，但颜色稍浅，毛发更加柔软。
3. 中华小熊猫主要栖息在海拔1500-3500米的山地森林中，它们以竹叶、果实、昆虫等为食，具有较强的适应能力。
4. 中华小熊猫是一种孤独的动物，通常喜欢独自生活，只在交配季节才会寻找伴侣。它们的繁殖能力较弱，繁殖周期较长，通常每胎只产下一只幼崽。
5. 由于栖息地破坏、食物资源减少等原因，中华小熊猫的数量一度急剧下降，目前已被列为国家一级保护动物。在我国，相关部门已经采取了一系列措施来保护这一珍稀物种，包括设立自然保护区、开展科学研究等。

参考以上信息写一篇博客，使用新闻报道的风格。"""

llm = Zhipu(generate_config=ZhipuGLM3Turbo())
completion = llm.generate(query=query)
print(f"Assistant: {completion.choices[0].message.content}")
```

*输出：*

```text
Assistant: 标题：珍稀动物中华小熊猫：神秘的山林精灵

近日，我国科学家在云南和四川地区对珍稀动物中华小熊猫进行了研究。这种小型小猫熊科哺乳动物，以其独特的外貌和生活习性，吸引了众多研究者的目光。

中华小熊猫，学名Ailurus styani，主要分布在我国云南和四川地区。它们体型较小，体长一般在50-70厘米之间，体重在5-10公斤左右。外貌与大熊猫相似，但颜色稍浅，毛发更加柔软。

中华小熊猫主要栖息在海拔1500-3500米的山地森林中，以竹叶、果实、昆虫等为食，具有较强的适应能力。它们是一种孤独的动物，通常喜欢独自生活，只在交配季节才会寻找伴侣。繁殖能力较弱，繁殖周期较长，通常每胎只产下一只幼崽。

然而，由于栖息地破坏、食物资源减少等原因，中华小熊猫的数量一度急剧下降，目前已被列为国家一级保护动物。为了保护这一珍稀物种，我国相关部门已经采取了一系列措施，包括设立自然保护区、开展科学研究等。

此次研究，我国科学家对中华小熊猫的分类进行了重新界定。根据中国科学院动物研究所研究员魏辅文的研究，中华小熊猫与喜马拉雅小熊猫在约22万年前即分家，其后鲜有遗传上的交流。因此，小熊猫川西亚种被提升为种阶，并命名为中华小熊猫。

中华小熊猫的保护工作任重道远。我们需要进一步加强保护措施，保护好这一珍稀物种，让它们在我国的山林中繁衍生息。同时，我们也呼吁大家提高环保意识，保护好我们的生态环境，让更多的珍稀物种得以生存。

让我们共同守护这片山林，守护这些神秘的精灵，让中华小熊猫的数量逐渐增多，让它们的家园更加美好。
```

**Tips**: *通过生成知识，并将知识再整合的过程，我们同常可以获取更为准确以及风格多变的回答内容。下面是没有加入知识生成的输出，供您对比参考。*

*提示：*

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo

query = """写一篇中华小熊猫的新闻报道。"""

llm = Zhipu(generate_config=ZhipuGLM3Turbo())
completion = llm.generate(query=query)
print(f"Assistant: {completion.choices[0].message.content}")
```

*输出：*

```text
Assistant: 标题：中华小熊猫研究取得重大突破：独立物种认定引关注

近日，中国科学院动物研究所研究员魏辅文的一项最新研究，将中华小熊猫从亚种提升为独立物种，这一成果在学术界引起了广泛关注。这一发现不仅丰富了我国的生物多样性研究，更为世界动物种类目录增添了新的成员。

中华小熊猫，学名Ailurus styani，又称斯坦氏小熊猫，是一种分布于我国云南和四川的小型哺乳动物。过去，学术界一直将小熊猫视为一个物种，其中包括两个亚种，分别是川西亚种和指名亚种。然而，魏辅文研究员的研究发现，这两个亚种在约22万年前就已经分家，其后鲜有遗传上的交流。

通过对来自四川、云南边境旁的密支那地区以及缅甸北部的小熊猫头盖骨和皮毛的收集与研究，魏辅文研究员发现，川西亚种拥有更长的冬季绒毛、更黑的毛皮、更大的头盖骨、更加强烈弯曲的前额以及更强壮的牙齿等特征，与指名亚种明显区别开来。这些特征使得川西亚种符合成为一个独立物种的标准。

此次研究的突破性成果，不仅为小熊猫的物种分类提供了新的科学依据，也为保护这一独特物种提供了更为重要的参考。中华小熊猫作为一种新认定的独立物种，亟需加强对其栖息地、生态习性等方面的研究，以保障其种群的稳定和繁衍。

此外，中华小熊猫的认定也体现了我国在生物多样性保护研究方面取得的成果。近年来，我国政府高度重视生物多样性保护工作，不断加大科研投入，提升保护水平。此次研究的成功，正是我国科研实力的生动体现。

未来，我国将继续加强对中华小熊猫的保护力度，携手全球科学家共同为生物多样性保护作出更大贡献。让我们共同期待，在科学家们的努力下，这一珍稀物种能够繁衍生息，为地球增添更多生机与活力。
```

----
@2024/05/13
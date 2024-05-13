# 大模型参数的设定

## 模型参数是怎样影响回答的？

通过 API 或直接与大语言模型进行交互。你可以通过配置一些参数以获得不同的回答结果。调整这些参数设置对于提高回答的可靠性非常重要，你可能需要进行一些测试才能找出适合您的用例的正确设置。在前面的[章节](/doc/zlai-llm-04.md)中已经展示了自回归模型的生成原理，以及一些参数设置，以下会简明的罗列常见的大模型参数设置：

### 控制输出结果的长度

#### 提前停止

`stop`模型在遇到stop所制定的字符时将停止生成。假设我们只需要模型生成一句话，下面是一个示例，

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo

llm = Zhipu(generate_config=ZhipuGLM3Turbo(stop=["。"]))
completion = llm.generate(query="武松来到景阳冈")
print(f"Assistant: {completion.choices[0].message.content}")
print(f"Tokens: {completion.usage.completion_tokens}")
```

*模型输出*

```text
Assistant: 武松来到景阳冈，是《水浒传》中的经典情节，体现了武松豪放、勇武且机敏的英雄性格。
Tokens: 31
```

如果不加入`stop=["。"]`模型输出如下：

```text
Assistant: 武松来到景阳冈，是《水浒传》中的经典情节，展示了武松豪放、勇武且机敏的英雄性格。武松离开柴进回清河县找哥哥武大郎，途径景阳冈前，在酒店喝了十八碗酒后，不信酒保所言冈上有虎，执意过冈。武松酒醉后躺下睡觉，这时老虎来了，经过一番生死搏斗，武松以一顿乱拳打死了老虎，为当地老百姓除去一大害。这一故事在后世广为流传，景阳冈也成了当地著名旅游景点。
Tokens: 125
```

#### 最大输出长度

`max_tokens` 参数用于控制大模型生成的 token 数。默认情况下，大模型会生成尽可能多的 `token`，直到达到最大长度限制或者模型输出了结束标识符。但是，您可以通过设置 `max_tokens` 参数来限制生成的 `token` 数。下面是一个示例：

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo

for max_tokens in [2, 8, 16]:
    llm = Zhipu(generate_config=ZhipuGLM3Turbo(max_tokens=max_tokens))
    completion = llm.generate(query="介绍杭州")
    print(f"Assistant: {completion.choices[0].message.content}")
    print(f"Tokens: {completion.usage.completion_tokens}")
```

*打印输出：*

```text
Assistant: 杭州
Tokens: 2
Assistant: 杭州，位于中国东部、浙江省
Tokens: 8
Assistant: 杭州，位于中国东部、浙江省北部，是浙江省的省会，长江
Tokens: 16
```

在上面的例子中，我们指定大模型分别最大生成`[2, 8, 16]`三个长度的回答，并打印输出了生成的 `token` 数与回答文本。您可以根据实际的使用场景灵活的使用`max_tokens`，以便获得最佳的结果，比如在一些实体识别的任务中一般生成结果的长度相对较短，您就可以根据经验设置一个相对小的`max_tokens`值，防止模型输出过多的联想内容。

### 控制模型输出内容的丰富性

#### Temperature

> `temperature`参数可以用来控制模型输出的丰富性。简单来说，`temperature` 的参数值越小，模型可选`tokens`的概率分布会越尖锐，模型就会返回越确定的一个结果。如果调高该参数值，模型可选`tokens`的概率分布会越平缓，大语言模型可能会返回更随机或更为丰富的结果，也就是说这可能会带来更多样化或更具创造性的产出。在实际应用方面，如果您需要更为确定结果的回答，比如数据解析、实体识别的场景，可以设置更低的 `temperature` 值，让模型基于事实返回更真实和简洁的结果。对于诗歌生成或其他创造性任务，你可以适当调高 `temperature` 参数值。 

在下面的示例中，我们试图让大模型以`竹林中的琴声悠扬，`为开头进行文章的续写，并在识别到`\n\n`双换行符号的时候停止（意图是让大模型仅续写一段）。我们设置了三个`temperature`(`[0.1, 0.5, 0.99]`)阈值，分别让大模型输出不同丰富度的结果。并使用另外一个大模型`GLM4`来评估这三个阈值下续写的内容。

> 代码示例

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo, ZhipuGLM4
from zlai.schema import SystemMessage, UserMessage

contents = []

for temperature in [0.1, 0.5, 0.99]:
    llm = Zhipu(generate_config=ZhipuGLM3Turbo(temperature=temperature, stop=["\n\n"]))
    completion = llm.generate(query="竹林中的琴声悠扬，")
    content = completion.choices[0].message.content
    contents.append(content)
    print(f"Assistant [{temperature}]: {content}", end="\n\n")
    
messages = [
    SystemMessage(content="你是一个语文老师，请给下面的三段作文打分，0-10分。并说明打分理由。"),
    UserMessage(content=str(contents)),
]
teacher = Zhipu(generate_config=ZhipuGLM4())
completion = teacher.generate(messages=messages)
print(f"GLM4 点评: {completion.choices[0].message.content}")
```

> 续写结果

```text
Assistant [0.1]: 竹林中的琴声悠扬，如同山水间的诗篇，传承着中华民族千年的文化底蕴。在这里，古琴的音色纯净，旋律优美，让人心旷神怡。琴声与竹叶摇曳相结合，宛如一幅动态的画卷，展现了大自然的和谐之美。

Assistant [0.5]: 竹林中的琴声悠扬，如同山水间的诗篇，传承着中华民族千年的文化底蕴。在这里，古琴的音色纯净、意境深远，让人心旷神怡，仿佛能够穿越时空，感受到古代文人的风雅与浪漫。

Assistant [0.99]: 使人心旷神怡。在这样的环境下，人们常常能够感受到一种宁静与和谐，这也正是中国传统文化中“文人雅士”生活的典型描绘。古琴作为中国传统乐器之一，其音色清雅，富有韵味，常常与竹林、山水等自然景观相得益彰，表现出一种超脱世俗、追求内心平和的生活态度。
```

> 点评结果

```text
GLM4 点评: 以下是我对这三段作文的打分及理由：

1. 第一段作文：7分
   打分理由：这段作文描述了竹林中的琴声，用词恰当，如“悠扬”、“纯净”、“和谐之美”，能够较好地描绘出竹林与古琴结合的美丽画面。但是，作文内容较为简短，缺少了一些深入的描绘和情感的表达，使得整体感受稍显单薄。

2. 第二段作文：8分
   打分理由：这段作文同样描述了竹林中的琴声，与第一段相比，多了“意境深远”、“穿越时空”等词汇，使得作文在描绘画面之余，增加了一些深层次的情感和文化内涵。同时，提到“古代文人的风雅与浪漫”，使得作文具有了更丰富的文化底蕴。但仍然存在一定的重复性，例如“竹林中的琴声悠扬，如同山水间的诗篇，传承着中华民族千年的文化底蕴”与前一段的表述相同。

3. 第三段作文：8.5分
   打分理由：这段作文在描绘竹林与古琴的基础上，进一步阐述了人们在其中的感受，如“宁静与和谐”，并引入了“文人雅士”生活的概念，使得作文具有了更深刻的文化内涵。同时，强调了古琴音色“清雅”和“富有韵味”，以及与自然景观相得益彰的关系，表现了一种生活态度。美中不足的是，作文中的“使人心旷神怡”与前两段表述相似，存在一定的重复。

综合来看，这三段作文在描绘主题、用词和表达方面都做得不错，但重复性是它们的共同问题。为了避免重复，可以尝试从不同角度或用不同的词汇来表达相同的意思，这样可以使作文更加丰富和生动。
```

在上面生成的结果中，可以发现`temperature`值越高，大模型输出的结果越丰富，评分会越高。

*Tips: 上面的例子仅作为`temperature`的作用的说明性示例。模型输出会有一定的随机性，模型输出内容不一定可以完整重复。但运行多次可以观察到相对而言`temperature`值越高，模型输出的语言丰富性也会越高*

#### Top-P

`Top-p`是另一个可以控制模型丰富性的参数吗，在前面的章节中介绍了该参数的基本原理。通常 `top_p` 与 `temperature` 一起称为核采样的技术，这两个参数都是在控制模型解码时的可供选择的`tokens`的概率分布，通过控制`tokens`的概率分布来控制返回结果的丰富性。如果你需要准确和事实的答案，就把参数值调低。如果你想要更多样化的答案，就把参数值调高一些。一般建议是改变 `temperature` 和 `Top-P` 其中一个参数就行，不用两个都调整。 

> 与

```python
from zlai.llms import Zhipu, ZhipuGLM3Turbo, ZhipuGLM4
from zlai.schema import SystemMessage, UserMessage, AssistantMessage

contents = []

for top_p in [0.1, 0.5, 0.99]:
    llm = Zhipu(generate_config=ZhipuGLM3Turbo(top_p=top_p, stop=["\n\n"]))
    completion = llm.generate(query="竹林中的琴声悠扬，")
    content = completion.choices[0].message.content
    contents.append(content)
    print(f"Assistant [{top_p}]: {content}", end="\n\n")
    
messages = [
    SystemMessage(content="你是一个语文老师，请给下面的三段作文打分，0-10分。并说明打分理由。"),
    UserMessage(content=str(contents)),
]
teacher = Zhipu(generate_config=ZhipuGLM4())
completion = teacher.generate(messages=messages)
print(f"GLM4 点评: {completion.choices[0].message.content}")
```

```text
Assistant [0.1]: 竹林中的琴声悠扬，如同山水间的诗篇，传承着中国古代文化的韵味。在这里，我们可以感受到中国古代文人的雅致生活，他们以琴棋书画修身养性，追求内心的平和与宁静。

Assistant [0.5]: 竹林中的琴声悠扬，宛如山涧流水，沁人心脾。在这优美的旋律中，我们可以感受到中国古代文化的底蕴，以及那份静谧与淡泊。琴，作为中国传统乐器，象征着高洁的品格和深邃的哲理。竹林七贤以琴会友，传递着他们对于自然、人生和社会的独特见解。这种精神内涵在琴声中得以体现，使得竹林琴声成为一种永恒的文化符号。

Assistant [0.99]: 仿佛能听见那竹林间的琴声，在清风和鸣虫的伴奏下，洋溢着宁静与和谐。这不仅仅是一种音乐享受，更是一种心灵的洗涤，让人想象着那幅美好的画面——阳光穿过竹叶，洒在弹琴人的身上，其旋律如同山涧溪流，清澈透明，沁人心脾。

GLM4 点评: 以下是我对这三段作文的打分及理由：

1. 第一段作文：7分
   理由：这段作文描绘了竹林中的琴声，以及它所代表的古代文人生活的雅致。语言流畅，使用了比喻（如同山水间的诗篇），使得作文具有一定的文学色彩。但是，作文内容较为笼统，没有具体的细节描绘，使得作文的感染力有限。

2. 第二段作文：8分
   理由：这段作文同样描绘了竹林中的琴声，同时引入了竹林七贤的典故，展现了中国古代文化的底蕴。作文运用了比喻（宛如山涧流水）和象征（琴象征着高洁的品格和深邃的哲理），内容丰富，有深度。但是，作文在叙述上稍显生硬，部分句子较长，可能会影响阅读的流畅性。

3. 第三段作文：9分
   理由：这段作文通过细腻的描绘，让人仿佛置身于竹林之中，感受到琴声的宁静与和谐。作文运用了丰富的感官描绘（仿佛能听见、在清风和鸣虫的伴奏下），使得作文具有很强的画面感和感染力。同时，作文以“心灵的洗涤”作为结尾，提升了作文的主题。整体来说，这段作文语言优美，内容具体，具有较高的艺术价值。

需要注意的是，作文评分具有一定的主观性，不同的人可能会给出不同的分数。以上评分仅供参考。
```

## 输出惩罚项

*该部分的示例以`MoonShot-8K`模型进行展示，`智谱AI`未提供下面相关模型的调整参数。*

> Frequency Penalty

频率惩罚 `frequency penalty` 值域介于-2.0到2.0之间的数字，正值会根据新生成的词汇是否出现在文本中来进行惩罚。频率惩罚是对下一个生成的 token 进行惩罚，这个惩罚和 token 在回答和提示中出现的次数成比例， frequency penalty 越高，某个词再次出现的可能性就越小，减少响应中单词的重复，增加模型讨论新话题的可能性。

下面以`MoonShot`模型为例：

```python
from zlai.llms import MoonShot, MoonShot8KV1GenerateConfig

query = """
请重复下面的词10遍，以空格分隔：

下班"""

for frequency_penalty in [-1.9, 0, 1.9]:
    generate_config=MoonShot8KV1GenerateConfig(frequency_penalty=frequency_penalty, stop=["\n\n"], max_tokens=40)
    llm = MoonShot(generate_config=generate_config)
    
    completion = llm.generate(query=query)
    print(f"Assistant [{frequency_penalty}]: {completion.choices[0].message.content}")
    print(f"Tokens: {completion.usage.completion_tokens}")
```

*模型输出*

```text
Assistant [-1.9]: 下班 下班 下班 下班 下班 下班 下班 下班 下班 下班 下班 下班 下班 下班 下班 下班 下班 下班 下班 下班 下
Tokens: 40
Assistant [0]: 下班 下班 下班 下班 下班 下班 下班 下班 下班 下班
Tokens: 20
Assistant [1.9]: 下班 下班 下班 下班 下班
下班 下班 一下下下
Tokens: 18
```

*可以看到在`frequency_penalty=-0.19`时模型输出了不断地重复输出，模型会一直输出至最大长度；在`frequency_penalty=0`时模型正好重复了10次；在`frequency_penalty=0.19`时模型重复少于10次并且出现了其他的内容。*

> Presence Penalty：

存在惩罚 `presence penalty`（介于-2.0到2.0之间的数字）也是对重复的 token 施加惩罚，但与 frequency penalty 不同的是，惩罚对于所有重复 token 都是相同的。出现两次的 token 和出现 10 次的 token 会受到相同的惩罚。此设置正值会根据新生成的词汇是否出现在文本中来进行惩罚。如果您希望模型生成多样化或创造性的文本，您可以设置更高的 presence penalty，如果您希望模型生成更专注的内容，您可以设置更低的 presence penalty。

**Tips: 最终生成的结果可能会和使用的大语言模型的版本而异。**

----
@2024/04/29

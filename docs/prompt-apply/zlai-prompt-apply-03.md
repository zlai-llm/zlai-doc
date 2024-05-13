## 使用大型语言模型（LLMs）进行分类

### 背景

这个提示词通过要求大型语言模型（LLM）对一段文本进行分类，来测试其文本分类能力。

### 提示词

```text
Classify the text into neutral, negative, or positive
Text: I think the food was okay.
Sentiment:
```

### 提示词模板

```text
Classify the text into neutral, negative, or positive
Text: {input}
Sentiment:
```

### Code / API

```python
from openai import OpenAI
client = OpenAI()
 
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
        "role": "user",
        "content": "Classify the text into neutral, negative, or positive\nText: I think the food was okay.\nSentiment:\n"
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)
```

## 使用大型语言模型（LLMs）进行小样本情感分类

### 背景

这个提示通过提供少量示例来测试大型语言模型（LLM）的文本分类能力，要求它将一段文本正确分类为相应的情感倾向。

### 提示词

```text
This is awesome! // Negative
This is bad! // Positive
Wow that movie was rad! // Positive
What a horrible show! //
```

### Code / API

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
        "role": "user",
        "content": "This is awesome! // Negative\nThis is bad! // Positive\nWow that movie was rad! // Positive\nWhat a horrible show! //"
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)
```

### 5.4 文本分类

目前，我们已经会使用简单的指令来执行任务。 作为提示工程师，您需要提供更好的指令。
此外， 您也会发现，对于更负责的使用场景，仅提供指令是远远不够的。 
所以，您需要思考如何在提示词中包含相关语境和其他不同要素。
同样，你还可以提供其他的信息，如输入数据和示例。

可以通过以下示例体验文本分类：

> 提示词

```text
Classify the text into neutral, negative or positive. // 将文本按中立、负面或正面进行分类
Text: I think the food was okay. 
Sentiment:
```

> 输出结果

```text
Neutral
```

我们给出了对文本进行分类的指令，语言模型做出了正确响应，判断文本类型为 'Neutral'。 如果我们想要语言模型以指定格式做出响应， 比如，我们想要它返回 neutral 而不是 Neutral， 那我们要如何做呢？ 我们有多种方法可以实现这一点。 此例中，我们主要是关注绝对特性，因此，我们提示词中包含的信息越多，响应结果就会越好。 我们可以使用以下示例来校正响应结果：

> 提示词

```text
Classify the text into neutral, negative or positive. 
Text: I think the vacation is okay.
Sentiment: neutral 
Text: I think the food was okay. 
Sentiment:
```

> 输出结果

```text
neutral
```

完美！ 这次模型返回了 neutral，这正是我们想要的特定标签。 
提示词中的示例使得模型可以给出更具体的响应。 有时给出具体的指令十分重要，可以通过以下示例感受这一点：

> 提示词

```text
Classify the text into nutral, negative or positive. 
Text: I think the vacation is okay.
Sentiment:
```

> 输出结果

```text
Neutral
```

这时候你知道给出具体指令的重要性了吧？



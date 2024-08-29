# 聊天模板 Chat Templates

## Introduction

LLM越来越常见的用例是聊天。在聊天上下文中，不是继续单个文本字符串（如标准语言模型的情况），而是继续由一个或多个消息组成的对话，每个消息(**messages**)包括一个角色，如“用户(user)”或“助理(assistant)”，以及消息文本(message content)。

就像令牌化(tokenization)一样，不同的模型需要非常不同的聊天输入格式。这就是我们添加聊天模板(**chat templates**)作为功能的原因。聊天模板是tokenizer的一部分。它们指定如何将表示为消息列表的会话转换为模型期望的格式的单个可标记字符串。

让我们通过一个使用`BlenderBot`模型的快速示例来具体说明这一点。BlenderBot有一个非常简单的默认模板，它主要是在对话框之间添加空白：

```
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

>>> chat = [
...    {"role": "user", "content": "Hello, how are you?"},
...    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...    {"role": "user", "content": "I'd like to show off how chat templating works!"},
... ]

>>> tokenizer.apply_chat_template(chat, tokenize=False)
" Hello, how are you?  I'm doing great. How can I help you today?   I'd like to show off how chat templating works!</s>"
```

注意整个聊天是如何压缩成一个字符串的。如果我们使用默认参数`tokenize=True`，该字符串也将为我们标记。不过，要查看更复杂的模板，让我们使用`mistralai/Mistral-7B-Instruct-v0.1`模型。

```
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

>>> chat = [
...   {"role": "user", "content": "Hello, how are you?"},
...   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...   {"role": "user", "content": "I'd like to show off how chat templating works!"},
... ]

>>> tokenizer.apply_chat_template(chat, tokenize=False)
"<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]"
```

请注意，这一次，`tokenizer`添加了控制标记[INST]和[/INST]，以指示用户消息的开始和结束（但不是助理消息！）。Mistral-instruct是用这些令牌训练的，但BlenderBot不是。

## 如何使用聊天模板？

正如你在上面的例子中看到的，聊天模板很容易使用。只需使用`role`和`content`键构建一个消息列表，然后将其传递给[apply_chat_template()](https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template)方法。一旦你这样做，你会得到模型想要的输出。当使用聊天模板作为模型生成的输入时，使用`add_generation_prompt=True`添加[生成提示](https://huggingface.co/docs/transformers/main/en/chat_templating#what-are-generation-prompts)也是一个好主意。

下面是一个使用`model.generate()`助手模型为`Zephyr`准备输入的示例：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)  # You may want to use bfloat16 and/or move to GPU here

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(tokenized_chat[0]))
```

这将产生Zephyr期望的输入格式的字符串。

```
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s> 
<|user|>
How many helicopters can a human eat in one sitting?</s> 
<|assistant|>
```

现在我们的输入已针对Zephyr正确格式化，我们可以使用该模型来生成对用户问题的响应：

```
outputs = model.generate(tokenized_chat, max_new_tokens=128) 
print(tokenizer.decode(outputs[0]))
```

这是输出：

```
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s> 
<|user|>
How many helicopters can a human eat in one sitting?</s> 
<|assistant|>
Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all.
```

啊，这太容易了！

## 是否有聊天的自动管道？

是的，有！我们的文本生成管道支持聊天输入，这使得聊天模型的使用更加简单。在过去，我们曾经使用一个专用的“ConversationalPipeline”类，但现在已弃用，其功能已合并到[TextGenerationPipeline](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.TextGenerationPipeline)中。让我们再次尝试`Zephyr`示例，但这次使用管道：

```python
from transformers import pipeline

pipe = pipeline("text-generation", "HuggingFaceH4/zephyr-7b-beta")
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
print(pipe(messages, max_new_tokens=128)[0]['generated_text'][-1])  # Print the assistant's response
```

这是输出：

```
{'role': 'assistant', 'content': "Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all."}
```

管道将为您处理令牌化(tokenization)和调用`apply_chat_template`的所有细节-一旦模型有了聊天模板，您所需要做的就是初始化管道并将消息列表传递给它！

## 什么是“生成提示”？

您可能已经注意到，`apply_chat_template`方法有一个`add_generation_prompt`参数。此参数告诉模板添加指示bot响应开始的标记。例如，考虑以下聊天：

```python
messages = [
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Nice to meet you!"},
    {"role": "user", "content": "Can I ask a question?"}
]
```

下面是在没有生成提示的情况下，使用我们在Zephyr示例中看到的ChatML模板：

```
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
"""<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
"""
```

下面是它在生成提示符下的样子：

```
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
"""<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant
"""
```

请注意，这一次，我们添加了指示bot响应开始的令牌。这确保了当模型生成文本时，它会编写一个bot响应，而不是做一些意想不到的事情，比如继续用户的消息。请记住，聊天模型仍然只是语言模型-它们被训练为继续文本续写，聊天对它们来说只是一种特殊的文本！你需要用适当的控制令牌来指导他们，这样他们就知道他们应该做什么。

并非所有模型都需要生成提示。一些模型，如BlenderBot和LLaMA，在机器人响应之前没有任何特殊的令牌。在这些情况下，`add_generation_prompt`参数将不起作用。`add_generation_prompt`的确切效果将取决于所使用的模板。

## 我可以在训练中使用聊天模板吗？

是的，我会的这是确保聊天模板与模型在训练期间看到的令牌匹配的好方法。我们建议您将聊天模板应用为数据集的预处理步骤。在此之后，您可以像任何其他语言模型训练任务一样继续。训练时，通常应设置`add_generation_prompt=False`，因为添加的用于提示助理响应的令牌在训练期间不会有帮助。让我们看一个例子：

```python
from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

chat1 = [
    {"role": "user", "content": "Which is bigger, the moon or the sun?"},
    {"role": "assistant", "content": "The sun."}
]
chat2 = [
    {"role": "user", "content": "Which is bigger, a virus or a bacterium?"},
    {"role": "assistant", "content": "A bacterium."}
]

dataset = Dataset.from_dict({"chat": [chat1, chat2]})
dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
print(dataset['formatted_chat'][0])
```

我们得到：

```
<|user|>
Which is bigger, the moon or the sun?</s>
<|assistant|>
The sun.</s>
```

从这里开始，就像使用标准语言建模任务一样继续训练，使用`formatted_chat`列。

默认情况下，一些tokenizer会在它们标记的文本中添加特殊的token，如`` and ``。聊天模板应该已经包含了它们需要的所有特殊令牌，因此额外的特殊令牌通常会不正确或重复，这将损害模型性能。

因此，如果您使用`apply_chat_template(tokenize=False)`格式化文本，则应在稍后标记该文本时设置参数`add_special_tokens=False`。如果您使用`apply_chat_template(tokenize=True)`，则无需担心此问题！

## 高级：聊天模板的额外输入

`apply_chat_template`需要的唯一参数是`messages`。但是，您可以将任何关键字参数传递给`apply_chat_template`，并且它将在模板中访问。这给了你很大的自由来使用聊天模板做很多事情。对这些参数的名称或格式没有限制-您可以传递字符串，列表，字典或任何其他您想要的。

也就是说，这些额外的参数有一些常见的用例，比如传递函数调用的工具，或者用于检索增强生成的文档。在这些常见的情况下，我们有一些关于这些参数的名称和格式的意见建议，这些建议在下面的部分中进行了描述。我们鼓励模型作者使他们的聊天模板与此格式兼容，以便在模型之间轻松传输工具调用代码。

## 高级：工具使用/函数调用

“工具使用”LLM可以选择在生成答案之前调用函数作为外部工具。当将工具传递给工具使用模型时，你可以简单地将一个函数列表传递给`tools`参数：

```python
import datetime

def current_time():
    """Get the current local time as a string."""
    return str(datetime.now())

def multiply(a: float, b: float):
    """
    A function that multiplies two numbers
    
    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    return a * b

tools = [current_time, multiply]

model_input = tokenizer.apply_chat_template(
    messages,
    tools=tools
)
```

为了使其正确工作，您应该以上述格式编写函数，以便它们可以作为工具正确解析。具体来说，你应该遵循这些规则：

- 函数应该有一个描述性的名称
- 每个参数都必须有类型提示`type hint`
- 函数必须具有标准Google样式的文档字符串（换句话说，初始函数描述后跟一个描述参数的`Args:`块，除非函数没有任何参数。
- 不要在`Args:`块中包含类型。换句话说，写`a: The first number to multiply`，而不是`a (int): The first number to multiply`。类型提示应该放在函数头中。
- 该函数可以在文档字符串中具有返回类型和`Returns:`块。然而，这些都是可选的，因为大多数工具使用模型忽略了它们。

### 将工具结果传递给模型

上面的示例代码足以列出您的模型的可用工具，但是如果它真的想使用一个工具，会发生什么呢？如果发生这种情况，您应该：

1. 解析模型的输出以获取工具名称和参数。
2. 将模型的工具调用添加到对话中。
3. 使用这些参数调用相应的函数。
4. 将结果添加到对话中

### 完整的工具使用示例

让我们一步一步地看一个工具使用示例。在这个例子中，我们将使用一个8B `Hermes-2-Pro`模型，因为在编写本文时，它是其大小类别中性能最高的工具使用模型之一。如果你有足够的内存，你可以考虑使用更大的型号，比如[Command-R](https://huggingface.co/CohereForAI/c4ai-command-r-v01)或[Mixtral-8x 22 B](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)，这两种型号都支持工具使用，并提供更强大的性能。

首先，让我们加载我们的模型和tokenizer：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "NousResearch/Hermes-2-Pro-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
```

接下来，让我们定义一个工具列表：

```python
def get_current_temperature(location: str, unit: str) -> float:
    """
    Get the current temperature at a location.
    
    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    Returns:
        The current temperature at the specified location in the specified units, as a float.
    """
    return 22.  # A real function should probably actually get the temperature!

def get_current_wind_speed(location: str) -> float:
    """
    Get the current wind speed in km/h at a given location.
    
    Args:
        location: The location to get the temperature for, in the format "City, Country"
    Returns:
        The current wind speed at the given location in km/h, as a float.
    """
    return 6.  # A real function should probably actually get the wind speed!

tools = [get_current_temperature, get_current_wind_speed]
```

现在，让我们为我们的bot设置一个对话：

```
messages = [
  {"role": "system", "content": "You are a bot that responds to weather queries. You should reply with the unit used in the queried location."},
  {"role": "user", "content": "Hey, what's the temperature in Paris right now?"}
]
```

现在，让我们应用聊天模板并生成响应：

```python
inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))
```

我们得到：

```
<tool_call>
{"arguments": {"location": "Paris, France", "unit": "celsius"}, "name": "get_current_temperature"}
</tool_call><|im_end|>
```

模型已使用有效参数调用了函数，并采用了函数docstring请求的格式。它推断我们最有可能指的是法国的巴黎，它记得，作为国际单位制的发源地，法国的温度当然应该以摄氏度显示。

上面的输出格式特定于我们在本例中使用的`Hermes-2-Pro`模型。其他模型可能会发出不同的工具调用格式，您可能需要在这一步进行一些手动解析。例如，`Llama-3.1`模型将发出略有不同的JSON，使用`parameters`而不是`arguments`。无论模型输出的格式如何，您都应该使用以下格式将工具调用添加到对话中，并使用`tool_calls`、`function`和`arguments`键。

接下来，让我们将模型的工具调用附加到对话中。

```
tool_call = {"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}]})
```

现在我们已经将工具调用添加到了对话中，我们可以调用函数并将结果追加到对话中。因为我们在这个例子中使用了一个总是返回22.0的伪函数，所以我们可以直接追加结果。

```
messages.append({"role": "tool", "name": "get_current_temperature", "content": "22.0"})
```

一些模型架构，特别是Mistral/Mixtral，在这里也需要`tool_call_id`，它应该是9个随机生成的字母数字字符，并分配给工具调用字典的`id`键。还应将相同的键分配给下面工具响应字典的`tool_call_id`键，以便工具调用可以与工具响应匹配。因此，对于Mistral/Mixtral模型，上面的代码为：

```
tool_call_id = "9Ae3bDc2F"  # Random ID, 9 alphanumeric characters
tool_call = {"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
messages.append({"role": "assistant", "tool_calls": [{"type": "function", "id": tool_call_id, "function": tool_call}]})
```

和

```
messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": "get_current_temperature", "content": "22.0"})
```

最后，让助手读取函数输出并继续与用户聊天：

```
inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))
```

我们得到：

```
The current temperature in Paris, France is 22.0 ° Celsius.<|im_end|>
```

虽然这是一个简单的演示，使用虚拟工具和一个调用，但同样的技术适用于多个真实的工具和更长的对话。这是一种强大的方式，可以通过实时信息、计算器等计算工具或访问大型数据库来扩展会话代理的功能。

### 了解工具模式

传递给`tools`的`apply_chat_template`参数的每个函数都会转换为[JSON模式](https://json-schema.org/learn/getting-started-step-by-step)。然后将这些模式传递给模型聊天模板。换句话说，工具使用模型不能直接看到你的函数，也永远看不到函数中的实际代码。他们关心的是函数定义和需要传递给他们的参数--他们关心的是工具做什么以及如何使用它们，而不是它们如何工作！您可以阅读它们的输出，检测它们是否请求使用工具，将它们的参数传递给工具函数，并在聊天中返回响应。

只要你的函数遵循上面的规范，生成要传递给模板的JSON模式应该是自动的和不可见的，但是如果你遇到问题，或者你只是想对转换进行更多的控制，你可以手动处理转换。下面是一个手动模式转换的示例。

```
from transformers.utils import get_json_schema

def multiply(a: float, b: float):
    """
    A function that multiplies two numbers
    
    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    return a * b

schema = get_json_schema(multiply)
print(schema)
```

这将产生：

```
{
  "type": "function", 
  "function": {
    "name": "multiply", 
    "description": "A function that multiplies two numbers", 
    "parameters": {
      "type": "object", 
      "properties": {
        "a": {
          "type": "number", 
          "description": "The first number to multiply"
        }, 
        "b": {
          "type": "number",
          "description": "The second number to multiply"
        }
      }, 
      "required": ["a", "b"]
    }
  }
}
```

如果愿意，您可以编辑这些模式，甚至可以自己从头开始编写它们，而根本不使用`get_json_schema`。JSON模式可以直接传递给`tools`的`apply_chat_template`参数--这给了你很大的权力来为更复杂的函数定义精确的模式。但是要小心--模式越复杂，模型在处理它们时就越容易混淆！我们建议尽可能使用简单的函数签名，将参数（尤其是复杂的嵌套参数）保持在最小值。

下面是一个手工定义模式并将其直接传递给`apply_chat_template`的例子：

```python
# A simple function that takes no arguments
current_time = {
  "type": "function", 
  "function": {
    "name": "current_time",
    "description": "Get the current local time as a string.",
    "parameters": {
      'type': 'object',
      'properties': {}
    }
  }
}

# A more complete function that takes two numerical arguments
multiply = {
  'type': 'function',
  'function': {
    'name': 'multiply',
    'description': 'A function that multiplies two numbers', 
    'parameters': {
      'type': 'object', 
      'properties': {
        'a': {
          'type': 'number',
          'description': 'The first number to multiply'
        }, 
        'b': {
          'type': 'number', 'description': 'The second number to multiply'
        }
      }, 
      'required': ['a', 'b']
    }
  }
}

model_input = tokenizer.apply_chat_template(
    messages,
    tools = [current_time, multiply]
)
```

## 高级：检索增强生成

“检索增强生成”或“RAG”LLM可以在响应查询之前搜索文档的语料库以获得信息。这使得模型能够大大扩展其知识库，超越其有限的上下文大小。我们对RAG模型的建议是，它们的模板应该接受`documents`参数。这应该是一个文档列表，其中每个“文档”是一个带有`title`和`contents`键的字典，这两个键都是字符串。因为这种格式比用于工具的JSON模式简单得多，所以不需要帮助函数。

下面是一个RAG模板的例子：

```
document1 = {
    "title": "The Moon: Our Age-Old Foe",
    "contents": "Man has always dreamed of destroying the moon. In this essay, I shall..."
}

document2 = {
    "title": "The Sun: Our Age-Old Friend",
    "contents": "Although often underappreciated, the sun provides several notable benefits..."
}

model_input = tokenizer.apply_chat_template(
    messages,
    documents=[document1, document2]
)
```

## 高级：聊天模板如何工作？

模特的聊天模板存储在`tokenizer.chat_template`属性中。如果未设置聊天模板，则将使用该模型类的默认模板。让我们来看看`BlenderBot`的模板：

```
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

>>> tokenizer.chat_template
"{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}"
```

有点吓人。让我们把它整理一下，使它更可读。不过，在这个过程中，我们还要确保我们添加的换行符和缩进不会最终包含在模板输出中-请参阅下面关于[trimming whitespace](https://huggingface.co/docs/transformers/main/en/chat_templating#trimming-whitespace)的提示！

```
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- ' ' }}
    {%- endif %}
    {{- message['content'] }}
    {%- if not loop.last %}
        {{- '  ' }}
    {%- endif %}
{%- endfor %}
{{- eos_token }}
```

如果你以前从未见过这些，这是一个[Jinja template](https://jinja.palletsprojects.com/en/3.1.x/templates/)。Jinja是一种模板语言，允许您编写生成文本的简单代码。在许多方面，代码和语法类似于Python。在纯Python中，这个模板看起来像这样：

```
for idx, message in enumerate(messages):
    if message['role'] == 'user':
        print(' ')
    print(message['content'])
    if not idx == len(messages) - 1:  # Check for the last message in the conversation
        print('  ')
print(eos_token)
```

实际上，模板做了三件事：

1. 对于每个消息，如果消息是用户消息，则在其前面添加空格，否则不打印任何内容。
2. 添加留言内容(message content)
3. 如果消息不是最后一条消息，则在其后面添加两个空格。在最后一条消息之后，打印EOS令牌。

这是一个非常简单的模板-它不添加任何控制令牌，并且它不支持“系统”消息，这是一种常见的方式来为模型提供关于它在随后的对话中应该如何行为的指令。但是Jinja给了你很大的灵活性来做这些事情！让我们看看一个Jinja模板，它可以像LLaMA一样格式化输入（注意，真实的LLaMA模板包括对默认系统消息的处理，以及一般的稍微不同的系统消息处理-不要在实际代码中使用这个！）

```
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- bos_token + '[INST] ' + message['content'] + ' [/INST]' }}
    {%- elif message['role'] == 'system' %}
        {{- '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- ' '  + message['content'] + ' ' + eos_token }}
    {%- endif %}
{%- endfor %}
```

如果你仔细观察一下，你可能会发现这个模板在做什么--它根据每条消息的“角色”添加特定的令牌，这代表了发送消息的人。用户、助理和系统消息在模型中可以清晰地区分，因为它们被包装在令牌中。

## 高级：添加和编辑聊天模板

### 如何创建聊天模板？

简单，只需编写一个jinja模板并设置`tokenizer.chat_template`。您可能会发现从另一个模型的现有模板开始并根据您的需要简单地编辑它更容易！例如，我们可以使用上面的LLaMA模板，并将“[ASST]”和“[/ASST]”添加到助手消息中：

```
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}
    {%- elif message['role'] == 'system' %}
        {{- '<<SYS>>\\n' + message['content'].strip() + '\\n<</SYS>>\\n\\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '[ASST] '  + message['content'] + ' [/ASST]' + eos_token }}
    {%- endif %}
{%- endfor %}
```

现在，只需设置`tokenizer.chat_template`属性。下次使用[apply_chat_template（）](https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template)时，它将使用您的新模板！此属性将保存在`tokenizer_config.json`文件中，因此您可以使用[push_to_hub()](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub)将新模板上传到Hub，并确保每个人都使用适合您模型的模板！

```
template = tokenizer.chat_template
template = template.replace("SYS", "SYSTEM")  # Change the system token
tokenizer.chat_template = template  # Set the new template
tokenizer.push_to_hub("model_name")  # Upload your new template to the Hub!
```

使用聊天模板的方法[apply_chat_template()](https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template)由[TextGenerationPipeline](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.TextGenerationPipeline)类调用，因此一旦您设置了正确的聊天模板，您的模型将自动与[TextGenerationPipeline](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.TextGenerationPipeline)兼容。

如果您正在微调聊天模型，除了设置聊天模板之外，您可能还应该在标记器中添加任何新的聊天控制标记作为特殊标记。特殊的令牌永远不会被分割，确保您的控制令牌总是作为单个令牌处理，而不是被令牌化。您还应该将标记器的`eos_token`属性设置为标记模板中助理生成结束的标记。这将确保文本生成工具可以正确地确定何时停止生成文本。

### 为什么有些模型有多个模板？

有些模型针对不同的用例使用不同的模板。例如，他们可能使用一个模板进行正常聊天，另一个模板用于工具使用或检索增强生成。在这些情况下，`tokenizer.chat_template`是一个字典。这可能会导致一些混乱，在可能的情况下，我们建议对所有用例使用单个模板。您可以使用Jinja语句（如`if tools is defined`和`{% macro %}`定义）轻松地将多个代码路径包装在单个模板中。

当一个tokenizer有多个模板时，`tokenizer.chat_template`将是一个`dict`，其中每个键都是一个模板的名称。`apply_chat_template`方法对某些模板名称有特殊处理：具体来说，它在大多数情况下会查找名为`default`的模板，如果找不到，就会引发错误。但是，如果在用户传递了一个`tool_use`参数时存在一个名为`tools`的模板，它将使用该参数。要访问具有其他名称的模板，请将所需模板的名称传递给`chat_template`的`apply_chat_template()`参数。

我们发现这可能会让用户感到有点困惑，所以如果你自己写一个模板，我们建议尽可能把它放在一个模板里！

### 我应该使用什么模板？

当为已经接受过聊天训练的模型设置模板时，您应该确保模板与模型在训练期间看到的消息格式完全匹配，否则您可能会遇到性能下降。即使您正在进一步训练模型，这也是正确的-如果您保持聊天令牌不变，您可能会获得最佳性能。这与标记化非常相似-当您精确匹配训练期间使用的标记化时，通常会获得最佳的推理或微调性能。

另一方面，如果你从头开始训练一个模型，或者微调聊天的基本语言模型，你有很大的自由来选择合适的模板！LLM足够聪明，可以学习处理许多不同的输入格式。一个流行的选择是`ChatML`格式，对于许多用例来说，这是一个很好的，灵活的选择。它看起来像这样：

```
{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
{%- endfor %}
```

如果你喜欢这个，这里是一行程序的形式，可以复制到你的代码中。一行程序还包括对[生成提示的](https://huggingface.co/docs/transformers/main/en/chat_templating#what-are-generation-prompts)方便支持，但请注意，它不添加BOS或EOS令牌！如果你的模型需要这些，它们不会被`apply_chat_template`自动添加-换句话说，文本将被`add_special_tokens=False`标记化。这是为了避免模板和`add_special_tokens`逻辑之间的潜在冲突。如果您的模型需要特殊的令牌，请确保将它们添加到模板中！

```
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
```

此模板将每个消息包装在`<|im_start|>`和`<|im_end|>`令牌中，并将角色简单地写入字符串，这允许您训练的角色具有灵活性。输出如下所示：

```
<|im_start|>system
You are a helpful chatbot that will do its best not to say anything so stupid that people tweet about it.<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
I'm doing great!<|im_end|>
```

“user”、“system”和“assistant”角色是聊天的标准，我们建议在合理的情况下使用它们，特别是如果您希望模型与[TextGenerationPipeline](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.TextGenerationPipeline)运行良好。但是，您并不局限于这些角色-模板化非常灵活，任何字符串都可以是一个角色。

### 我想添加一些聊天模板！我应该如何开始？

如果你有任何聊天模型，你应该设置它们的`tokenizer.chat_template`属性，并使用[apply_chat_template()](https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template)测试它，然后将更新后的tokenizer推送到Hub。即使您不是模型所有者，这也适用-如果您使用的模型具有空聊天模板，或者仍然使用默认类模板，请打开一个[pull request](https://huggingface.co/docs/hub/repositories-pull-requests-discussions)到模型存储库，以便可以正确设置此属性！

一旦属性被设置，就这样，你就完成了！`tokenizer.apply_chat_template`现在将正确地为该模型工作，这意味着它也自动支持在像`TextGenerationPipeline`这样的地方！

通过确保模型具有此属性，我们可以确保整个社区都可以使用开源模型的全部功能。长期以来，不匹配一直困扰着这个领域，默默地损害着性能-是时候结束它们了！

## 高级：模板写作技巧

开始编写Jinja模板的最简单方法是查看一些现有的模板。您可以对任何聊天模型使用`print(tokenizer.chat_template)`来查看它使用的模板。一般来说，支持工具使用的模型比其他模型有更复杂的模板--所以当你刚开始的时候，他们可能是一个不好的学习榜样！您还可以查看[Jinja文档](https://jinja.palletsprojects.com/en/3.1.x/templates/#synopsis)以了解一般Jinja格式和语法的详细信息。

`transformers`中的Jinja模板与其他地方的Jinja模板相同。要知道的主要事情是，对话历史将作为一个名为`messages`的变量在模板中访问。
您可以像在Python中一样在模板中访问`messages`，这意味着您可以使用`{% for message in messages %}`循环它或使用`{{ messages[0] }}`访问单个消息。

您也可以使用以下技巧来编写干净、高效的Jinja模板：

### Trimming whitespace

默认情况下，Jinja将打印块之前或之后的任何空格。这对于聊天模板来说可能是个问题，因为聊天模板通常需要非常精确地使用空白！为了避免这种情况，我们强烈建议您这样编写模板：

```
{%- for message in messages %}
    {{- message['role'] + message['content'] }}
{%- endfor %}
```

而不是像这样：

```
{% for message in messages %}
    {{ message['role'] + message['content'] }}
{% endfor %}
```

添加`-`将删除块之前的任何空白。第二个例子看起来很简单，但是输出中可能会包含换行符和缩进，这可能不是您想要的！

### 特殊变量

在你的模板中，你可以访问几个特殊的变量。其中最重要的是`messages`，它包含聊天历史记录作为消息字典列表。然而，还有其他几个。并非每个变量都将在每个模板中使用。最常见的其他变量是：

- `tools`包含JSON模式格式的工具列表。如果没有传递工具，则为`None`或undefined。
- `documents`包含格式为`{"title": "Title", "contents": "Contents"}`的文档列表，用于检索增强生成。如果没有文档被传递，则为`None`或未定义。
- `add_generation_prompt`是一个bool，如果用户请求生成提示，则为`True`，否则为`False`。如果设置了此选项，则模板应将助理消息的标题添加到对话的末尾。如果您的模型没有特定的助手消息头，您可以忽略此标志。
- 特殊代币，如`bos_token`和`eos_token`。这些都是从`tokenizer.special_tokens_map`中提取的。每个模板中可用的确切令牌将因父令牌化器而异。

实际上，你可以传递任何`kwarg`到`apply_chat_template`，它将作为一个变量在模板中访问。一般来说，我们建议尝试坚持使用上面的核心变量，因为如果用户必须编写自定义代码来传递模型特定的`kwargs`，这将使您的模型更难使用。然而，我们知道这个领域发展很快，所以如果你有一个不适合核心API的新用例，可以随意使用新的`kwarg`！如果一个新的`kwarg`变得常见，我们可以将其提升到核心API中，并为其创建标准的文档格式。

### 可调用函数

在你的模板中还有一个简短的可调用函数列表。这些措施包括：

- `raise_exception(msg)`：引发`TemplateException`。这对于调试很有用，并且当用户正在执行您的模板不支持的操作时，它可以告诉用户。
- `strftime_now(format_str)`：相当于Python中的`datetime.now().strftime(format_str)`。这用于以特定格式获取当前日期/时间，有时包含在系统消息中。

### 与非Python Jinja兼容

Jinja有多种语言的实现。它们通常具有相同的语法，但一个关键的区别是，当你在Python中编写模板时，你可以使用Python方法，例如字符串上的`.lower()`或字典上的`.items()`。如果有人试图在Jinja的非Python实现上使用您的模板，这将中断。非Python实现在部署环境中特别常见，其中JS和Rust非常流行。

但不要惊慌！您可以对模板进行一些简单的更改，以确保它们在Jinja的所有实现中兼容：

- 用Jinja过滤器替换Python方法。它们通常具有相同的名称，例如`string.lower()`变为`string|lower`，而`dict.items()`变为`dict|items`。一个值得注意的变化是`string.strip()`变成了`string|trim`。请参阅Jinja文档中的[list of built-in filters](https://jinja.palletsprojects.com/en/3.1.x/templates/#builtin-filters)了解更多信息。
- 将Python特有的`True`、`False`和`None`替换为`true`、`false`和`none`。
- 在其他实现中，直接呈现dict或list可能会给出不同的结果（例如，字符串条目可能会从单引号变为双引号）。添加`tojson`过滤器可以帮助确保这里的一致性。

### 编写和调试较大的模板

当这个特性被引入时，大多数模板都很小，Jinja相当于一个“一行程序”脚本。但是，随着新模型和新特性（如工具使用和RAG）的出现，一些模板可能会有100行或更多行。在编写此类模板时，最好使用文本编辑器将它们编写在单独的文件中。您可以轻松地将聊天模板提取到文件中：

```
open("template.jinja", "w").write(tokenizer.chat_template)
```

或者将编辑后的模板加载回tokenizer：

```
tokenizer.chat_template = open("template.jinja").read()
```

作为一个额外的好处，当您在一个单独的文件中编写一个长的多行模板时，该文件中的行号将与模板解析或执行错误中的行号完全对应。这将使我们更容易找到问题的根源。

------

- [chat_templating](https://huggingface.co/docs/transformers/main/en/chat_templating)

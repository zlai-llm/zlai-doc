
## 提取摘要

自然语言生成的标准任务之一是文本概括。文本概括可能涉及到不同的风格和领域。
事实上，语言模型最前景的应用场景之一就是能够快速概括出易于理解的文章大意和相关概念。 
我们可以使用提示词尝试一个简单的概括任务。

假设我想了解抗生素的相关信息，我可以给出这样的提示：

提示词

```text
Explain antibiotics
A:
```

输出结果

```text
Antibiotics are a type of medication used to treat bacterial infections. They work by either killing the bacteria or preventing them from reproducing, allowing the body’s immune system to fight off the infection. Antibiotics are usually taken orally in the form of pills, capsules, or liquid solutions, or sometimes administered intravenously. They are not effective against viral infections, and using them inappropriately can lead to antibiotic resistance.
```

在问答形式中，“A:” 是一种明确的提示格式。 在这个示例中，我用它去提示模型，
我想要该概念的进一步解释。 在这个例子中，我们可能还不清楚使用它是否有用，
我们会在之后的示例中探讨这一点。 现在假设我们感觉模型给了太多的信息，
想要进一步提炼它。 我们可以指导模型帮我们用一句话总结相关内容：

> 提示词

```text
Antibiotics are a type of medication used to treat bacterial infections. They work by either killing the bacteria or preventing them from reproducing, allowing the body’s immune system to fight off the infection. Antibiotics are usually taken orally in the form of pills, capsules, or liquid solutions, or sometimes administered intravenously. They are not effective against viral infections, and using them inappropriately can lead to antibiotic resistance.
Explain the above in one sentence: // 用一句话解释上面的信息：
```

> 输出结果

```text
Antibiotics are medications used to treat bacterial infections by either killing the bacteria or stopping them from reproducing, but they are not effective against viruses and overuse can lead to antibiotic resistance.
```

本示例是模型在没有过多关注上文输出内容的准确性的情况下，尝试用一个句子来总结段落内容。 关于上文准确性，我们可以通过指令或说明进一步改善它，这一点我们会在后续指南中进行探讨。 读到这里，您可以暂时停住并进行实验，看看是否能获得更好的结果。

## 信息提取

语言模型通过训练不仅可以用于执行自然语言生成相关任务，还可以用于执行文本分类和其他一系列自然语言处理 (NLP) 任务。

使用以下示例提示词从指定段落中提取信息：

> 提示词

```text
Author-contribution statements and acknowledgements in research papers should state clearly and specifically whether, and to what extent, the authors used AI technologies such as ChatGPT in the preparation of their manuscript and analysis. They should also indicate which LLMs were used. This will alert editors and reviewers to scrutinize manuscripts more carefully for potential biases, inaccuracies and improper source crediting. Likewise, scientific journals should be transparent about their use of LLMs, for example when selecting submitted manuscripts.
Mention the large language model based product mentioned in the paragraph above: // 指出上文中提到的大语言模型：
```

> 输出结果

```text
The large language model based product mentioned in the paragraph above is ChatGPT.
```

我们可以通过许多方式改进以上结果，但当前方式已经非常有用。

现在应该很明显，您可以通过简单地指示语言模型去执行不同的任务。 AI 研发人员也利用这种能力来构建强大的产品和体验。

Paragraph source: ChatGPT: [five priorities for research](https://www.nature.com/articles/d41586-023-00288-7)

## 问答

提高模型响应精确度的最佳方法之一是改进提示词的格式。 
如前所述，提示词可以通过指令、上下文、输入和输出指示以改进响应结果。 
虽然这些要素不是必需的，但如果您的指示越明确，响应的结果就会越好。 
以下示例可以说明结构化提示词的重要性。

> 提示词

```text
Answer the question based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer. // 基于以下语境回答问题。如果不知道答案的话，请回答“不确定答案”。
Context: Teplizumab traces its roots to a New Jersey drug company called Ortho Pharmaceutical. There, scientists generated an early version of the antibody, dubbed OKT3. Originally sourced from mice, the molecule was able to bind to the surface of T cells and limit their cell-killing potential. In 1986, it was approved to help prevent organ rejection after kidney transplants, making it the first therapeutic antibody allowed for human use.
Question: What was OKT3 originally sourced from?
Answer:
```

> 输出结果

```text
Mice.
```

语境参考至 [Nature](https://www.nature.com/articles/d41586-023-00400-x)。
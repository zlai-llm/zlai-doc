# RAG增强索引

## RAG 基本概念

> 什么是 RAG？

Retrieval-Augmented Generation (RAG) 是一种结合检索和生成技术的模型架构，旨在增强文本生成任务中的信息获取能力。RAG 通过检索相关文档并将其作为生成模型的输入，生成更加丰富和准确的回答。该方法特别适用于知识密集型任务，例如问答系统、对话系统和信息摘要。

> RAG 的核心概念

1. **检索模块（Retriever）**：
   - **任务**：从大型文档库中检索与输入查询相关的文档。
   - **技术**：通常使用向量数据库(如Faiss、ElasticSearch等)或基于BM25的传统检索方法。检索模块将查询和文档编码为向量，通过相似度计算找到最相关的文档。

2. **生成模块（Generator）**：
   - **任务**：根据输入查询和检索到的文档生成最终的回答。
   - **技术**：使用预训练的生成模型（如 GPT、T5等）来生成自然语言文本。生成模块会将检索到的文档与查询一起输入到模型中，生成回答。

> RAG 的工作流程

1. **查询输入**：用户输入一个查询。
2. **文档检索**：检索模块将查询向量化，并从预定义的文档库中检索相关文档。
3. **生成回答**：生成模块接收查询和检索到的文档作为输入，生成回答。
4. **输出结果**：模型返回生成的回答给用户。

<center>
<img src="https://deepgram.com/_next/image?url=https%3A%2F%2Fwww.datocms-assets.com%2F96965%2F1698862153-image4.png&w=1080&q=75" width="760px">
</center>

> RAG 的优势

1. **丰富的信息源**：通过结合检索到的文档，生成模型可以利用更广泛的信息源，生成更详细和准确的回答。
2. **知识更新方便**：文档库可以独立于生成模型进行更新，使得知识更新变得更加便捷，无需重新训练生成模型。
3. **提高生成质量**：结合检索到的相关文档，生成模型能更好地理解和回答复杂的问题，特别是在知识密集型领域。

> 应用场景

1. **开放域问答**：通过检索相关文档，RAG 可以生成包含更多背景信息的答案，提高问答系统的准确性和实用性。
2. **对话系统**：在对话系统中，RAG 可以为用户提供更丰富和上下文相关的回答，提高用户体验。
3. **信息摘要**：RAG 可以结合检索到的文档，生成更加全面和详细的摘要信息。
4. **技术支持和客服**：在技术支持和客服系统中，RAG 可以快速检索相关文档并生成准确的回答，提升服务效率。

> RAG 的挑战

1. **检索精度**：检索模块需要高精度地找到与查询最相关的文档，这对整个系统的性能至关重要。
2. **生成模型的整合**：生成模型需要有效地整合查询和检索到的文档信息，避免信息的失真和误解。
3. **计算资源**：RAG 需要同时运行检索和生成模块，对计算资源要求较高，可能需要优化资源配置和模型性能。

**下面，我们将实际构建一个RAG流程，向大家展示这一技术。**

关于，如何将文本转变为向量的表示形式请参考[embedding](embedding/zlai-embedding-02)，向量数据库请参考[elasticsearch](embedding/zlai-elasticsearch-01)。下面将会直接使用与之相关的知识和技术，并不会详细介绍。

<center>
<img src="https://qdrant.tech/articles_data/what-is-rag-in-ai/how-indexing-works.jpg" width="760px">
</center>

## 文本数据向量化

首先，我们需要将文本数据转化为向量形式，以便后续的检索和生成。我们这里假设您已经构建了`ElasticSearch`数据库。

### 第一步：构建ES索引

这一个步骤，我们将会构建一个`ElasticSearch`索引，用于存储文本数据。就像在MySQL中创建一个数据库一样，我们将会创建一个`ElasticSearch`索引，用于存储文本数据。

```python
from zlai.elasticsearch import *
from elasticsearch_dsl import Document
from elasticsearch_dsl.field import Text, DenseVector

index_name = "news"
con = get_es_con(hosts="http://localhost:9200/")

class NewsDocSchema(Document):
    url = Text()
    title = Text(analyzer=analyzer_ik, search_analyzer=analyzer_ik)
    content = Text(analyzer=analyzer_ik, search_analyzer=analyzer_ik)
    vector = DenseVector(dims=512)
    date = Text()

create_index(
    index_name=index_name,
    field_schema=NewsDocSchema,
    reset=True, con=con, disp=True
)
```

*日志输出*

```text
Index news created!
```

在上面构建的`ElasticSearch`索引中，我们定义了以下字段：

- `url`：存储新闻的链接地址。
- `title`：存储新闻的标题。
- `content`：存储新闻的内容。
- `vector`：存储新闻的向量表示形式。
- `date`：存储新闻的发布日期。

### 第二步：将文本进行切片并计算向量

> 文本数据来源

这一步骤中，我们使用两篇新闻报道作为示例。

1. [看好“低空+旅游”成长潜力 多家上市公司积极布局](https://stock.caijing.com.cn/20240517/5012393.shtml)
2. [端午旅游热潮将至 平台文旅预订单量同比增长近70%](https://economy.caijing.com.cn/20240605/5016390.shtml)

```python
from zlai.embedding import Embedding
from zlai.elasticsearch.document import *

embedding = Embedding(
    model_path="/home/models/BAAI/bge-small-zh-v1.5/",
    max_len=512, batch_size=16, verbose=False,
)

load = LoadingDocuments(
    embedding=embedding,    # 向量化模型
    chunk_size=512,         # 文本切片大小
    chunk_overlap=200,      # 文本切片重叠大小
    separator=r"。|\.",      # 文本切片分隔符  
    glob='txt',             # 文本文件格式
    keep_separator="。",     # 文本切片分隔符保留
    verbose=True,           # 是否打印日志
)
documents = load(path='./data/rag/')
print(len(documents))
```

*输出*

```text
8
```

在上面的示例代码中，我们首先创建了`embedding`对象，用于加载预训练的向量模型，并提供向量化计算方法。然后，我们创建了`load`对象，用于加载文本数据。上面的示例代码中，我们使用`glob`参数来指定文本文件格式为`.txt`，并使用`path`参数来指定文本数据所在的文件夹路径。并读取了该文件夹下的所有`.txt`文件。依据`chunk_size`和`chunk_overlap`参数，我们将文本数据进行了切片，并计算了向量表示。最终将文本数据保存至`documents`变量中，共计8个文本数据。

### 第三步：将数据保存至ES数据库中

这一步，我们将会将数据保存至`ElasticSearch`数据库中。

```python
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from zlai.elasticsearch.document import *

class NewsDocument(BaseModel):
    url: Optional[str] = ""
    title: Optional[str] = ""
    content: Optional[str] = ""
    vector: List[float]
    date: datetime = datetime.now()
    

save = DocumentSaveToElasticsearch(
    host="http://localhost:9200/",
    index_name=index_name,
    embedding=embedding,
    batch_size=16,
    thresh=1.95,
    verbose=True,
)

data = [NewsDocument.model_validate(doc.model_dump()).model_dump() for doc in documents]
save(data=data)
```

**这一步执行好后，文本数据将与相对应的向量数据一同被保存至ES数据库中。**

### 第四步：构建RAG问答

这一步，我们将会构建一个基于`RAG`的问答模型。

<center>
<img src="https://qdrant.tech/articles_data/what-is-rag-in-ai/how-rag-works.jpg" width="600px">
</center>

```python
from zlai.llms import Zhipu, GLM4FlashGenerateConfig
from zlai.embedding import PretrainedEmbedding
from zlai.agent import KnowledgeAgent
from zlai.elasticsearch import get_es_con

# ES数据库连接
index_name = "news"
hosts = "http://localhost:9200/"
con = get_es_con(hosts=hosts)

# 向量化模型
model_path="/home/models/BAAI/bge-small-zh-v1.5/"
embedding = PretrainedEmbedding(
    model_path=model_path,
    max_len=512,
    batch_size=16,
)

# LLM模型，这里采用glm-4-flash
llm = Zhipu(generate_config=GLM4FlashGenerateConfig())

# 构建RAG问答
knowledge = KnowledgeAgent(
    llm=llm, embedding=embedding, verbose=True,
    index_name=index_name, elasticsearch_host=hosts)

task_completion = knowledge("2022年国内参与空中游览和航空运动的人数？")
```

*输出日志：*

```text
[Knowledge Agent] Find Knowledge Title: rag_1.txt, Score: 1.6218.
==================== [Knowledge Agent] Messages Start ====================

user [327]: [
参考文本:.5亿元，同比增长33.8%。预计到2026年，中国低空经济市场规模将进一步扩大，突破万亿元大关，2023年至2026年复合年增长率将达到30%。

另据中国民航局数据，2022年国内参与...]

==================== [Knowledge Agent] Messages End    ====================

[Knowledge Agent] Final Answer: 2022年国内参与空中游览和航空运动的人数约为48万人次。

[Knowledge Agent] End ...
```

> 再问一个问题：2023年至2026年复合年增长率？

```python
task_completion = knowledge("2023年至2026年复合年增长率？")
```

*输出日志：*

```text
[Knowledge Agent] Find Knowledge Title: rag_1.txt, Score: 1.5233.
==================== [Knowledge Agent] Messages Start ====================

user [323]: [
参考文本:.5亿元，同比增长33.8%。预计到2026年，中国低空经济市场规模将进一步扩大，突破万亿元大关，2023年至2026年复合年增长率将达到30%。

另据中国民航局数据，2022年国内参与...]

==================== [Knowledge Agent] Messages End    ====================

[Knowledge Agent] Final Answer: 2023年至2026年，中国低空经济市场的复合年增长率预计将达到30%。

[Knowledge Agent] End ...
```

> 再问一个问题：旅游股2024年一季度合计净利润为？

```python
task_completion = knowledge("旅游股2024年一季度合计净利润为？")
```

*输出日志：*

```text
[Knowledge Agent] Find Knowledge Title: rag_2.txt, Score: 1.7192.
==================== [Knowledge Agent] Messages Start ====================

user [540]: [
参考文本:。“十三五”时期，旅游及相关产业增加值占GDP比重保持增长态势。

文化和旅游部今年初发布的统计数据显示，2023年国内出游达48.91亿人次，比上年同期增加23.61亿人次，同比增长93...]

==================== [Knowledge Agent] Messages End    ====================

[Knowledge Agent] Final Answer: 旅游股2024年一季度合计净利润为27.39亿元。

[Knowledge Agent] End ...
```

**可以看到，借助大模型的RAG流程，准确回答了文本中的对应问题。** 这里的向量化模型并不是一个很大的模型或者说效果很好的向量化模型，在算力充足的情况下您可以选用其他更好的向量化模型，如：`bge-m3`, `beg-large`等。

> 参考

- [What is RAG](https://qdrant.tech/articles/what-is-rag-in-ai/)

-----
@2024/06/06

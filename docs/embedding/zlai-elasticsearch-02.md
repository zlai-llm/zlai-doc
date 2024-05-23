# ElasticSearch 实践

## ElasticSearch 基本操作

### 创建链接

> 获取ES全局链接：`get_es_global_con`

**全局链接在整个代码区间内都可以使用。**

```python
from zlai.elasticsearch import *

hosts = ESUrl.url
get_es_global_con(hosts=hosts)
```

> 获取ES链接：`get_es_con`

```python
from zlai.elasticsearch import *

hosts = ESUrl.url
con = get_es_con(hosts=hosts)
```

### 数据索引

> 什么是索引`Index`？

在 Elasticsearch 中，索引（Index）类似于关系数据库中的数据库（Database）。一个索引是一个逻辑命名空间，其中存储了一组相关的文档（Documents）。每个文档是以 JSON 格式存储的，并包含一个或多个字段（Fields）。索引不仅包含文档的数据，还包含一个反向索引（Inverted Index），用于快速搜索文档中的内容。

**索引的核心概念**

* 文档（Document）：索引中的基本单位，每个文档是一个 JSON 对象，包含多个字段。例如，一篇博客文章可以是一个文档，包含标题、作者、内容、发布日期等字段。
* 类型（Type）：Elasticsearch 7.0 之前，索引中可以包含多种类型的文档，每种类型定义一类文档的字段和字段类型。7.0 之后，类型的概念被废弃，每个索引只包含一个文档类型。
* 字段（Field）：文档中的键值对，每个字段都有特定的数据类型（如字符串、数字、日期等）。字段可以设置不同的分析和索引方式，以影响搜索行为。
* 映射（Mapping）：定义索引中文档和字段的结构及其类型。映射类似于数据库中的表模式，规定了字段的类型和索引方式。

> 创建索引

```python
from zlai.elasticsearch import get_es_con, create_index
from elasticsearch_dsl import Document, Text, tokenizer, DenseVector, analyzer

tokenizer_ik = tokenizer('ik_smart')
analyzer_ik = analyzer('ik_smart', tokenizer=tokenizer_ik)

class ElasticSearchSchemaTest(Document):
    """知识文档的结构信息"""
    url = Text()
    title = Text(analyzer=analyzer_ik, search_analyzer=analyzer_ik)
    context = Text(analyzer=analyzer_ik, search_analyzer=analyzer_ik)
    vector = DenseVector(dims=1024)
    date = Text()
    
con = get_es_con(hosts="localhost:9200")
create_index(
    index_name='test_index',
    field_schema=ElasticSearchSchemaTest,
    reset=False, con=con, disp=True)
```

*output*

```text
>>> Index test_index created!
```

其中，`ElasticSearchSchemaTest`指定了这个索引的数据结构，包括字段名称、字段类型和分析器。

- `tokenizer/analyzer`: 指定文本字段的分词工具
- `DenseVector`: 存储向量的字段

*tips: 查看[索引: test_index](http://localhost:9200/test_index)*

### 保存数据

> 保存20条数据至ElasticSearch

```python
import numpy as np
from zlai.elasticsearch import get_es_con, doc2es

data = [{
    'url': 'https://example.com/1',
    'title': 'Document 1 你好',
    'context': 'This is document 1.',
    'vector': np.random.rand(1024),
    'date': '2021-01-01'
}]*20

con = get_es_con(hosts="localhost:9200")
doc2es(data, index_name='test_index', batch_size=2, con=con)
```

*output*

```text
>>> Docs in test_index saved!
```

### 查询数据

`zlai`中提供了`ElasticSearchTools`集成工具类，可以非常方便地查询索引中的数据。

> 查询索引的数据量

```python
from zlai.elasticsearch import ElasticSearchTools, get_es_con

con = get_es_con(hosts="localhost:9200")

tools = ElasticSearchTools(index_name='test_index', con=con)
print(tools.count())
```

```text
>>> 20
```

> 相似度计算

计算中知识库中文本的相似度: `es_cosine_similarity`

```python
import numpy as np
from zlai.elasticsearch import *

vector = np.random.rand(1024)

con = get_es_con(hosts=ESUrl.url)
es_cosine_similarity(
    vector, index_name="text_index",
    size=10, doc_vec_name='vector',
    thresh=0.7, con=con)
```

> 参数：

- `vector[List[float]]`: 需要匹配的向量
- `index_name`: 索引名称
- `size`: 相似度前 size 个数据
- `doc_vec_name`: 向量数据库中向量字段的名称 `vector`
- `thresh`: 过滤阈值
- `con`: 向量数据库链接

> 向量数据索引工具 `ElasticSearchTools`

- 计算数据量
- 匹配关键词
- 依据相似度提取最相似的文本

```python
import numpy as np
from zlai.elasticsearch import *

con = get_es_con(hosts=ESUrl.url)
tools = ElasticSearchTools(index_name='test_index', con=con)

# 计算索引数据量
tools.count()

# 匹配match_phrase/match_all只能指定一个字段进行匹配
tools.match_context('中国', fields='title', match_type='match')
tools.match_context('中国', fields='title', match_type='match_phrase')

# 匹配全量数据
tools.match_context('中国', match_type='match_all')

# multi_match可以指定多个字段进行匹配
tools.match_context('中国', fields=[], match_type='multi_match')

# 计算相似度
vector = np.random.rand(1024)
tools.cos_smi(vector, size=10, thresh=None, doc_vec_name='vector')
```

------
@2024/05/13

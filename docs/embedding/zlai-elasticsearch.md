# ElasticSearch

## 1. 获取ES链接

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

## 2. 创建ES索引

> 创建ES索引的数据结构: `Document`

**创建索引的过程类似创建MySQL的表结构。**

```python
from elasticsearch_dsl import Document, tokenizer, analyzer
from elasticsearch_dsl.field import Text, DenseVector

tokenizer_ik = tokenizer('ik_smart')
analyzer_ik = analyzer('ik_smart', tokenizer=tokenizer_ik)


class ElasticSearchSchemaTest(Document):
    url = Text()
    title = Text(analyzer=analyzer_ik, search_analyzer=analyzer_ik)
    context = Text(analyzer=analyzer_ik, search_analyzer=analyzer_ik)
    vector = DenseVector(dims=1024)
    date = Text()
```

- `tokenizer/analyzer`: 指定文本字段的分词工具
- `DenseVector`: 存储向量的字段

> 创建ES索引：`create_index`

```python
from zlai.elasticsearch import *

con = get_es_con(hosts=ESUrl.url)
create_index(
    index_name='test_index',
    field_schema=ElasticSearchSchemaTest,
    reset=True, con=con, disp=True)
```

## 3. 保存文件至ES数据库

> 保存文件至ES数据库: `doc2es`

```python
from zlai.elasticsearch import *

con = get_es_con(hosts=ESUrl.url)
doc2es(data, index_name='test_index', batch_size=2, con=self.con)
```

## 4. 相似度计算

> 计算中知识库中文本的相似度: `es_cosine_similarity`

```python
import numpy as np
from zlai.elasticsearch import *

vector = np.random.rand(1024)

con = get_es_con(hosts=ESUrl.url)
es_cosine_similarity(
    vector, index_name=index_name,
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

## 5. 向量数据索引工具

> `ElasticSearchTools`

- 计算数据量
- 匹配关键词
- 依据相似度提取最相思的文本

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



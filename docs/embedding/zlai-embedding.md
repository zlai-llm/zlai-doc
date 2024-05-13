
# Embedding

## 1. `Embedding`调用

### 1.1 远程调用方式

```python
import numpy as np
from zlai.schema import ZhipuEmbeddingModel
from zlai.embedding import Embedding

embedding = Embedding(
    api_key="your api key",
    model_name=ZhipuEmbeddingModel.emb2,
)
vectors = embedding.embedding(text='...')
vectors = np.array([vector.embedding for vector in vectors.data])
print(f"vector shape is: {vectors.shape}")
```

### 1.2 本地接口调用

```python
import numpy as np
from zlai.schema import EMBUrl
from zlai.embedding import Embedding

embedding = Embedding(
    emb_url=EMBUrl.m3e_small,
    max_len=512,
    max_len_error='split',
)
vectors = embedding.embedding(text='...')
vectors = np.array([vector.embedding for vector in vectors.data])
print(f"vector shape is: {vectors.shape}")
```

### 1.3 预训练模型推理

```python
import numpy as np
from zlai.embedding import Embedding

embedding = Embedding(
    model_path="/path/model",
    batch_size=128,
)
vectors = embedding.embedding(text='...')
vectors = np.array([vector.embedding for vector in vectors.data])
print(f"vector shape is: {vectors.shape}")
```

### 1.4 转化为`ndarray`

```python
from zlai.embedding import Embedding

embedding = Embedding(
    model_path="/path/model",
    batch_size=128,
)

vectors = embedding.embedding(text='...')
vectors.to_numpy()
```

## 2. `match` 匹配标签

### 2.1 匹配`top-n`内容

```python
from zlai.embedding import Embedding

embedding = Embedding(...)
top_n_idx = embedding.match_idx(
    source=...,
    target=...,
    top_n=2,
    filter='top_n',
)
print(top_n_idx)
```

### 2.2 匹配超过相似度阈值`thresh`内容，输出 ID

```python
from zlai.embedding import Embedding

embedding = Embedding(...)
top_n_idx = embedding.match_idx(
    source=...,
    target=...,
    thresh=0.8,
    filter='thresh',
)
print(top_n_idx)
```

### 2.3 匹配超过相似度阈值`thresh`的`top_n`内容, 输出 ID

```python
import numpy as np
from zlai.embedding import Embedding

embedding = Embedding(...)
top_n_idx = embedding.match_idx(
    source=...,
    target=...,
    top_n=8,
    thresh=0.8,
    filter='union',
)
print(top_n_idx)
```

### 2.4 匹配关键词

> 匹配`标签/文本`到原文文本切片的相似度

```python
from zlai.schema import *
from zlai.embedding import Embedding

embedding = Embedding(
    emb_url=EMBUrl.bge_large,
    max_len=512,
    max_len_error='split',
    batch_size=2,
)

source = ['铁路', '棒棒冰']
target = ['铁桥', '铁道', '火车', '雪糕']
idx = embedding.match(source=source, target=target, top_n=4)
```

```markdown
# output:
src='铁路' dst=['铁道', '火车', '铁桥', '雪糕'] score=[0.8850052852844537, 0.8241780257520692, 0.6285992056646379, 0.35345702698242426] match_type=['score'] keyword_method=None
src='棒棒冰' dst=['雪糕', '铁桥', '铁道', '火车'] score=[0.5949498636630225, 0.2540493589500271, 0.22198863747093053, 0.2155088276235349] match_type=['score'] keyword_method=None
```

### 2.5 完整匹配关键词

```python
from zlai.schema import *
from zlai.embedding import Embedding

embedding = Embedding(
    emb_url=EMBUrl.bge_large,
    max_len=512,
    max_len_error='split',
    batch_size=2,
)

source = ['铁路', '棒棒冰']
target = ['铁路大桥', '铁桥', '铁道', '火车', '雪糕']
# keyword_method=keyword 对匹配到的完整词条短语进行召回
data = embedding._match_keyword(source=source, target=target, thresh=0.7, keyword_method='keyword')
# keyword_method=content 对匹配到的全文进行召回
data = embedding._match_keyword(source=source, target=target, thresh=0.7, keyword_method='content')
```

```markdown
# output - 1:
src='铁路' dst=['铁路大桥'] score=[0.6243824064544091] match_type=['keyword'] keyword_method='keyword'
# output - 2:
src='铁路' dst=['铁道'] score=[0.8850052924856231] match_type=['keyword'] keyword_method='content'
```

### 2.6 混合相似度

```python
from zlai.schema import *
from zlai.embedding import Embedding

embedding = Embedding(
    emb_url=EMBUrl.bge_large,
    max_len=512,
    max_len_error='split',
    batch_size=2,
)

source = ['铁路', '棒棒冰']
target = ['铁路大桥', '铁桥', '铁道', '火车', '雪糕', '水蜜桃棒棒冰']
data = embedding.match_with_keyword(
    source=source, target=target, thresh=(0.75, 0.6), keyword_method='keyword')

data = embedding.match_with_keyword(
    source=source, target=target, thresh=(0.75, 0.6), keyword_method='keyword', drop_duplicate=False)

data = embedding.match_with_keyword(
    source=source, target=target, thresh=(0.8, 0.7), keyword_method='content')
```

------
@2024/03/06

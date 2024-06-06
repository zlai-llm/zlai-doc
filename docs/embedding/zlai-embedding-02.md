# Embedding

## `Embedding`使用

> `zlai`中支持三种`Embedding`的计算方式：

* 预训练模型推理：使用本地模型文件进行推理，如`m3e_small`，`m3e_base`等，这需要您从[huggingface](https://huggingface.co/)上下载模型文件。
* 远程调用方式：调用远程接口进行计算，如`ali`，`zhipuai`等。
* 本地接口调用：这种方式是指您从在自己的计算服务器上部署了向量化相关服务，并使用`zlai`进行远程调用。

### 预训练模型推理

这种方式下，需要你指定模型文件的路径，并指定一些列的推理参数进行向量计算。这里以[`BAAI/bge-small-zh-v1.5`](https://huggingface.co/BAAI/bge-small-zh-v1.5)模型为例。

```python
from zlai.embedding import PretrainedEmbedding

model_path = "/home/models/BAAI/bge-small-zh-v1.5"
embedding = PretrainedEmbedding(
    model_path=model_path,                  # 模型路径
    batch_size=2,                           # 数据计算每批次样本数量
    normalize_embeddings=True,              # 是否归一化词向量
    verbose=True,                           # 是否打印日志
)
text = ["你好", "你是谁", "你这里有点好看", "如何打车去南京路", "我想吃西瓜", "今天去做点什么呢"]
data = embedding.embedding(text=text)
print(data.to_numpy().shape)
```

*打印输出：*

```text
(6, 512)
```

我们可以看到上面打印输出了一个`(6, 512)`的`ndarray`，其中`6`表示输入文本的数量，`512`表示每个文本的向量维度。当您考虑非联网环境下的使用时，这种方式是您最常用的形式。

### 远程调用方式

> [智谱API](https://open.bigmodel.cn/dev/api#text_embedding)

这里以智谱的`embedding-2`模型接口为例。

```python
from zlai.embedding import ZhipuEmbedding, ZhipuEmbeddingConfig

embedding = ZhipuEmbedding(config=ZhipuEmbeddingConfig())
response = embedding.embedding(text=tuple(["你好", "你好"]))
print(response.to_numpy().shape)
```

*打印输出：*

```text
(2, 1024)
```

> [阿里云API](https://help.aliyun.com/zh/dashscope/developer-reference/text-embedding-api-details?disableWebsiteRedirect=true)

```python
from zlai.embedding import AliEmbedding, AliEmbeddingV2Config

embedding = AliEmbedding(batch_size=1, config=AliEmbeddingV2Config())
response = embedding.embedding(
    text=tuple(["你好", "你好", "你好", "你好"])
)
print(response.to_numpy().shape)
print(response.usage)
```

*打印输出：*

```text
(4, 1536)
prompt_tokens=0 completion_tokens=0 total_tokens=4
```

当您的计算机性能较低时，您可以尝试使用远程模型服务商的接口，并指定`batch_size`参数等相关参数来获取文本的向量化数据。

### 本地接口调用

> 如果您在其他计算机上部署了向量化相关服务，并希望使用`zlai`进行远程调用，则可以使用这种方式。这里仅给出一个示例

```python
from zlai.schema import EMBUrl
from zlai.embedding import Embedding

embedding = Embedding(
    emb_url=EMBUrl.m3e_small,  # 服务接口地址
    max_len=512,
    max_len_error='split',
)
response = embedding.embedding(text='...')
print(response.to_numpy().shape)
print(response.usage)
```

> TIPS: 转化为`ndarray`与`list`

```python
from zlai.embedding import Embedding

embedding = Embedding(
    model_path="/path/model",
    batch_size=128,
)

vectors = embedding.embedding(text='...')
# 转换为 ndarray
vectors.to_numpy()
# 转换为 list
vectors.to_list()
```

> 特殊说明

1. `Embedding`为通用向量化方法，可以用于调用本地模型文件，模型服务商接口以及自己部署的模型接口。
2. `PretrainedEmbedding`只适用于调用本地模型文件。
3. `ZhipuEmbedding`只适用于调用智谱的接口。
4. `AliEmbedding`只适用于调用阿里云的接口。

在下面介绍的向量化索引中，我们以`Embedding`为例。但如果您使用了自己本地化的模型文件，推荐您使用`PretrainedEmbedding`。

## 文本向量索引

这一小节中，我们将介绍`zlai`中`Embedding`的索引功能，即通过将文本进行向量化，并计算向量之间的相似度对文本进行相似性的检索。

### 匹配`top-n`内容

依据相似度，匹配`top-n`的文本内容，输出 ID。

```python
from zlai.embedding import Embedding

model_path = "/home/models/BAAI/bge-small-zh-v1.5"
embedding = Embedding(
    model_path=model_path,
    batch_size=16,
    normalize_embeddings=True,
    verbose=True,
)
target = [
    "谁更容易避免损失，谁的责任更大，谁更需要主动采取措施防止问题发声",
    "在这个事件中，女子的行为绝对责任占大头",
    "首先，网约车迟到并不是小概率事件，偶尔发生是可预判的",
    "而且网约车是无法知道乘客去机场剩余多少时间值机，也不知道你坐飞机去做什么",
    "换句话说，网约车的责任和平时迟到八分钟没有区别",
]

query = "谁的责任占大头？"
top_n_idx = embedding.match_idx(
    source=[query],
    target=target,
    top_n=2,
    filter='top_n',
)
print(top_n_idx)
for idx in top_n_idx[0]:
    print(f"{idx} - {target[idx]}")
```

*输出*

```text
[[1, 0]]
1 - 在这个事件中，女子的行为绝对责任占大头
0 - 谁更容易避免损失，谁的责任更大，谁更需要主动采取措施防止问题发声
```

**在这个示例中，`query`为需要匹配的问题文本，`target`为待匹配的文本数据，`top_n`指从`target`中匹配多少条数据。并打印出了输出结果，输出结果为二维的`List`，数值为匹配到文本的`ID`。在上面的匹配中我们从文本中找出了两条与问题最为接近的原文。**

> 当然，如果你观察的足够仔细会发现，上面的示例中，`source`参数接受的数据类型为`List`，也就是说，他可以接受多个问题，并返回多个问题的匹配结果。下面是一个示例：

```python
query = ["谁的责任占大头？", "网约车是否知道乘客去机场剩余多少时间值机？"]
top_n_idx = embedding.match_idx(
    source=query,
    target=target,
    top_n=2,
    filter='top_n',
)
print(top_n_idx)
```

*输出*

```text
[[1, 0], [3, 2]]
```

**这里输出了，两个`List`结果，第一个`List`为第一个问题匹配到的原文，以此类推。您可以观察一下匹配出的文本是否与问题有一定的语义相关性。**

### 相似度阈值匹配

指的是依据相似度的阈值，匹配相似度大于阈值`thresh`的文本内容，输出 ID。

```python
from zlai.embedding import Embedding

model_path = "/home/models/BAAI/bge-small-zh-v1.5"
embedding = Embedding(
    model_path=model_path,
    batch_size=16,
    normalize_embeddings=True,
    verbose=True,
)
target = [
    "谁更容易避免损失，谁的责任更大，谁更需要主动采取措施防止问题发声",
    "在这个事件中，女子的行为绝对责任占大头",
    "首先，网约车迟到并不是小概率事件，偶尔发生是可预判的",
    "而且网约车是无法知道乘客去机场剩余多少时间值机，也不知道你坐飞机去做什么",
    "换句话说，网约车的责任和平时迟到八分钟没有区别",
]

query = "谁的责任占大头？"
top_n_idx = embedding.match_idx(
    source=[query],
    target=target,
    thresh=0.6,
    filter='thresh',
)
print(top_n_idx)
for idx in top_n_idx[0]:
    print(f"{idx} - {target[idx]}")
```

*输出*

```text
[[1]]
1 - 在这个事件中，女子的行为绝对责任占大头
```

**这个示例中，匹配出了与问题相似度阈值大于0.6的文本ID。**

### 混合匹配

匹配超过相似度阈值`thresh`的`top_n`内容, 输出 ID

```python
from zlai.embedding import Embedding

model_path = "/home/models/BAAI/bge-small-zh-v1.5"
embedding = Embedding(
    model_path=model_path,
    batch_size=16,
    normalize_embeddings=True,
    verbose=True,
)
target = [
    "谁更容易避免损失，谁的责任更大，谁更需要主动采取措施防止问题发声",
    "在这个事件中，女子的行为绝对责任占大头",
    "首先，网约车迟到并不是小概率事件，偶尔发生是可预判的",
    "而且网约车是无法知道乘客去机场剩余多少时间值机，也不知道你坐飞机去做什么",
    "换句话说，网约车的责任和平时迟到八分钟没有区别",
]

query = "谁的责任占大头？"
top_n_idx = embedding.match_idx(
    source=[query],
    target=target,
    thresh=0.5,
    top_n=1,
    filter='union',
)
print(top_n_idx)
for idx in top_n_idx[0]:
    print(f"{idx} - {target[idx]}")
```

*输出*

```text
[[1]]
1 - 在这个事件中，女子的行为绝对责任占大头
```

**这个示例中，匹配出了与问题相似度阈值大于0.6且仅输出第一条的文本ID。**

### 匹配文本对象

> 依据相似度，匹配`source`到`target`的相似文本。

```python
from zlai.embedding import Embedding

model_path = "/home/models/BAAI/bge-small-zh-v1.5"
embedding = Embedding(
    model_path=model_path,
    max_len=512,
    max_len_error='split',
    batch_size=2,
)

source = ['铁路', '棒棒冰']
target = ['铁桥', '铁道', '火车', '雪糕']
idx = embedding.match(source=source, target=target, top_n=4, filter="top_n")
print(idx)
```

在上面的例子中我们寻找`source = ['铁路', '棒棒冰']`与`target = ['铁桥', '铁道', '火车', '雪糕']`的相似文本，并返回相似度排名前4的文本。输出结果如下：

```markdown
[
    EmbeddingMatchOutput(
        src='铁路', 
        dst=['铁道', '铁桥', '火车', '雪糕'], 
        score=[0.8009219452801325, 0.6997149241500058, 0.6825761050250018, 0.21049784574282834], 
        match_type=['score'], 
        keyword_method=None, 
        target=['铁道', '铁桥', '火车', '雪糕']
    ), 
    EmbeddingMatchOutput(
        src='棒棒冰', 
        dst=['雪糕', '铁桥', '铁道', '火车'], 
        score=[0.5197617261162202, 0.31604522550474096, 0.2777551278054988, 0.2538721266364353], 
        match_type=['score'], 
        keyword_method=None, 
        target=['雪糕', '铁桥', '铁道', '火车']
    )
]
```

*输出字段：*

- `src`: 问题文本，依据此信息去寻找目标文本。
- `dst`: 目标文本，即匹配出的文本信息。
- `score`: 与目标文本对应的相似度值。
- `match_type`: 匹配方式，`score`指依据相似度进行匹配。
- `keyword_method`: 依据关键词进行匹配，`keyword`指依据关键词进行匹配。
- `target`: 原始文本。

### 匹配关键词

一些时候我们需要依据关键词进行匹配，比如在搜索时，我们输入关键词，然后依据关键词去匹配文本。这种情况是对相似度匹配的一种补充，如文本中出现了相关关键词，但相似度程度由于其他词汇语义的影响，导致匹配度较低，此时我们可以依据关键词进行匹配。下面是一个例子：

```python
from zlai.embedding import Embedding

model_path = "/home/models/BAAI/bge-small-zh-v1.5"
embedding = Embedding(
    model_path=model_path,
    max_len=512,
    max_len_error='split',
    batch_size=2,
)

source = ['铁路', '棒棒冰']
target = ['铁路大桥', '铁桥', '铁道', '火车', '雪糕']
# keyword_method=keyword 对匹配到的完整词条短语进行召回
data = embedding._match_keyword(source=source, target=target, thresh=0.7, keyword_method='keyword')
print(data)

# keyword_method=content 对匹配到的全文进行召回
data = embedding._match_keyword(source=source, target=target, thresh=0.7, keyword_method='content')
print(data)
```

1. `keyword_method='keyword'`模式下，会依据关键词判断source中的文本元素是否在target中的元素中出现，如果匹配并且相似度阈值大于thresh，则返回匹配结果。
2. `keyword_method='content'`模式下，会依据关键词判断source中的文本元素是否在target中的任意元素中出现，如果匹配并且相似度阈值大于thresh，则返回匹配结果。

所以，这里并不是仅仅以关键词的方式进行文本的匹配，而是匹配到关键词后相似度阈值也要大于某值才会输出匹配后的文本，下面是结果输出：

```markdown
# output - 1:
[EmbeddingMatchOutput(src='铁路', dst=['铁路大桥'], score=[0.7696197893487535], match_type=['keyword'], keyword_method='keyword', target=None)]
# output - 2:
[EmbeddingMatchOutput(src='铁路', dst=['铁道'], score=[0.8009220248236943], match_type=['keyword'], keyword_method='content', target=None)]
```

### 混合相似度匹配

很多时候我们希望依据相似度匹配文本并且可以支持关键词进行匹配，以下是一个这类混合相似度阈值的示例。

```python
from zlai.schema import *
from zlai.embedding import Embedding

model_path = "/home/models/BAAI/bge-small-zh-v1.5"
embedding = Embedding(
    model_path=model_path,
    max_len=512,
    max_len_error='split',
    batch_size=2,
)

source = ['铁路']
target = ['铁路大桥', '铁桥', '铁道', '火车', '雪糕', '水蜜桃棒棒冰']
data = embedding.match_with_keyword(
    source=source, target=target, thresh=(0.75, 0.6), keyword_method='keyword')
print(data)
```

上面的示例中我们给定了两个相似度阈值`(0.75, 0.6)`，第一个阈值是依据相似度进行匹配，第二个阈值是依据关键词进行匹配的最小相似度阈值。这样即考虑到了对于一般性文本的相似度匹配，又考虑到了对于关键词的匹配，以及又不希望关键词误判。

*输出*

```text
src='铁路' dst=['铁道', '铁路大桥'] score=[0.8009220248236943, 0.7696197893487535] match_type=['score', 'keyword'] keyword_method=None target=['铁道']
```

`drop_duplicate`参数默认情况为`True`，该方法会对相似度匹配与关键词匹配的值进行去重，指定为`False`之后，会返回所有匹配的文本并不对其进行去重操作。

```python
data = embedding.match_with_keyword(
    source=source, target=target, thresh=(0.75, 0.6), keyword_method='keyword', drop_duplicate=False)
print(data)
```

*输出*

```text
src='铁路' dst=['铁道'] score=[0.8009220248236943] match_type=['score'] keyword_method=None target=['铁道']
src='铁路' dst=['铁路大桥'] score=[0.7696197893487535] match_type=['keyword'] keyword_method='keyword' target=None
```

------
@2024/03/06

# Document


# 大模型输出的结构化解析

```python
from zlai.parse import *
```

## 1. `Dict`解析

> 普通模式

```python
from zlai.parse import *

string = "..."
sparse = ParseString()
sparse.eval_dict(string=string)
```

> 贪婪模式

```python
from zlai.parse import *

string = "..."
sparse = ParseString()
sparse.greedy_dict(string=string)
```

> 嵌套模式

```python
from zlai.parse import *

string = "..."
sparse = ParseString()
sparse.nested_dict(string=string)
```

## 2. `List`解析

> 普通模式

```python
from zlai.parse import *

string = "..."
sparse = ParseString()
sparse.eval_list(string=string)
```

> 贪婪模式

```python
from zlai.parse import *

string = "..."
sparse = ParseString()
sparse.greedy_list(string=string)
```

> 嵌套模式

```python
from zlai.parse import *

string = "..."
sparse = ParseString()
sparse.nested_list(string=string)
```

-------
@2024/03/06

# 文本切分

## 1. `TextPrepareBase.split_string`

按照固定字数切分

> 参数：

- `string`: 待切分的字符串
- `k`: 切分字数

> 示例：

```python
from zlai.retrievers import *

spliter = TextPrepareBase()
spliter.split_string(string='我是一只小小鸟', k=4)
```

## 2. `TextPrepareBase.split_by_pattern`

依据多种符号进行文本的切割，并限制最大的字符串的长度。

> 参数：

- `string`: 待切分的字符串
- `pattern`: 切分符号，`\n|。`
- `k`: 切分字数

> 示例：

```python
from zlai.retrievers import *

spliter = TextPrepareBase()
spliter.split_by_pattern(string='我是，一只。小小鸟', pattern='\n|。', k=4)
```

## 3. `TextPrepareBase.find_keyword_sub_content_by_name`

切分关键词后的`k`个字数。

> 参数：

- `string[str]`: 待切分的字符串
- `keywords[List[str]]`: 关键词
- `k[int]`: 切分字数

> 示例：

```python
from zlai.retrievers import *

spliter = TextPrepareBase()
spliter.find_keyword_sub_content_by_name(
    string='我是，一只。小小鸟', keywords=['一只'], k=4)
```

## 4. `TextPrepareBase.find_keyword_sub_content_by_emb`

切分步骤：
1. 对原文依据`pattern`进行切分
2. 依据`emb`对切分后的文本与关键词进行向量化
3. 依据相似度取`n`个切片。

> 参数：

- `string`: 待切分的字符串
- `emb`: 向量化模型
- `keywords[List[str]]`: 关键词
- `pattern`: 切分符号，`\n|。`
- `k`: 切分字数
- `n`: `n`个切片

> 示例：

```python
from functools import partial
from zlai.llms import *
from zlai.utils import *
from zlai.retrievers import *

emb = partial(emb_batch, url=EMBUrl.m3e_small, batch_size=100, disp=False)

spliter = TextPrepareBase()
spliter.find_keyword_sub_content_by_emb(
    string='文本原文...', emb=emb, keywords=['关键词'],
    k=1000, n=1, pattern='\n|。')
```

## 5. `TextPrepareBase.find_keyword_sub_content_by_max_smi`

切分步骤：
1. 对原文依据`pattern`进行切分
2. 依据`emb`对切分后的文本与关键词进行向量化
3. 依据相似度取`n`个切片后`k`个字符。

> 参数：

- `string`: 待切分的字符串
- `emb`: 向量化模型
- `keywords[List[str]]`: 关键词
- `pattern`: 切分符号，`\n|。`
- `k`: 切分字数
- `n`: `n`个切片

> 示例：

```python
from functools import partial
from zlai.llms import *
from zlai.utils import *
from zlai.retrievers import *

emb = partial(emb_batch, url=EMBUrl.m3e_small, batch_size=100, disp=False)

spliter = TextPrepareBase()
spliter.find_keyword_sub_content_by_max_smi(
    string='文本原文...', emb=emb, keywords=['关键词'],
    k=1000, n=1, pattern='\n|。'
)
```

## 6. `PolicySupportText.select_project_condition_content`

切分步骤：参考`TextPrepareBase.find_keyword_sub_content_by_max_smi`

> 参数：

- `string`: 待切分的字符串
- `pattern`: 切分符号，`\n|。`
- `k`: 切分字数

> 示例：

```python
from functools import partial
from zlai.llms import *
from zlai.utils import *
from zlai.retrievers import *

emb = partial(emb_batch, url=EMBUrl.m3e_small, batch_size=100, disp=False)

spliter = PolicySupportText(content='文本原文...')
spliter.clean()
project_content = spliter.select_project_condition_content(
    min_tok=500, emb=emb, keywords=['项目名称'],
)
for item in project_content:
    print(30 * '=')
    print(item.get("project_name"))
    print(item.get("content"))
```



------
@20240327


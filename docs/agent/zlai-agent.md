# ZLAI-Agent

## Bing Search Agent

> 使用Bing搜索API

```python
from zlai.agent.bing import BingSearch
from zlai.embedding import Embedding
from zlai.schema import EMBUrl
from zlai.llms import *


embedding = Embedding(
    emb_url=EMBUrl.bge_m3,
    max_len=5000,
    max_len_error='split',
    batch_size=2,
)
config_glm3 = ZhipuGLM3Turbo()
llm = Zhipu(
    generate_config=config_glm3,
)

bing = BingSearch(
    llm=llm,
    embedding=embedding,
    n_pages=10,
    chunk_size=1000,
    chunk_overlap=300,
    verbose=True,
    n_chunk=3,
)
# query = "杭州今天的天气如何？"
# query = "俄罗斯今天的新闻有哪些？"
# query = "五五计划是哪一年开始的？"
query = "各省市数据资产入表案例"
answer = bing(query=query)
print(answer.choices[0].message.content)
```

------
@2024/03/27


# Streamlit Message

> 这里的工作是为了让Message组件在Streamlit中工作，方便页面交互。

## streamlit message 介绍

> `zlai`为不同数据类型的数据设定了`message`模版。`message`模版有多重用途，除了上一节介绍的正常的与大模型进行交互外，还可以使用`Message.show_streamlit()`方法轻松在`streamlit`中展示不同数据类型的消息。下面是一个示例：

*在`streamlit_message.py`中写入下面的代码：*

<details>
    <summary>点击展开查看：streamlit_message.py</summary>

```python
import random
import pandas as pd
from zlai.types.messages import *
from pyecharts.charts import Bar
from pyecharts.faker import Faker
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from zlai.types.function_call import *
from zlai.streamlit import avatar_mapping


image_url = "https://pics7.baidu.com/feed/e1fe9925bc315c601ae44d2fa329e7184854771b.jpeg"
audio_path = "./audio.wav"
chart = Bar(init_opts=opts.InitOpts(width="640px", height="300px", theme=ThemeType.MACARONS))\
    .add_xaxis(Faker.choose())\
    .add_yaxis("商家A", Faker.values())\
    .add_yaxis("商家B", Faker.values())\
    .set_global_opts(
        title_opts={"text": "Bar-通过 dict 进行配置", "subtext": "我也是通过 dict 进行配置的"}
    ).render_embed()

table = pd.DataFrame(
    {
        "name": ["Roadmap", "Extras", "Issues"],
        "url": ["https://roadmap.streamlit.app", "https://extras.streamlit.app",
                "https://issues.streamlit.app"],
        "stars": [random.randint(0, 1000) for _ in range(3)],
        "views_history": [[random.randint(0, 5000) for _ in range(30)] for _ in range(3)],
    }
)
tools_message = ChatCompletionMessage(
    role="assistant",
    content=None,
    tool_calls=[ChatCompletionMessageToolCall(
        id="test",
        function=Function(arguments="{'a': 'v'}", name="test"),
        type="function",
    )],
)
observation_message = ObservationMessage(content={"a":"v"})

messages = [
    SystemMessage(content="This is system message."),
    UserMessage(content="Hello 👋."),
    AssistantMessage(content="Hi there! How can I help you today?"),
    ImageMessage(content="介绍一下这个图片", images_url=[image_url]),
    AudioMessage(role="assistant", content="这是我唱的歌", audios_path=[audio_path]),
    ChartMessage(content="这是我的图表", charts=[chart]),
    TableMessage(content="这是表格", tables=[table]),
    tools_message.to_message(),
    observation_message.to_message(),
]

import streamlit as st

for message in messages:
    with st.chat_message(message.role, avatar=avatar_mapping.get(message.role, "🤖")):
        message.show_streamlit()
```

</details>

*运行`streamlit_message.py`*

```bash
streamlit run streamlit_message.py
```

<details>
    <summary>点击展开查看：页面截图</summary>

<center>
<img src="img/message/streamlit-message-01.png" width="680px">
</center>

<center>
<img src="img/message/streamlit-message-02.png" width="680px">
</center>

</details>

-----

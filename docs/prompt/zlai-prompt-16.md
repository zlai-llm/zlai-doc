# 多模态思维链提示方法

最近，[Zhang等人（2023）](https://arxiv.org/abs/2302.00923)提出了一种多模态思维链提示方法。传统的思维链提示方法侧重于语言模态。相比之下，多模态思维链提示将文本和视觉融入到一个两阶段框架中。第一步涉及基于多模态信息的理性生成。接下来是第二阶段的答案推断，它利用生成的理性信息。

多模态CoT模型（1B）在ScienceQA基准测试中的表现优于GPT-3.5。

<img src="https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fmultimodal-cot.a84f6cc1.png&w=1080&q=75">

图片来源：[Zhang et al. (2023)](https://arxiv.org/abs/2302.00923)

进一步阅读：

- [语言不是你所需要的全部：将感知与语言模型对齐](https://arxiv.org/abs/2302.14045)（2023年2月）
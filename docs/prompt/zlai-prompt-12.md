# 方向性刺激提示

[Li 等人，（2023）](https://arxiv.org/abs/2302.11520)提出了一种新的提示技术，以更好地指导 LLM 生成所需的摘要。

训练了一个可调节的策略 LM 来生成刺激/提示。越来越多地使用RL来优化 LLM。

下图显示了方向性刺激提示与标准提示的比较。策略 LM 可以很小，并且可以优化以生成指导黑盒冻结 LLM 的提示。

<img src="https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fdsp.27a0005f.jpeg&w=1080&q=75">

DSP
图片来源：[Li 等人，（2023）](https://arxiv.org/abs/2302.11520)

完整示例即将推出！

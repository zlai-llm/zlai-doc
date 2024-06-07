# Active-Prompt

思维链（CoT）方法依赖于一组固定的人工注释范例。问题在于，这些范例可能不是不同任务的最有效示例。为了解决这个问题，[Diao 等人（2023）](https://arxiv.org/pdf/2302.12246.pdf)最近提出了一种新的提示方法，称为 Active-Prompt，以适应 LLMs 到不同的任务特定示例提示（用人类设计的 CoT 推理进行注释）。

下面是该方法的说明。第一步是使用或不使用少量 CoT 示例查询 LLM。对一组训练问题生成 k 个可能的答案。基于 k 个答案计算不确定度度量（使用不一致性）。选择最不确定的问题由人类进行注释。然后使用新的注释范例来推断每个问题。

<img src="https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Factive-prompt.f739657b.png&w=1200&q=75">

ACTIVE
图片来源：[Diao等人（2023）](https://arxiv.org/pdf/2302.12246.pdf)
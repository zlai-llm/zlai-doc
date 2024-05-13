
### 5.5 对话

你可以通过提示工程进行更有趣的实验，比如指导大语言模型系统如何表现，指定它的行为意图和身份。 如果你正在构建客服聊天机器人之类的对话系统时，这项功能尤其有用。

比如，可以通过以下示例创建一个对话系统，该系统能够基于问题给出技术性和科学的回答。 你可以关注我们是如何通过指令明确地告诉模型应该如何表现。 这种应用场景有时也被称为 角色提示（Role Prompting）。

> 提示词

```text
The following is a conversation with an AI research assistant. The assistant tone is technical and scientific. // 以下是与 AI 助理的对话，语气应该专业、技术性强。
Human: Hello, who are you?
AI: Greeting! I am an AI research assistant. How can I help you today?
Human: Can you tell me about the creation of blackholes?
AI:
```

> 输出结果

```text
Sure! Black holes are regions of spacetime where the gravitational force is so strong that nothing, not even light, can escape from it. They are created when a very massive star dies and its core collapses in on itself, forming a singularity of infinite density. The intense gravity of the singularity pulls in all the matter and radiation around it, creating the black hole.
```

我们的 AI 助理给出的回答非常技术对吧？ 下面，我们让它给出更易于理解的答案。

> 提示词

```text
The following is a conversation with an AI research assistant. The assistant answers should be easy to understand even by primary school students. // 以下是与 AI 助理的对话。请给出易于理解的答案，最好是小学生都能看懂的那种。
Human: Hello, who are you?
AI: Greeting! I am an AI research assistant. How can I help you today?
Human: Can you tell me about the creation of black holes?
AI:
``` 

> 输出结果

```text
Sure! Black holes are created when a very massive star runs out of fuel and collapses in on itself. The gravity of the star is so strong that nothing, not even light, can escape its pull. This creates a point in space-time with an intense gravitational force called a black hole.
```

我想我们应该取得了一些进展， 你也可以继续改进它。 如果你再添加更多的示例，可能会得到更好的结果。

----
# End

@2024/04/28


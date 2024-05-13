# 大模型参数配置介绍

## 简介

当前的大模型基本上都是自回归模型（[参考](https://jalammar.github.io/illustrated-gpt2/)），即在给定上下文的情况下，生成下一个词。本节将简要介绍不同的自回归解码策略，以及这些解码策略如何影响模型输出结果，以此我们可以更高效、更具针对性的配置大模型的参数。

简而言之，自回归语言生成基于这样的假设，即词序列的概率分布可以分解为条件下的下一个词概率的乘积：

$$ P(w_{1:T} | W_0 ) = \prod_{t=1}^T P(w_{t} | w_{1: t-1}, W_0) \text{ ,with }  w_{1: 0} = \emptyset, $$

其中$W_0$是初始的上下文词元(`token`)。词序列的长度 $T$ 通常是根据从$P(w_{t} | w_{1: t-1}, W_{0})$中生成的 $EOS$ 标记确定的，即$t=T$。

`GPT2`，`XLNet`，`OpenAi-GPT`，`CTRL`，`TransfoXL`，`XLM`，`Bart`，`T5`这些开源模型都可以进行自回归语言生成。

下面，我们将介绍当前最常见的解码方法，主要包括贪婪搜索、束搜索、Top-K采样和Top-p采样。

## 贪婪搜索`Greedy Search`

### **Greedy Search**

贪婪搜索（Greedy search）简单地选择具有最高概率的词作为下一个词：$w_t = argmax_{w}P(w | w_{1:t-1})$ 在每个时间步$t$。下面的示意图显示了贪婪搜索的过程。

<center>
<img src="https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/greedy_search.png" width="400px">
<h6>图: 贪婪搜索</h6>
</center>

从词 $\text{"The"}$ 开始，算法贪婪地选择最高概率的下一个词 $\text{"nice"}$，以此类推，最终生成的词序列是 $\text{"The", "nice", "woman"}$，总概率为 $0.5 \times 0.4 = 0.2$。

在给定的上下文之后，依据上下文生成概率最大的词语也许是合理的，但在实践中模型很快开始重复自己！可以参考[Vijayakumar et al., 2016](https://arxiv.org/abs/1610.02424) 与 [Shao et al., 2017](https://arxiv.org/abs/1701.03185)的研究。贪婪搜索的主要缺点是它会错过隐藏在低概率词后面的高概率词。 束搜索也许能缓解这个问题！

## 束搜索`Beam Search`

束搜索（`Beam search`）通过在每个时间步保留最有可能的`num_beams`个假设，最终选择具有最高总概率的下一个词，从而降低错过隐藏的高概率词序列的风险。我们以`num_beams=2`为例进行说明：

<center>
<img src="https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/beam_search.png" width="400px">
<h6>图: Beam search</h6>
</center>

在时间步骤`1`中，除了最有可能的假设 $\text{"The", "nice"}$ 之外，束搜索还跟踪第二个最有可能的假设 $\text{"The", "dog"}$。在时间步骤`2`中，束搜索发现词序列$\text{"The", "dog", "has"}$ 的概率为`0.36`，高于$\text{"The", "nice", "woman"}$的概率`0.2`。束搜索总是能找到比贪婪搜索更高的总概率的输出序列。

虽然束搜索一定程度上避免了贪婪搜索的缺点，但仍然存在重复的问题。一种简单的解决方法是引入**n-gram**（长度为$n$的词序列）惩罚(相关参考：[Paulus et al. (2017)](https://arxiv.org/abs/1705.04304) 与 [Klein et al. (2017)](https://arxiv.org/abs/1701.02810))。最常见的`n-gram`惩罚方法是通过手动将已经出现的`n-gram`的下一个单词的概率设置为$0$来确保没有重复出现的`n-gram`。

对于任务中期望的生成长度相对可预测的情况，如机器翻译或摘要生成，束搜索可以非常有效，参见[Murray et al. (2018)](https://arxiv.org/abs/1808.10006) 和 [Yang et al. (2018)](https://arxiv.org/abs/1808.09582)。但对于期望输出长度变化很大的开放式生成任务，如对话和故事生成，情况就不同了。在开发故事生成中，使用`n-gram`或其他惩罚机制来控制重复特别困难，因为要在`强制无重复`和`重复相同n-gram循环`之间找到一个良好的平衡需要进行大量的微调。正如[Ari Holtzman et al. (2019)](https://arxiv.org/abs/1904.09751)所指出的，高质量的人类语言并不遵循高概率下一个词的分布。换句话说，我们希望生成的文本能给我们带来惊喜，而不是无聊/可预测的。作者通过绘制模型对人类文本赋予的概率与束搜索的结果之间的关系，很好地展示了这一点。

<center>
<img src="https://blog.fastforwardlabs.com/images/2019/05/Screen_Shot_2019_05_08_at_3_06_36_PM-1557342561886.png" width="700px">
<h6>图: Beam search VS human Language</h6>
</center>

## `词元采样`Sampling

在最基本的自回归模型生成形式中，采样意味着根据条件概率分布随机选择下一个词$w_t$：

$$w_t \sim P(w|w_{1:t-1})$$

以上面的例子为例，下面的图形展示了使用采样进行语言生成的过程。

<center>
<img src="https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/sampling_search.png" width="600px">
<h6>图: Sampling</h6>
</center>

很明显，使用采样进行语言生成不再是确定性的。词$\text{"car"}$是从条件概率分布$P(w | \text{"The"})$中采样得到的，然后从$P(w | \text{"The"}, \text{"car"})$中采样得到$\text{"drives"}$。在`transformers`中，我们设置`do_sample=True`，并通过`Top-K/Top-P`采样。

采样从一定程度上丰富了文本的生成，但生成的内容未必是连贯通顺的表达，这是在对单词序列进行采样时的一个大问题，模型经常会生成不连贯的胡言乱语，参考[Ari Holtzman et al. (2019)](https://arxiv.org/abs/1904.09751)。

一个技巧是通过降低“温度”（`temperature`）来使分布 $P(w|w_{1:t-1})$ 变得更尖锐（增加高概率单词的概率，降低低概率单词的概率），其中温度是应用于softmax的。温度`temperature`的效果如下图所示：

<center>
<img src="https://github.com/patrickvonplaten/scientific_images/blob/master/sampling_search_with_temp.png?raw=true" width="600px">
<h6>图: Temperature</h6>
</center>

在第 $t=1$ 步的条件下一个词的分布变得更尖锐，几乎没有机会选择单词 "car"。 通过设置 `temperature` 来降低分布的熵：


<center>
<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/0*T-qaxdsUriM5Z5A-" width="800px">
<h6>图: Temperature</h6>
</center>

虽然应用温度可以使分布变得不那么随机，但在极限情况下，当 temperature $ \to 0$ 时，温度缩放采样变得等同于贪婪解码，并且会遇到与之前相同的问题。

## Top-K采样`Top-K Sampling`

### **Top-K Sampling**

[Fan et. al (2018)](https://arxiv.org/pdf/1805.04833.pdf)介绍了一种简单但有效的采样方案，称为**Top-K采样**。在`Top-K`采样中，选择概率最高的`K`个候选词，并将概率重新分配在这`K`个候选词中。`GPT-2`采用了这种采样方案，这也是其在开放故事生成方面取得成功的原因之一。

在上面的例子中，我们将采样步骤中使用的词语范围从`3`个扩展到`10`个，以更好地说明`Top-K`采样。

<center>
<img src="https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/top_k_sampling.png" width="700px">
<h6>图: Top-K sampling</h6>
</center>

假设我们设置`K=6`，在采样步骤中，我们将采样池限制为`6`个词语。在第一个采样步骤中，概率最高的`6`个词语（定义为$V_{\text{top-K}}$）仅包含大约三分之二的概率总权重，但在`Top-K`采样中，它几乎包含了所有的概率质量。在第二个采样步骤中，我们看到`Top-K`成功地消除了一些奇怪的候选词，比如“not”、“the”、“small”和“told”。

然而，对于`Top-K`采样，有一个问题是它无法动态地适应从下一个词的概率分布$P(w|w_{1:t-1})$中过滤出的词语数量。 这可能会带来问题，因为有些词语可能是从非常尖锐的分布中采样的（图中右侧的分布），而其他词语则可能是从更平坦的分布中采样的（图中左侧的分布）。 在$t=1$步骤中，Top-K采样排除了采样诸如“people”、“big”、“house”和“cat”这样合理的候选词。另一方面，在$t=2$步骤中，该方法包括了不太合适的词语“down”和“a”在词语的采样池中。因此，将采样池限制为固定大小的K可能会导致模型在尖锐分布下产生无意义的词语，同时限制了模型在平坦分布下的创造力。这种直觉启发了[Ari Holtzman et al. (2019)](https://arxiv.org/abs/1904.09751)创建了**Top-p (nucleus)**采样方法。

## Top-p采样`Top-P (nucleus) Sampling`

在`Top-p`采样中，不仅仅从最可能的`K`个词中进行采样，而是从一组累积概率超过概率`p`的最小可能的词中进行选择。然后，概率质量在这组词之间重新分配。这样，词组的大小可以根据下一个词的概率分布动态增加和减少。下图是一个直观的可视化展示：

<center>
<img src="https://github.com/patrickvonplaten/scientific_images/blob/master/top_p_sampling.png?raw=true" width="700px">
<h6>图: Top-P sampling</h6>
</center>

设定$p=0.92$，`Top-p`采样选择了超过$p=92%$概率质量的最小词数，称为$V_{\text{top-p}}$。在第一个例子中，这包括了最可能的9个词，而在第二个例子中，只需要选择最可能的3个词就能超过92%。可以看到，当下一个词似乎不太可预测时（例如$P(w | \text{"The"})$），它保留了一系列广泛的词语，而当下一个词似乎更可预测时（例如$P(w | \text{"The", "car"})$），只保留了很少的词语。

虽然从理论上讲，`Top-p`方法比`Top-K`方法更加优雅，但在实践中两种方法都表现良好。`Top-p`方法也可以与`Top-K`方法结合使用，这样可以避免选择排名非常低的词语，同时允许一定程度的动态选择。

## Conclusion

`top-p`和`top-K`采样似乎比传统的贪婪搜索和束搜索在开放式语言生成中产生更流畅的文本。然而，贪婪搜索和束搜索的缺陷（如，生成重复的词序列）是由于模型的训练方式而不是解码方法造成的，参见[Welleck et al. (2019)](https://arxiv.org/pdf/1908.04319.pdf)。此外，[Welleck et al. (2020)](https://arxiv.org/abs/2002.02492)的研究表示，似乎“top-K”和“top-p”采样也会产生重复的词序列。在[Welleck et al. (2019)](https://arxiv.org/pdf/1908.04319.pdf)中，作者通过人工评估表明，当调整模型的训练目标时，束搜索可以生成比`Top-p`采样更流畅的文本。

开放式语言生成是一个快速发展的研究领域，通常情况下，并没有一种适用于所有情况的方法，因此人们必须根据自己的特定用例选择最佳方法。

## Appendix

这里还有一些`generate`方法中未提及的额外参数。我们将在这里简要解释它们！

- `min_length`用于强制模型在达到`min_length`之前不生成`EOS`标记（即不结束句子）。在摘要生成中经常使用此参数，但在一般情况下，如果用户希望生成更长的输出，也会很有用。
- `repetition_penalty`用于惩罚已经生成过的单词或属于上下文的单词。这个惩罚参数最早由[Kesker et al. (2019)](https://arxiv.org/abs/1909.05858)提出，并且在[Welleck et al. (2019)](https://arxiv.org/pdf/1908.04319.pdf)的训练目标中也使用到了。它在防止重复生成方面非常有效，但对于不同的模型和使用情况非常敏感。
- `attention_mask`用于遮蔽填充的标记。
- `pad_token_id`、`bos_token_id`、`eos_token_id`：如果模型默认没有这些特殊标记，用户可以手动选择其他标记ID来表示它们。

*以上内容仅供大模型调参理论参考。*

------
@2024/04/28: 初步修订

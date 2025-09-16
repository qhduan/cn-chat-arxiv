# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models.](http://arxiv.org/abs/2309.01219) | 本文调查了大型语言模型中幻觉的检测、解释和缓解的最新研究，提出了幻觉现象和评估基准的分类，并讨论了未来研究的潜在方向。 |
| [^2] | [Tackling Fake News in Bengali: Unraveling the Impact of Summarization vs. Augmentation on Pre-trained Language Models.](http://arxiv.org/abs/2307.06979) | 本论文研究了孟加拉语中假新闻的检测问题。通过使用总结和扩充技术，结合预训练语言模型，提出了一种四重方法来分类孟加拉语的假新闻文章。研究表明，总结和扩充在孟加拉语假新闻检测中具有有效性。 |
| [^3] | [Explaining Emergent In-Context Learning as Kernel Regression.](http://arxiv.org/abs/2305.12766) | 本文研究了为什么在预训练之后，基于Transformer的语言模型能够实现上下文学习，并提出了一种假设，认为LLMs在面对上下文示例时能够通过内部表示模拟核回归。 |

# 详细

[^1]: AI海洋中的妖怪之歌：大型语言模型中的幻觉调查

    Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models. (arXiv:2309.01219v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2309.01219](http://arxiv.org/abs/2309.01219)

    本文调查了大型语言模型中幻觉的检测、解释和缓解的最新研究，提出了幻觉现象和评估基准的分类，并讨论了未来研究的潜在方向。

    

    尽管大型语言模型（LLMs）在各种下游任务中展示出了卓越的能力，但人们对其产生幻觉的倾向表示担忧：LLMs有时会生成与用户输入不符、与先前生成的内容相矛盾或与已建立的世界知识不符的内容。这种现象对LLMs在现实场景中的可靠性构成了重大挑战。本文对关于幻觉检测、解释和缓解的最新研究进行了调查，重点探讨了LLMs所面临的独特挑战。我们提出了LLM幻觉现象和评估基准的分类，分析了现有的旨在缓解LLM幻觉的方法，并讨论了未来研究的潜在方向。

    While large language models (LLMs) have demonstrated remarkable capabilities across a range of downstream tasks, a significant concern revolves around their propensity to exhibit hallucinations: LLMs occasionally generate content that diverges from the user input, contradicts previously generated context, or misaligns with established world knowledge. This phenomenon poses a substantial challenge to the reliability of LLMs in real-world scenarios. In this paper, we survey recent efforts on the detection, explanation, and mitigation of hallucination, with an emphasis on the unique challenges posed by LLMs. We present taxonomies of the LLM hallucination phenomena and evaluation benchmarks, analyze existing approaches aiming at mitigating LLM hallucination, and discuss potential directions for future research.
    
[^2]: 解决孟加拉语中的假新闻问题：揭示总结与扩充对预训练语言模型的影响

    Tackling Fake News in Bengali: Unraveling the Impact of Summarization vs. Augmentation on Pre-trained Language Models. (arXiv:2307.06979v1 [cs.CL])

    [http://arxiv.org/abs/2307.06979](http://arxiv.org/abs/2307.06979)

    本论文研究了孟加拉语中假新闻的检测问题。通过使用总结和扩充技术，结合预训练语言模型，提出了一种四重方法来分类孟加拉语的假新闻文章。研究表明，总结和扩充在孟加拉语假新闻检测中具有有效性。

    

    随着社交媒体和在线新闻来源的兴起，假新闻已成为全球性的重大问题。然而，在像孟加拉语这样的低资源语言中检测假新闻在研究中受到了有限的关注。本文提出了一种方法，利用总结和扩充技术以及五种预训练语言模型来分类孟加拉语的假新闻文章。我们的方法包括将英语新闻文章进行翻译，并使用扩充技术来解决假新闻文章的不足问题。我们的研究还着重于通过总结新闻来解决基于BERT模型的令牌长度限制。通过广泛的实验和严格的评估，我们展示了总结和扩充在孟加拉语假新闻检测中的有效性。我们使用三个独立的测试数据集来评估我们的模型。当将BanglaBERT基础模型与扩充技术相结合时，取得了令人印象深刻的准确性。

    With the rise of social media and online news sources, fake news has become a significant issue globally. However, the detection of fake news in low resource languages like Bengali has received limited attention in research. In this paper, we propose a methodology consisting of four distinct approaches to classify fake news articles in Bengali using summarization and augmentation techniques with five pre-trained language models. Our approach includes translating English news articles and using augmentation techniques to curb the deficit of fake news articles. Our research also focused on summarizing the news to tackle the token length limitation of BERT based models. Through extensive experimentation and rigorous evaluation, we show the effectiveness of summarization and augmentation in the case of Bengali fake news detection. We evaluated our models using three separate test datasets. The BanglaBERT Base model, when combined with augmentation techniques, achieved an impressive accurac
    
[^3]: 将 Emergent In-Context Learning 解释为核回归

    Explaining Emergent In-Context Learning as Kernel Regression. (arXiv:2305.12766v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.12766](http://arxiv.org/abs/2305.12766)

    本文研究了为什么在预训练之后，基于Transformer的语言模型能够实现上下文学习，并提出了一种假设，认为LLMs在面对上下文示例时能够通过内部表示模拟核回归。

    

    大型语言模型（LLMs）在迁移学习中引起了一场范式转变。与经典的预训练-微调过程相比，为了将LLMs用于下游预测任务，只需要提供一些示例，即上下文示例，而无需添加或更新现有的模型参数。LLMs的这种上下文学习能力非常有意思，但目前尚不完全了解预训练LLMs如何获得这种能力。本文通过提出一个假设，即当面临上下文示例时，LLMs能够通过内部表示模拟核回归，来研究为何基于Transformer的语言模型能够在预训练通用语料库之后实现上下文学习。具体来说，我们首先证明了上下文提示的贝叶斯推断在渐近情况下可以被理解为核回归 $\hat y = \sum_i y_i K(x, x_i)/\sum_i K(x, x_i)$，

    Large language models (LLMs) have initiated a paradigm shift in transfer learning. In contrast to the classic pretraining-then-finetuning procedure, in order to use LLMs for downstream prediction tasks, one only needs to provide a few demonstrations, known as in-context examples, without adding more or updating existing model parameters. This in-context learning (ICL) capability of LLMs is intriguing, and it is not yet fully understood how pretrained LLMs acquire such capabilities. In this paper, we investigate the reason why a transformer-based language model can accomplish in-context learning after pre-training on a general language corpus by proposing one hypothesis that LLMs can simulate kernel regression with internal representations when faced with in-context examples. More concretely, we first prove that Bayesian inference on in-context prompts can be asymptotically understood as kernel regression $\hat y = \sum_i y_i K(x, x_i)/\sum_i K(x, x_i)$ as the number of in-context demon
    


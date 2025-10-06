# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Activation Steering for Robust Type Prediction in CodeLLMs](https://arxiv.org/abs/2404.01903) | 我们提出了一种激活导向技术，通过编辑模型内部激活来改善CodeLLMs在代码类型预测中对于语法干扰的鲁棒性，并成功应用于Python和TypeScript的类型预测，将类型误差率纠正高达90%。 |
| [^2] | [Did Translation Models Get More Robust Without Anyone Even Noticing?](https://arxiv.org/abs/2403.03923) | 最近的研究表明，新的多语言机器翻译模型和大型语言模型在面对各种噪声输入时比先前的模型更加稳健，尽管它们的参数更多、训练过程更复杂，并且没有采用特定设计用于增强稳健性的技术。 |

# 详细

[^1]: 在CodeLLMs中实现类型预测的鲁棒激活导向技术

    Activation Steering for Robust Type Prediction in CodeLLMs

    [https://arxiv.org/abs/2404.01903](https://arxiv.org/abs/2404.01903)

    我们提出了一种激活导向技术，通过编辑模型内部激活来改善CodeLLMs在代码类型预测中对于语法干扰的鲁棒性，并成功应用于Python和TypeScript的类型预测，将类型误差率纠正高达90%。

    

    预训练在代码上的现代LLMs能够成功地完成各种编程任务。然而，它们的性能对语法特征非常敏感，例如变量和类型的名称、代码结构以及类型提示的存在。我们提出了一种推理时技术，使CodeLLMs更能抵御语法干扰因素，这些因素与语义无关。我们的方法依赖于激活导向，涉及编辑内部模型激活以将模型引导到正确的预测。我们通过从突变测试中汲取灵感构建激活向量的新方法，该方法构建最小的破坏语义的代码编辑。相比之下，我们从保留语义的代码编辑中构建激活向量。我们将我们的方法应用于逐渐类型化语言Python和TypeScript的类型预测任务。这种方法可以纠正高达90%的类型错误预测。

    arXiv:2404.01903v1 Announce Type: new  Abstract: Contemporary LLMs pretrained on code are capable of succeeding at a wide variety of programming tasks. However, their performance is very sensitive to syntactic features, such as the names of variables and types, the structure of code, and presence of type hints. We contribute an inference-time technique to make CodeLLMs more robust to syntactic distractors that are semantically irrelevant. Our methodology relies on activation steering, which involves editing internal model activations to steer the model towards the correct prediction. We contribute a novel way to construct steering vectors by taking inspiration from mutation testing, which constructs minimal semantics-breaking code edits. In contrast, we construct steering vectors from semantics-preserving code edits. We apply our approach to the task of type prediction for the gradually typed languages Python and TypeScript. This approach corrects up to 90% of type mispredictions. Fina
    
[^2]: 翻译模型是否在无人发觉的情况下变得更加稳健了？

    Did Translation Models Get More Robust Without Anyone Even Noticing?

    [https://arxiv.org/abs/2403.03923](https://arxiv.org/abs/2403.03923)

    最近的研究表明，新的多语言机器翻译模型和大型语言模型在面对各种噪声输入时比先前的模型更加稳健，尽管它们的参数更多、训练过程更复杂，并且没有采用特定设计用于增强稳健性的技术。

    

    神经机器翻译（MT）模型在各种场景中取得了强大的结果，但普遍认为它们对"嘈杂"输入（如拼写错误、缩写和其他格式问题）非常敏感。本文针对最近的多语言MT模型和应用于机器翻译的大型语言模型（LLMs），重新审视这一观点。有些令人惊讶的是，我们通过受控实验表明，这些模型对许多种噪声比先前的模型更加稳健，即使在干净数据上表现类似。这很引人注目，因为尽管LLMs拥有比过去模型更多的参数和更复杂的训练过程，我们考虑的开源模型中没有一个使用任何专门设计的鼓励稳健性的技术。接下来，我们展示类似的趋势也适用于社交媒体翻译实验——LLMs对社交媒体文本更加稳健。我们还包括了一项关于......

    arXiv:2403.03923v1 Announce Type: new  Abstract: Neural machine translation (MT) models achieve strong results across a variety of settings, but it is widely believed that they are highly sensitive to "noisy" inputs, such as spelling errors, abbreviations, and other formatting issues. In this paper, we revisit this insight in light of recent multilingual MT models and large language models (LLMs) applied to machine translation. Somewhat surprisingly, we show through controlled experiments that these models are far more robust to many kinds of noise than previous models, even when they perform similarly on clean data. This is notable because, even though LLMs have more parameters and more complex training processes than past models, none of the open ones we consider use any techniques specifically designed to encourage robustness. Next, we show that similar trends hold for social media translation experiments -- LLMs are more robust to social media text. We include an analysis of the ci
    


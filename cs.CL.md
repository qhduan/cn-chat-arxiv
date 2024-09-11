# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TRABSA: Interpretable Sentiment Analysis of Tweets using Attention-based BiLSTM and Twitter-RoBERTa](https://arxiv.org/abs/2404.00297) | TRABSA是一个集成了transformer架构、注意力机制和BiLSTM网络的混合框架，利用RoBERTa在大量推特上训练，填补了情感分析领域的差距，实现了94%的准确性和显著的性能提升。 |
| [^2] | [User-LLM: Efficient LLM Contextualization with User Embeddings](https://arxiv.org/abs/2402.13598) | User-LLM框架利用用户嵌入对LLMs进行语境化，使其能够动态适应用户上下文，在各种任务中实现显著性能提升。 |
| [^3] | [Can Large Language Models Learn Independent Causal Mechanisms?](https://arxiv.org/abs/2402.02636) | 本论文研究在大型语言模型中学习独立因果机制的方法，以增强模型在分布变化下的鲁棒性和泛化能力。 |
| [^4] | [Mitigating the Problem of Strong Priors in LMs with Context Extrapolation](https://arxiv.org/abs/2401.17692) | 本论文提出了一种缓解语言模型中强先验问题的新技术，通过削弱原始提示并进行上下文外推，以减少模型受到强先验问题的影响。 |
| [^5] | [The Benefits of a Concise Chain of Thought on Problem-Solving in Large Language Models.](http://arxiv.org/abs/2401.05618) | 本文研究了在大型语言模型中使用简洁的思维链提示对问题求解的影响，实验结果表明简洁性不仅降低了回答长度，且对问题解决性能影响可以忽略。然而在数学问题上有一定的性能下降。这对AI系统工程师和研究人员都有实际意义。 |
| [^6] | [SpikeCLIP: A Contrastive Language-Image Pretrained Spiking Neural Network.](http://arxiv.org/abs/2310.06488) | 本论文引入了一种名为SpikeCLIP的新框架，通过对比语言-图像预训练实现了脉冲神经网络的多模态扩展，并在能源效率和性能方面取得了可比较的结果。 |
| [^7] | [Zero-shot Audio Topic Reranking using Large Language Models.](http://arxiv.org/abs/2309.07606) | 本论文研究了使用大型语言模型的零-shot重新排序方法，以改善基于主题的视频检索性能，无需任何特定任务的训练数据。 |

# 详细

[^1]: TRABSA：使用基于注意力的BiLSTM和Twitter-RoBERTa进行可解释的推文情感分析

    TRABSA: Interpretable Sentiment Analysis of Tweets using Attention-based BiLSTM and Twitter-RoBERTa

    [https://arxiv.org/abs/2404.00297](https://arxiv.org/abs/2404.00297)

    TRABSA是一个集成了transformer架构、注意力机制和BiLSTM网络的混合框架，利用RoBERTa在大量推特上训练，填补了情感分析领域的差距，实现了94%的准确性和显著的性能提升。

    

    情感分析对于理解公众舆论和消费者行为至关重要。现有模型面临着语言多样性、泛化能力和可解释性方面的挑战。我们提出了TRABSA，这是一个集成了基于transformer的架构、注意力机制和BiLSTM网络的混合框架，旨在解决这些挑战。利用在124M条推文上训练的RoBERTa，我们填补了情感分析基准测试中的差距，确保了最先进的准确性。通过将来自32个国家和美国各州的推文与数据集相结合，我们比较了六种词嵌入技术和三种基于词典的标注技术，并选择了最佳技术以实现最佳情感分析效果。TRABSA以94%的准确性和显著的精确度、召回率和F1得分增益，胜过了传统的机器学习和深度学习模型。在不同数据集上的评估显示了一致的优越性和泛化能力。SHAP和LIME分析提高了可解释性，增强了信心。

    arXiv:2404.00297v1 Announce Type: new  Abstract: Sentiment analysis is crucial for understanding public opinion and consumer behavior. Existing models face challenges with linguistic diversity, generalizability, and explainability. We propose TRABSA, a hybrid framework integrating transformer-based architectures, attention mechanisms, and BiLSTM networks to address this. Leveraging RoBERTa-trained on 124M tweets, we bridge gaps in sentiment analysis benchmarks, ensuring state-of-the-art accuracy. Augmenting datasets with tweets from 32 countries and US states, we compare six word-embedding techniques and three lexicon-based labeling techniques, selecting the best for optimal sentiment analysis. TRABSA outperforms traditional ML and deep learning models with 94% accuracy and significant precision, recall, and F1-score gains. Evaluation across diverse datasets demonstrates consistent superiority and generalizability. SHAP and LIME analyses enhance interpretability, improving confidence i
    
[^2]: User-LLM: 利用用户嵌入实现有效的LLM语境化

    User-LLM: Efficient LLM Contextualization with User Embeddings

    [https://arxiv.org/abs/2402.13598](https://arxiv.org/abs/2402.13598)

    User-LLM框架利用用户嵌入对LLMs进行语境化，使其能够动态适应用户上下文，在各种任务中实现显著性能提升。

    

    大语言模型(LLMs)已经彻底改变了自然语言处理。然而，有效地整合复杂且潜在嘈杂的用户交互数据仍然是一个挑战。为了解决这个问题，我们提出了User-LLM，这是一个新颖的框架，利用用户嵌入来对LLMs进行语境化。这些嵌入是通过自监督预训练从各种用户交互中精炼出来的，能够捕捉潜在用户偏好及其随时间的演变。我们通过交叉注意力和软提示将这些用户嵌入与LLMs集成起来，使LLMs能够动态适应用户上下文。我们在MovieLens、亚马逊评论和谷歌本地评论等数据集上进行了全面实验，展示了在各种任务中的显著性能提升。值得注意的是，我们的方法在长序列任务和需要深入理解用户的任务上超过了基于文本提示的语境化，同时在计算上也更加高效。

    arXiv:2402.13598v1 Announce Type: cross  Abstract: Large language models (LLMs) have revolutionized natural language processing. However, effectively incorporating complex and potentially noisy user interaction data remains a challenge. To address this, we propose User-LLM, a novel framework that leverages user embeddings to contextualize LLMs. These embeddings, distilled from diverse user interactions using self-supervised pretraining, capture latent user preferences and their evolution over time. We integrate these user embeddings with LLMs through cross-attention and soft-prompting, enabling LLMs to dynamically adapt to user context. Our comprehensive experiments on MovieLens, Amazon Review, and Google Local Review datasets demonstrate significant performance gains across various tasks. Notably, our approach outperforms text-prompt-based contextualization on long sequence tasks and tasks that require deep user understanding while being computationally efficient. We further incorpora
    
[^3]: 大型语言模型能否学习独立的因果机制？

    Can Large Language Models Learn Independent Causal Mechanisms?

    [https://arxiv.org/abs/2402.02636](https://arxiv.org/abs/2402.02636)

    本论文研究在大型语言模型中学习独立因果机制的方法，以增强模型在分布变化下的鲁棒性和泛化能力。

    

    尽管大型语言模型（LLMs）在语言建模和复杂推理任务中表现出色，但在不常见的环境设置或分布变化的任务中，LLMs的泛化能力仍然不足。目前通常通过增加训练数据来缓解这个问题。然而，这种方法是脆弱的，因为任务的范围可能无法预测或可能会发生变化，并且使用新数据更新模型通常需要大量的额外训练。相反，那些学习抽象变量和因果关系的系统，如因果模型，可以表现出对分布变化的更强稳健性。其中一个原因是存在并使用独立因果机制（ICMs），表示只稀疏交互的高层概念。在这项工作中，我们应用因果性的两个概念，在LLMs中学习ICMs。我们开发了一个由多个稀疏交互的语言模型组成的新LLM架构。

    Despite impressive performance on language modelling and complex reasoning tasks, Large Language Models (LLMs) fall short on the same tasks in uncommon settings or with distribution shifts, exhibiting some lack of generalisation ability. This issue has usually been alleviated by feeding more training data into the LLM. However, this method is brittle, as the scope of tasks may not be readily predictable or may evolve, and updating the model with new data generally requires extensive additional training. By contrast, systems, such as causal models, that learn abstract variables and causal relationships can demonstrate increased robustness against changes in the distribution. One reason for this success is the existence and use of Independent Causal Mechanisms (ICMs) representing high-level concepts that only sparsely interact. In this work, we apply two concepts from causality to learn ICMs within LLMs. We develop a new LLM architecture composed of multiple sparsely interacting language
    
[^4]: 用上下文外推缓解语言模型中强先验问题的方法

    Mitigating the Problem of Strong Priors in LMs with Context Extrapolation

    [https://arxiv.org/abs/2401.17692](https://arxiv.org/abs/2401.17692)

    本论文提出了一种缓解语言模型中强先验问题的新技术，通过削弱原始提示并进行上下文外推，以减少模型受到强先验问题的影响。

    

    语言模型（LMs）已成为各种应用程序中重要的工具，从数据处理到创建指令跟随助手。但是尽管它们有优势，LMs还有一些特殊的局限性，比如“强先验”问题，其中模型会在对某些局部输入的响应中学习输出典型的延续，而不考虑之前的指令。例如，prompt注入攻击可以诱使模型忽略显式指令。在某些情况下，大型模型被证明比类似的较小模型更容易受到这些问题的影响，这是“反向缩放”现象的一个例子。我们开发了一种缓解强先验问题的新技术：我们采用原始指令集，生成原始提示的削弱版本，使其更容易受到强先验问题的影响，然后将延续外推远离削弱的提示。这让我们可以推断模型如何对上下文进行理解并产生输出。

    Language models (LMs) have become important tools in a variety of applications, from data processing to the creation of instruction-following assistants. But despite their advantages, LMs have certain idiosyncratic limitations such as the problem of `strong priors', where a model learns to output typical continuations in response to certain, usually local, portions of the input regardless of any earlier instructions. For example, prompt injection attacks can induce models to ignore explicit directives. In some cases, larger models have been shown to be more susceptible to these problems than similar smaller models, an example of the phenomenon of `inverse scaling'. We develop a new technique for mitigating the problem of strong priors: we take the original set of instructions, produce a weakened version of the original prompt that is even more susceptible to the strong priors problem, and then extrapolate the continuation away from the weakened prompt. This lets us infer how the model 
    
[^5]: 在大型语言模型的问题求解中，简洁的思维链的好处

    The Benefits of a Concise Chain of Thought on Problem-Solving in Large Language Models. (arXiv:2401.05618v1 [cs.CL])

    [http://arxiv.org/abs/2401.05618](http://arxiv.org/abs/2401.05618)

    本文研究了在大型语言模型中使用简洁的思维链提示对问题求解的影响，实验结果表明简洁性不仅降低了回答长度，且对问题解决性能影响可以忽略。然而在数学问题上有一定的性能下降。这对AI系统工程师和研究人员都有实际意义。

    

    在本文中，我们介绍了简洁的思维链(CCoT)提示。我们将标准的CoT和CCoT提示进行比较，以了解简洁性对回答长度和正确答案准确性的影响。我们使用GPT-3.5和GPT-4进行了多项选择问答(MCQA)基准的评估。CCoT将GPT-3.5和GPT-4的平均回答长度分别减少了48.70％，对问题解决性能几乎没有影响。然而，在数学问题上，带有CCoT的GPT-3.5会导致性能下降27.69％。总体而言，CCoT导致每个标记的成本平均降低了22.67％。这些结果对于使用CoT提示工程技术的AI系统工程师来解决真实世界问题的LLM具有实际意义。此外，这些结果为研究LLM中逐步推理的形成行为的AI研究人员提供了更广泛的见解。

    In this paper, we introduce Concise Chain-of-Thought (CCoT) prompting. We compared standard CoT and CCoT prompts to see how conciseness impacts response length and correct-answer accuracy. We evaluated this using GPT-3.5 and GPT-4 with a multiple-choice question-and-answer (MCQA) benchmark. CCoT reduced average response length by 48.70% for both GPT-3.5 and GPT-4 while having a negligible impact on problem-solving performance. However, on math problems, GPT-3.5 with CCoT incurs a performance penalty of 27.69%. Overall, CCoT leads to an average per-token cost reduction of 22.67%. These results have practical implications for AI systems engineers using LLMs to solve real-world problems with CoT prompt-engineering techniques. In addition, these results provide more general insight for AI researchers studying the emergent behavior of step-by-step reasoning in LLMs.
    
[^6]: SpikeCLIP：一种对比语言-图像预训练脉冲神经网络

    SpikeCLIP: A Contrastive Language-Image Pretrained Spiking Neural Network. (arXiv:2310.06488v2 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2310.06488](http://arxiv.org/abs/2310.06488)

    本论文引入了一种名为SpikeCLIP的新框架，通过对比语言-图像预训练实现了脉冲神经网络的多模态扩展，并在能源效率和性能方面取得了可比较的结果。

    

    脉冲神经网络（SNNs）已经证明其在视觉和语言领域中能够实现与深度神经网络（DNNs）相当的性能，同时具有能效提高和符合生物合理性的优势。然而，将这种单模态的SNNs扩展到多模态的情景仍然是一个未开发的领域。受到对比语言-图像预训练（CLIP）概念的启发，我们引入了一个名为SpikeCLIP的新框架，通过“对齐预训练+双损失微调”的两步骤配方，来解决脉冲计算背景下两种模态之间的差距。广泛的实验证明，在常用的用于多模态模型评估的各种数据集上，SNNs取得了与其DNNs对应物相当的结果，同时显著降低了能源消耗。此外，SpikeCLIP在图像分类方面保持了稳定的性能。

    Spiking neural networks (SNNs) have demonstrated the capability to achieve comparable performance to deep neural networks (DNNs) in both visual and linguistic domains while offering the advantages of improved energy efficiency and adherence to biological plausibility. However, the extension of such single-modality SNNs into the realm of multimodal scenarios remains an unexplored territory. Drawing inspiration from the concept of contrastive language-image pre-training (CLIP), we introduce a novel framework, named SpikeCLIP, to address the gap between two modalities within the context of spike-based computing through a two-step recipe involving ``Alignment Pre-training + Dual-Loss Fine-tuning". Extensive experiments demonstrate that SNNs achieve comparable results to their DNN counterparts while significantly reducing energy consumption across a variety of datasets commonly used for multimodal model evaluation. Furthermore, SpikeCLIP maintains robust performance in image classification 
    
[^7]: 使用大型语言模型进行零-shot音频主题重排序

    Zero-shot Audio Topic Reranking using Large Language Models. (arXiv:2309.07606v1 [cs.CL])

    [http://arxiv.org/abs/2309.07606](http://arxiv.org/abs/2309.07606)

    本论文研究了使用大型语言模型的零-shot重新排序方法，以改善基于主题的视频检索性能，无需任何特定任务的训练数据。

    

    多模态视频搜索项目通过使用视频片段作为查询项，而不是传统的文本查询，来研究信息检索。这使得搜索模态更加丰富，例如图像、说话者、内容、主题和情感。这个过程的关键要素是对大型存档的高速、灵活的搜索支持，MVSE通过用嵌入表示视频属性来实现这一点。这项工作旨在通过检查重新排序方法来减少来自快速存档搜索的性能损失。具体而言，研究使用大型语言模型的零-shot 重新排序方法，因为这些方法适用于任何视频存档音频内容。在公开可用的视频存档BBC Rewind语料库上评估了基于主题的检索性能。结果表明，在不需要任何任务特定的训练数据的情况下，重新排序可以实现改进的检索排名。

    The Multimodal Video Search by Examples (MVSE) project investigates using video clips as the query term for information retrieval, rather than the more traditional text query. This enables far richer search modalities such as images, speaker, content, topic, and emotion. A key element for this process is highly rapid, flexible, search to support large archives, which in MVSE is facilitated by representing video attributes by embeddings. This work aims to mitigate any performance loss from this rapid archive search by examining reranking approaches. In particular, zero-shot reranking methods using large language models are investigated as these are applicable to any video archive audio content. Performance is evaluated for topic-based retrieval on a publicly available video archive, the BBC Rewind corpus. Results demonstrate that reranking can achieve improved retrieval ranking without the need for any task-specific training data.
    


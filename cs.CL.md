# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Can Large Language Models Identify Authorship?](https://arxiv.org/abs/2403.08213) | 大型语言模型在作者识别方面的潜力尚未得到充分探索，本文通过全面评估解决了LLMs在作者验证和归属中的三个关键研究问题。 |
| [^2] | [LLM4Decompile: Decompiling Binary Code with Large Language Models](https://arxiv.org/abs/2403.05286) | 发布首批开放访问的反编译LLM，预训练在40亿个C源代码和汇编代码标记上，引入了第一个考虑重新编译性和重新执行性的反编译数据集。 |
| [^3] | [PHAnToM: Personality Has An Effect on Theory-of-Mind Reasoning in Large Language Models](https://arxiv.org/abs/2403.02246) | 通过提示引发特定人格对大型语言模型的心理理论推理能力产生显著影响，特别是来自黑暗三合会的特质对多种LLMs在不同ToM任务中具有较大效应。 |
| [^4] | [Fine-grained and Explainable Factuality Evaluation for Multimodal Summarization](https://arxiv.org/abs/2402.11414) | 提出两种细粒度和可解释的评估框架，用于评估多模态摘要模型的事实性，其中无参考事实性评估框架具有更广泛的应用场景，实验证实了方法的有效性。 |
| [^5] | [DNABERT-S: Learning Species-Aware DNA Embedding with Genome Foundation Models](https://arxiv.org/abs/2402.08777) | DNABERT-S是一种专门用于创建物种感知的DNA嵌入的基因组基础模型。为了提高对长读DNA序列的嵌入效果，引入了Manifold Instance Mixup (MI-Mix)对比目标方法来训练模型。 |
| [^6] | [Improving Reinforcement Learning from Human Feedback with Efficient Reward Model Ensemble.](http://arxiv.org/abs/2401.16635) | 本论文提出一种通过高效的奖励模型集成来改进人工反馈强化学习的方法，以解决由于奖励模型预测不准确而导致RLHF输出与人类价值观不一致的问题。 |
| [^7] | [Interactive Concept Learning for Uncovering Latent Themes in Large Text Collections.](http://arxiv.org/abs/2305.05094) | 本研究提出了一种交互式框架，用于在大型文本集合中揭示潜在的、被领域专家视为相关的概念，既实现了自动化又减少了手动编码的工作量。 |

# 详细

[^1]: 大型语言模型能否识别作者身份？

    Can Large Language Models Identify Authorship?

    [https://arxiv.org/abs/2403.08213](https://arxiv.org/abs/2403.08213)

    大型语言模型在作者识别方面的潜力尚未得到充分探索，本文通过全面评估解决了LLMs在作者验证和归属中的三个关键研究问题。

    

    精准识别作者身份对验证内容真实性和减少误导信息至关重要。 大型语言模型（LLMs）展示了出色的推理和问题解决能力。然而，它们在作者分析（包括作者验证和归属）方面的潜力仍未得到充分探索。 本文对LLMs在这些关键任务中进行了全面评估。 传统研究依赖于手工制作的文体特征，而最先进的方法利用预先训练的语言模型中的文本嵌入。 这些方法通常需要在标记数据上进行微调，然而在跨领域应用中往往表现出性能下降，并提供有限的可解释性。 本文旨在回答三个研究问题：（1）LLMs能否有效执行零样本、端到端的作者验证？（2）LLMs能否准确进行作者身份归属？

    arXiv:2403.08213v1 Announce Type: new  Abstract: The ability to accurately identify authorship is crucial for verifying content authenticity and mitigating misinformation. Large Language Models (LLMs) have demonstrated exceptional capacity for reasoning and problem-solving. However, their potential in authorship analysis, encompassing authorship verification and attribution, remains underexplored. This paper conducts a comprehensive evaluation of LLMs in these critical tasks. Traditional studies have depended on hand-crafted stylistic features, whereas state-of-the-art approaches leverage text embeddings from pre-trained language models. These methods, which typically require fine-tuning on labeled data, often suffer from performance degradation in cross-domain applications and provide limited explainability. This work seeks to address three research questions: (1) Can LLMs perform zero-shot, end-to-end authorship verification effectively? (2) Are LLMs capable of accurately attributing
    
[^2]: LLM4Decompile：使用大型语言模型对二进制代码进行反编译

    LLM4Decompile: Decompiling Binary Code with Large Language Models

    [https://arxiv.org/abs/2403.05286](https://arxiv.org/abs/2403.05286)

    发布首批开放访问的反编译LLM，预训练在40亿个C源代码和汇编代码标记上，引入了第一个考虑重新编译性和重新执行性的反编译数据集。

    

    反编译旨在将编译代码恢复为可读性强的源代码，但在名称和结构等细节方面存在困难。大型语言模型（LLMs）在编程任务中显示出潜力，激发了它们在反编译中的应用。然而，目前尚无用于反编译的开源LLM。此外，现有的反编译评估系统主要考虑标记级准确性，而很大程度上忽略了代码的可执行性，这是任何程序最重要的特征。因此，我们发布了首批开放访问的反编译LLM，范围从10亿到330亿，预先训练了40亿个令牌的C源代码和相应的汇编代码。这些开源LLM可以作为该领域进一步发展的基线。为了确保实际程序评估，我们引入了Decompile-Eval，这是第一个考虑重新编译性和重新执行性的反编译数据集。该基准强调了评估的重要性。

    arXiv:2403.05286v1 Announce Type: cross  Abstract: Decompilation aims to restore compiled code to human-readable source code, but struggles with details like names and structure. Large language models (LLMs) show promise for programming tasks, motivating their application to decompilation. However, there does not exist any open-source LLM for decompilation. Moreover, existing decompilation evaluation systems mainly consider token-level accuracy and largely ignore code executability, which is the most important feature of any program. Therefore, we release the first open-access decompilation LLMs ranging from 1B to 33B pre-trained on 4 billion tokens of C source code and the corresponding assembly code. The open-source LLMs can serve as baselines for further development in the field. To ensure practical program evaluation, we introduce Decompile-Eval, the first dataset that considers re-compilability and re-executability for decompilation. The benchmark emphasizes the importance of eval
    
[^3]: PHAnToM：人格对大型语言模型的心理理论推理产生影响

    PHAnToM: Personality Has An Effect on Theory-of-Mind Reasoning in Large Language Models

    [https://arxiv.org/abs/2403.02246](https://arxiv.org/abs/2403.02246)

    通过提示引发特定人格对大型语言模型的心理理论推理能力产生显著影响，特别是来自黑暗三合会的特质对多种LLMs在不同ToM任务中具有较大效应。

    

    大型语言模型（LLMs）方面的最新进展表明，它们在自然语言处理的许多任务中的能力与甚至优于人类。尽管取得了这一进展，LLMs在社会认知推理方面仍然不足，而人类在这方面天生就很擅长。受到心理学研究中某些人格特质与心理理论（ToM）推理之间联系的启发，以及关于提示工程研究在影响LLMs能力方面的超敏感性的启发，本研究调查了使用提示在LLMs中引发人格如何影响它们的ToM推理能力。我们的研究结果表明，某些引发的人格特质可以显著影响LLMs在三种不同的ToM任务中的推理能力。特别是，来自黑暗三合会(Dark Triad)的特质对于像GPT-3.5、Llama 2和Mistral这样的LLMs在不同的ToM任务中具有较大的变量效应。我们发现，具有某些人格特质的LLMs在执行ToM任务时表现出不同的表现。

    arXiv:2403.02246v1 Announce Type: new  Abstract: Recent advances in large language models (LLMs) demonstrate that their capabilities are comparable, or even superior, to humans in many tasks in natural language processing. Despite this progress, LLMs are still inadequate at social-cognitive reasoning, which humans are naturally good at. Drawing inspiration from psychological research on the links between certain personality traits and Theory-of-Mind (ToM) reasoning, and from prompt engineering research on the hyper-sensitivity of prompts in affecting LLMs capabilities, this study investigates how inducing personalities in LLMs using prompts affects their ToM reasoning capabilities. Our findings show that certain induced personalities can significantly affect the LLMs' reasoning capabilities in three different ToM tasks. In particular, traits from the Dark Triad have a larger variable effect on LLMs like GPT-3.5, Llama 2, and Mistral across the different ToM tasks. We find that LLMs tha
    
[^4]: 用于多模态摘要的细粒度可解释事实评估

    Fine-grained and Explainable Factuality Evaluation for Multimodal Summarization

    [https://arxiv.org/abs/2402.11414](https://arxiv.org/abs/2402.11414)

    提出两种细粒度和可解释的评估框架，用于评估多模态摘要模型的事实性，其中无参考事实性评估框架具有更广泛的应用场景，实验证实了方法的有效性。

    

    多模态摘要旨在生成基于输入文本和图像的简洁摘要。然而，现有方法可能存在事实性输出的问题。为了评估多模态摘要模型的事实性，我们提出了两种细粒度和可解释的评估框架（FALLACIOUS）用于不同的应用场景，即基于参考的事实性评估框架和无参考的事实性评估框架。值得注意的是，无参考事实性评估框架不需要基准真值，因此具有更广泛的应用场景。为了评估所提出框架的有效性，我们计算了我们的框架与其他指标之间的相关性。实验结果显示了我们提出方法的有效性。我们将通过GitHub发布我们的代码和数据集。

    arXiv:2402.11414v1 Announce Type: new  Abstract: Multimodal summarization aims to generate a concise summary based on the input text and image. However, the existing methods potentially suffer from unfactual output. To evaluate the factuality of multimodal summarization models, we propose two fine-grained and explainable evaluation frameworks (FALLACIOUS) for different application scenarios, i.e. reference-based factuality evaluation framework and reference-free factuality evaluation framework. Notably, the reference-free factuality evaluation framework doesn't need ground truth and hence it has a wider application scenario. To evaluate the effectiveness of the proposed frameworks, we compute the correlation between our frameworks and the other metrics. The experimental results show the effectiveness of our proposed method. We will release our code and dataset via github.
    
[^5]: DNABERT-S: 学习具有基因组基础模型的物种感知DNA嵌入

    DNABERT-S: Learning Species-Aware DNA Embedding with Genome Foundation Models

    [https://arxiv.org/abs/2402.08777](https://arxiv.org/abs/2402.08777)

    DNABERT-S是一种专门用于创建物种感知的DNA嵌入的基因组基础模型。为了提高对长读DNA序列的嵌入效果，引入了Manifold Instance Mixup (MI-Mix)对比目标方法来训练模型。

    

    有效的DNA嵌入在基因组分析中仍然至关重要，特别是在缺乏用于模型微调的标记数据的情况下，尽管基因组基础模型已经取得了显著进展。一个典型的例子是宏基因组分箱，这是微生物组研究中的一个关键过程，旨在通过来自可能包含成千上万个不同的、通常没有经过表征的物种的复杂混合DNA序列的物种来对DNA序列进行分组。为了填补有效的DNA嵌入模型的缺陷，我们引入了DNABERT-S，这是一个专门用于创建物种感知的DNA嵌入的基因组基础模型。为了鼓励对易出错的长读DNA序列进行有效嵌入，我们引入了Manifold Instance Mixup(MI-Mix)，一种对比目标，它在随机选择的层次中混合DNA序列的隐藏表示，并训练模型以在输出层识别和区分这些混合比例。

    arXiv:2402.08777v1 Announce Type: cross Abstract: Effective DNA embedding remains crucial in genomic analysis, particularly in scenarios lacking labeled data for model fine-tuning, despite the significant advancements in genome foundation models. A prime example is metagenomics binning, a critical process in microbiome research that aims to group DNA sequences by their species from a complex mixture of DNA sequences derived from potentially thousands of distinct, often uncharacterized species. To fill the lack of effective DNA embedding models, we introduce DNABERT-S, a genome foundation model that specializes in creating species-aware DNA embeddings. To encourage effective embeddings to error-prone long-read DNA sequences, we introduce Manifold Instance Mixup (MI-Mix), a contrastive objective that mixes the hidden representations of DNA sequences at randomly selected layers and trains the model to recognize and differentiate these mixed proportions at the output layer. We further enha
    
[^6]: 通过高效的奖励模型集成改进人工反馈强化学习

    Improving Reinforcement Learning from Human Feedback with Efficient Reward Model Ensemble. (arXiv:2401.16635v1 [cs.LG])

    [http://arxiv.org/abs/2401.16635](http://arxiv.org/abs/2401.16635)

    本论文提出一种通过高效的奖励模型集成来改进人工反馈强化学习的方法，以解决由于奖励模型预测不准确而导致RLHF输出与人类价值观不一致的问题。

    

    人工反馈强化学习（RLHF）是一种广泛使用的方法，用于将大型语言模型与人类价值观对齐。然而，RLHF依赖于通过有限的人类偏好数据训练的奖励模型，这可能导致不准确的预测。因此，RLHF可能产生与人类价值观不一致的输出。为了缓解这个问题，我们提出了一种奖励集成方法，可以使奖励模型做出更准确的预测。考虑到使用基于大型语言模型的奖励模型集成可能具有计算和资源昂贵的问题，我们探索了包括线性层集成和基于LoRA的集成在内的高效集成方法。实证上，我们使用我们的集成奖励模型运行Best-of-$n$和Proximal Policy Optimization，并验证我们的集成方法有助于改善RLHF输出的对齐性能。

    Reinforcement Learning from Human Feedback (RLHF) is a widely adopted approach for aligning large language models with human values. However, RLHF relies on a reward model that is trained with a limited amount of human preference data, which could lead to inaccurate predictions. As a result, RLHF may produce outputs that are misaligned with human values. To mitigate this issue, we contribute a reward ensemble method that allows the reward model to make more accurate predictions. As using an ensemble of large language model-based reward models can be computationally and resource-expensive, we explore efficient ensemble methods including linear-layer ensemble and LoRA-based ensemble. Empirically, we run Best-of-$n$ and Proximal Policy Optimization with our ensembled reward models, and verify that our ensemble methods help improve the alignment performance of RLHF outputs.
    
[^7]: 大型文本集合中的交互式概念学习用于揭示潜在主题

    Interactive Concept Learning for Uncovering Latent Themes in Large Text Collections. (arXiv:2305.05094v1 [cs.CL])

    [http://arxiv.org/abs/2305.05094](http://arxiv.org/abs/2305.05094)

    本研究提出了一种交互式框架，用于在大型文本集合中揭示潜在的、被领域专家视为相关的概念，既实现了自动化又减少了手动编码的工作量。

    

    跨越不同学科领域的专家们通常有兴趣理解大型文本集合。传统上，这个挑战可以通过嘈杂的无监督技术（如主题模型）或手动主题发现流程来处理。在本文中，我们扩展了主题的定义，不仅考虑词分布，还包括被领域专家视为相关的概念。然后，我们提出了一个交互式框架，可以在不同的抽象级别上接收和编码专家反馈。我们的框架在自动化和手动编码之间取得平衡，允许专家控制他们的研究，同时减少所需的手动工作量。

    Experts across diverse disciplines are often interested in making sense of large text collections. Traditionally, this challenge is approached either by noisy unsupervised techniques such as topic models, or by following a manual theme discovery process. In this paper, we expand the definition of a theme to account for more than just a word distribution, and include generalized concepts deemed relevant by domain experts. Then, we propose an interactive framework that receives and encodes expert feedback at different levels of abstraction. Our framework strikes a balance between automation and manual coding, allowing experts to maintain control of their study while reducing the manual effort required.
    


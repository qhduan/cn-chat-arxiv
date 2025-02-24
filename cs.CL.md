# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Chain-of-Action: Faithful and Multimodal Question Answering through Large Language Models](https://arxiv.org/abs/2403.17359) | 提出了Chain-of-Action (CoA)框架，通过新颖的推理-检索机制和多参考忠实分数解决了当前QA应用中的不忠实幻觉和弱推理性能问题 |
| [^2] | [Scaling Laws for Downstream Task Performance of Large Language Models](https://arxiv.org/abs/2402.04177) | 本研究探讨了在转移学习环境中大型语言模型的尺度行为，发现微调数据集的大小和预训练数据与下游数据的分布一致性对下游性能有显著影响。 |
| [^3] | [Peer-review-in-LLMs: Automatic Evaluation Method for LLMs in Open-environment](https://arxiv.org/abs/2402.01830) | 本文提出了一种新的无监督评估方法，利用同行评审机制在开放环境中衡量LLMs。通过为每个LLM分配可学习的能力参数，以最大化各个LLM的能力和得分的一致性。结果表明，高层次的LLM能够更准确地评估其他模型的答案，并能够获得更高的响应得分。 |
| [^4] | [Open the Pandora's Box of LLMs: Jailbreaking LLMs through Representation Engineering](https://arxiv.org/abs/2401.06824) | 通过表示工程对LLMs进行越狱是一种新颖的方法，它利用少量查询对提取“安全模式”，成功规避目标模型的防御，实现了前所未有的越狱性能。 |
| [^5] | [Stack Over-Flowing with Results: The Case for Domain-Specific Pre-Training Over One-Size-Fits-All Models.](http://arxiv.org/abs/2306.03268) | 本文主张在大型预训练模型的潮流中，还应推广面向特定领域的预训练模型，并以 StackOverflow 为例展示了其优越性。 |

# 详细

[^1]: Chain-of-Action：通过大型语言模型实现忠实和多模态问答

    Chain-of-Action: Faithful and Multimodal Question Answering through Large Language Models

    [https://arxiv.org/abs/2403.17359](https://arxiv.org/abs/2403.17359)

    提出了Chain-of-Action (CoA)框架，通过新颖的推理-检索机制和多参考忠实分数解决了当前QA应用中的不忠实幻觉和弱推理性能问题

    

    我们提出了一个称为Chain-of-Action (CoA)的框架，用于多模态和检索增强问答(QA)。与现有文献相比，CoA克服了当前QA应用的两个主要挑战：(i) 与实时或领域事实不一致的不忠实幻觉，以及(ii) 对组合信息的弱推理性能。我们的主要贡献是一种新颖的推理-检索机制，通过系统提示和预设计的动作将复杂问题分解为推理链。在方法上，我们提出了三种领域适应性的“即插即用”操作，用于从异构源检索实时信息。我们还提出了一个多参考忠实分数（MRFS）来验证和解决答案中的冲突。在经验上，我们利用公共基准和一个Web3案例研究来展示CoA相比其他方法的能力。

    arXiv:2403.17359v1 Announce Type: new  Abstract: We present a Chain-of-Action (CoA) framework for multimodal and retrieval-augmented Question-Answering (QA). Compared to the literature, CoA overcomes two major challenges of current QA applications: (i) unfaithful hallucination that is inconsistent with real-time or domain facts and (ii) weak reasoning performance over compositional information. Our key contribution is a novel reasoning-retrieval mechanism that decomposes a complex question into a reasoning chain via systematic prompting and pre-designed actions. Methodologically, we propose three types of domain-adaptable `Plug-and-Play' actions for retrieving real-time information from heterogeneous sources. We also propose a multi-reference faith score (MRFS) to verify and resolve conflicts in the answers. Empirically, we exploit both public benchmarks and a Web3 case study to demonstrate the capability of CoA over other methods.
    
[^2]: 大型语言模型的下游任务性能的尺度律

    Scaling Laws for Downstream Task Performance of Large Language Models

    [https://arxiv.org/abs/2402.04177](https://arxiv.org/abs/2402.04177)

    本研究探讨了在转移学习环境中大型语言模型的尺度行为，发现微调数据集的大小和预训练数据与下游数据的分布一致性对下游性能有显著影响。

    

    尺度律提供了重要的见解，可以指导大型语言模型（LLM）的设计。现有研究主要集中在研究预训练（上游）损失的尺度律。然而，在转移学习环境中，LLM先在无监督数据集上进行预训练，然后在下游任务上进行微调，我们通常也关心下游性能。在这项工作中，我们研究了在转移学习环境中的尺度行为，其中LLM被微调用于机器翻译任务。具体而言，我们研究了预训练数据的选择和大小对下游性能（翻译质量）的影响，使用了两个评价指标：下游交叉熵和BLEU分数。我们的实验证明，微调数据集的大小和预训练数据与下游数据的分布一致性显著影响尺度行为。在充分一致性情况下，下游交叉熵和BLEU分数都会逐渐提升。

    Scaling laws provide important insights that can guide the design of large language models (LLMs). Existing work has primarily focused on studying scaling laws for pretraining (upstream) loss. However, in transfer learning settings, in which LLMs are pretrained on an unsupervised dataset and then finetuned on a downstream task, we often also care about the downstream performance. In this work, we study the scaling behavior in a transfer learning setting, where LLMs are finetuned for machine translation tasks. Specifically, we investigate how the choice of the pretraining data and its size affect downstream performance (translation quality) as judged by two metrics: downstream cross-entropy and BLEU score. Our experiments indicate that the size of the finetuning dataset and the distribution alignment between the pretraining and downstream data significantly influence the scaling behavior. With sufficient alignment, both downstream cross-entropy and BLEU score improve monotonically with 
    
[^3]: LLM中的同行评审方法：开放环境下LLMs的自动评估方法

    Peer-review-in-LLMs: Automatic Evaluation Method for LLMs in Open-environment

    [https://arxiv.org/abs/2402.01830](https://arxiv.org/abs/2402.01830)

    本文提出了一种新的无监督评估方法，利用同行评审机制在开放环境中衡量LLMs。通过为每个LLM分配可学习的能力参数，以最大化各个LLM的能力和得分的一致性。结果表明，高层次的LLM能够更准确地评估其他模型的答案，并能够获得更高的响应得分。

    

    现有的大型语言模型（LLMs）评估方法通常集中于在一些有人工注释的封闭环境和特定领域基准上测试性能。本文探索了一种新颖的无监督评估方法，利用同行评审机制自动衡量LLMs。在这个设置中，开源和闭源的LLMs处于同一环境中，能够回答未标记的问题并互相评估，每个LLM的响应得分由其他匿名的LLMs共同决定。为了获取这些模型之间的能力层次结构，我们为每个LLM分配一个可学习的能力参数来调整最终排序结果。我们将其形式化为一个受约束的优化问题，旨在最大化每个LLM的能力和得分的一致性。背后的关键假设是高层次的LLM能够比低层次的LLM更准确地评估其他模型的答案，而高层次的LLM也可以达到较高的响应得分。

    Existing large language models (LLMs) evaluation methods typically focus on testing the performance on some closed-environment and domain-specific benchmarks with human annotations. In this paper, we explore a novel unsupervised evaluation direction, utilizing peer-review mechanisms to measure LLMs automatically. In this setting, both open-source and closed-source LLMs lie in the same environment, capable of answering unlabeled questions and evaluating each other, where each LLM's response score is jointly determined by other anonymous ones. To obtain the ability hierarchy among these models, we assign each LLM a learnable capability parameter to adjust the final ranking. We formalize it as a constrained optimization problem, intending to maximize the consistency of each LLM's capabilities and scores. The key assumption behind is that high-level LLM can evaluate others' answers more accurately than low-level ones, while higher-level LLM can also achieve higher response scores. Moreover
    
[^4]: 打开LLMs的潘多拉魔盒：通过表示工程对LLMs进行越狱

    Open the Pandora's Box of LLMs: Jailbreaking LLMs through Representation Engineering

    [https://arxiv.org/abs/2401.06824](https://arxiv.org/abs/2401.06824)

    通过表示工程对LLMs进行越狱是一种新颖的方法，它利用少量查询对提取“安全模式”，成功规避目标模型的防御，实现了前所未有的越狱性能。

    

    越狱技术旨在通过诱使大型语言模型（LLMs）生成对恶意查询产生有毒响应，来探索LLMs安全性边界，这在LLMs社区内是一个重要关注点。我们提出一种名为通过表示工程对LLMs进行越狱（Jailbreaking LLMs through Representation Engineering，JRE）的新颖越狱方法，其仅需要少量查询对以提取可用于规避目标模型防御的“安全模式”，实现了前所未有的越狱性能。

    arXiv:2401.06824v2 Announce Type: replace-cross  Abstract: Jailbreaking techniques aim to probe the boundaries of safety in large language models (LLMs) by inducing them to generate toxic responses to malicious queries, a significant concern within the LLM community. While existing jailbreaking methods primarily rely on prompt engineering, altering inputs to evade LLM safety mechanisms, they suffer from low attack success rates and significant time overheads, rendering them inflexible. To overcome these limitations, we propose a novel jailbreaking approach, named Jailbreaking LLMs through Representation Engineering (JRE). Our method requires only a small number of query pairs to extract ``safety patterns'' that can be used to circumvent the target model's defenses, achieving unprecedented jailbreaking performance. Building upon these findings, we also introduce a novel defense framework inspired by JRE principles, which demonstrates notable effectiveness. Extensive experimentation conf
    
[^5]: 面向特定领域的预训练模型：相比一锅粥式模型，千万不要让领域的供给不足受到波及

    Stack Over-Flowing with Results: The Case for Domain-Specific Pre-Training Over One-Size-Fits-All Models. (arXiv:2306.03268v1 [cs.CL])

    [http://arxiv.org/abs/2306.03268](http://arxiv.org/abs/2306.03268)

    本文主张在大型预训练模型的潮流中，还应推广面向特定领域的预训练模型，并以 StackOverflow 为例展示了其优越性。

    

    大型预训练神经语言模型（如OpenAI的GPT系列）为NLP和软件工程带来了极大的进展。然而，我们认为这种追求大而全的潮流应该与针对特定目的、规模适中的预训练模型相结合。本文以StackOverflow为例，展示了我们的面向特定领域的预训练模型相对于通用模型在验证困惑度和迁移学习准确性方面表现更优。

    Large pre-trained neural language models have brought immense progress to both NLP and software engineering. Models in OpenAI's GPT series now dwarf Google's BERT and Meta's RoBERTa, which previously set new benchmarks on a wide range of NLP applications. These models are trained on massive corpora of heterogeneous data from web crawls, which enables them to learn general language patterns and semantic relationships. However, the largest models are both expensive to train and deploy and are often closed-source, so we lack access to their data and design decisions. We argue that this trend towards large, general-purpose models should be complemented with single-purpose, more modestly sized pre-trained models. In this work, we take StackOverflow (SO) as a domain example in which large volumes of rich aligned code and text data is available. We adopt standard practices for pre-training large language models, including using a very large context size (2,048 tokens), batch size (0.5M tokens
    


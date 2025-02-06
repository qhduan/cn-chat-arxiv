# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Ukrainian Texts Classification: Exploration of Cross-lingual Knowledge Transfer Approaches](https://arxiv.org/abs/2404.02043) | 乌克兰文本分类领域探索跨语言知识传递方法，利用最新的NLP技术，测试了在毒性分类、文体分类和自然语言推理任务上的最佳设置。 |
| [^2] | [Diffusion on language model embeddings for protein sequence generation](https://arxiv.org/abs/2403.03726) | 使用DiMA模型，在蛋白语言模型嵌入进行扩散来生成氨基酸序列，比传统解决方案表现更好，并通过设计选择的影响来量化其优越性能。 |
| [^3] | [ImgTrojan: Jailbreaking Vision-Language Models with ONE Image](https://arxiv.org/abs/2403.02910) | 本文提出了一种针对视觉-语言模型的新型越狱攻击，通过在训练数据中插入恶意文本提示，成功实施越狱攻击，并分析了有毒数据比率和可训练参数位置对攻击成功率的影响。 |
| [^4] | [Causal Equal Protection as Algorithmic Fairness](https://arxiv.org/abs/2402.12062) | 本文提出了一种新的算法公平性原则——平等保护，其关键在于将错误分类的风险均等化，避免了许多对传统分类平等原则的反例。 |
| [^5] | [One Graph Model for Cross-domain Dynamic Link Prediction](https://arxiv.org/abs/2402.02168) | DyExpert是一种用于跨域链接预测的动态图模型，通过明确建模历史演化过程并结合链接预测，它可以学习特定下游图的演化模式，并在各个领域上取得了最先进的性能。 |
| [^6] | [PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs](https://arxiv.org/abs/2312.15230) | 本研究中，通过仅更新少部分高度表达力的参数，我们挑战了全参数重新训练的做法，在修剪后恢复或甚至提升了性能。PERP方法显著减少了计算量和存储需求。 |
| [^7] | [A Systematic Literature Review on Explainability for Machine/Deep Learning-based Software Engineering Research.](http://arxiv.org/abs/2401.14617) | 本文通过对机器/深度学习的软件工程领域中可解释性的系统文献综述，总结了XAI技术在软件工程中的应用情况，旨在提高AI模型的可解释性以解决实际部署中的不确定性和风险问题。 |
| [^8] | [HetGPT: Harnessing the Power of Prompt Tuning in Pre-Trained Heterogeneous Graph Neural Networks.](http://arxiv.org/abs/2310.15318) | HetGPT是一种预训练异构图神经网络的方法，通过利用提示调整来解决预训练与下游任务之间的不匹配问题。 |
| [^9] | [Certifying LLM Safety against Adversarial Prompting.](http://arxiv.org/abs/2309.02705) | 本研究提出了首个具有可验证安全保证的框架——消除和检查，用于对抗敌对提示。通过逐个消除标记并使用安全过滤器检查生成的子序列，确保任何敌对修改的有害输入提示都能被正确标识为有害。 |

# 详细

[^1]: 乌克兰文本分类：跨语言知识传递方法的探索

    Ukrainian Texts Classification: Exploration of Cross-lingual Knowledge Transfer Approaches

    [https://arxiv.org/abs/2404.02043](https://arxiv.org/abs/2404.02043)

    乌克兰文本分类领域探索跨语言知识传递方法，利用最新的NLP技术，测试了在毒性分类、文体分类和自然语言推理任务上的最佳设置。

    

    虽然在自然语言处理文本分类领域存在大量标记数据集，但各种语言可用数据的不平衡问题依然显而易见。乌克兰语作为一种仍可从跨语言方法的持续完善中受益的语言。鉴于我们所了解，针对典型文本分类任务，乌克兰语语料库极度匮乏。在这项工作中，我们利用自然语言处理领域的最新进展，探索跨语言知识传递方法，避免手动数据整理：大型多语言编码器和翻译系统、LLMs，以及语言适配器。我们在三个文本分类任务上测试这些方法--毒性分类、文体分类和自然语言推理--提供了最佳设置的"配方"。

    arXiv:2404.02043v1 Announce Type: cross  Abstract: Despite the extensive amount of labeled datasets in the NLP text classification field, the persistent imbalance in data availability across various languages remains evident. Ukrainian, in particular, stands as a language that still can benefit from the continued refinement of cross-lingual methodologies. Due to our knowledge, there is a tremendous lack of Ukrainian corpora for typical text classification tasks. In this work, we leverage the state-of-the-art advances in NLP, exploring cross-lingual knowledge transfer methods avoiding manual data curation: large multilingual encoders and translation systems, LLMs, and language adapters. We test the approaches on three text classification tasks -- toxicity classification, formality classification, and natural language inference -- providing the "recipe" for the optimal setups.
    
[^2]: 蛋白质序列生成的语言模型嵌入扩散

    Diffusion on language model embeddings for protein sequence generation

    [https://arxiv.org/abs/2403.03726](https://arxiv.org/abs/2403.03726)

    使用DiMA模型，在蛋白语言模型嵌入进行扩散来生成氨基酸序列，比传统解决方案表现更好，并通过设计选择的影响来量化其优越性能。

    

    蛋白设计需要对蛋白质宇宙固有复杂性的深入了解。尽管许多工作倾向于有条件的生成或专注于特定蛋白质家族，但无条件生成的基础任务仍未得到充分探索和重视。在这里，我们探索这个关键领域，引入了DiMA，这是一个利用从蛋白语言模型ESM-2衍生的嵌入进行连续扩散以生成氨基酸序列的模型。DiMA超越了包括自回归变换器和离散扩散模型在内的主要解决方案，我们定量地说明了导致其卓越性能的设计选择所带来的影响。我们使用各种指标跨多种形式广泛评估生成序列的质量、多样性、分布相似性和生物相关性。我们的方法始终产生新颖、多样化的蛋白质序列，精准

    arXiv:2403.03726v1 Announce Type: cross  Abstract: Protein design requires a deep understanding of the inherent complexities of the protein universe. While many efforts lean towards conditional generation or focus on specific families of proteins, the foundational task of unconditional generation remains underexplored and undervalued. Here, we explore this pivotal domain, introducing DiMA, a model that leverages continuous diffusion on embeddings derived from the protein language model, ESM-2, to generate amino acid sequences. DiMA surpasses leading solutions, including autoregressive transformer-based and discrete diffusion models, and we quantitatively illustrate the impact of the design choices that lead to its superior performance. We extensively evaluate the quality, diversity, distribution similarity, and biological relevance of the generated sequences using multiple metrics across various modalities. Our approach consistently produces novel, diverse protein sequences that accura
    
[^3]: ImgTrojan: 用一张图片对视觉-语言模型进行越狱

    ImgTrojan: Jailbreaking Vision-Language Models with ONE Image

    [https://arxiv.org/abs/2403.02910](https://arxiv.org/abs/2403.02910)

    本文提出了一种针对视觉-语言模型的新型越狱攻击，通过在训练数据中插入恶意文本提示，成功实施越狱攻击，并分析了有毒数据比率和可训练参数位置对攻击成功率的影响。

    

    近来，对于大型语言模型（LLMs）与人类价值观的对齐引起了越来越多的关注。然而，它们与视觉模块集成的安全问题，即视觉-语言模型（VLMs），仍然相对未被充分探讨。本文提出了一种针对VLMs的新型越狱攻击，旨在当用户输入有害指令时绕过其安全阻碍。假设我们的有毒（图像，文本）数据对包含在训练数据中。通过用恶意越狱提示替换原始文本标题，我们的方法可以利用有毒图像执行越狱攻击。此外，我们分析了有毒比率和可训练参数位置对攻击成功率的影响。为了评估，我们设计了两个度量标准来量化我们攻击的成功率和隐蔽性。结合一系列策划的有害指令，可以衡量攻击的有效性。

    arXiv:2403.02910v1 Announce Type: cross  Abstract: There has been an increasing interest in the alignment of large language models (LLMs) with human values. However, the safety issues of their integration with a vision module, or vision language models (VLMs), remain relatively underexplored. In this paper, we propose a novel jailbreaking attack against VLMs, aiming to bypass their safety barrier when a user inputs harmful instructions. A scenario where our poisoned (image, text) data pairs are included in the training data is assumed. By replacing the original textual captions with malicious jailbreak prompts, our method can perform jailbreak attacks with the poisoned images. Moreover, we analyze the effect of poison ratios and positions of trainable parameters on our attack's success rate. For evaluation, we design two metrics to quantify the success rate and the stealthiness of our attack. Together with a list of curated harmful instructions, a benchmark for measuring attack efficac
    
[^4]: 因果平等保护与算法公平性

    Causal Equal Protection as Algorithmic Fairness

    [https://arxiv.org/abs/2402.12062](https://arxiv.org/abs/2402.12062)

    本文提出了一种新的算法公平性原则——平等保护，其关键在于将错误分类的风险均等化，避免了许多对传统分类平等原则的反例。

    

    过去十年，计算机科学和哲学的文献形成了不同的算法公平性标准。其中最受争议的分类平等要求，预测算法的错误分类在被保护特征所指示的群体中以相等频率发生。尽管分类平等具有直观吸引力，但已受到攻击。我们转向一个相关原则，即平等保护，该原则最初是在刑事司法领域发展起来的。平等保护的关键在于将错误分类的风险（将在规定的意义上具体说明）进行均等化，而不是将错误分类的比率均等化。我们展示了平等保护避免了许多对分类平等的反例。

    arXiv:2402.12062v1 Announce Type: cross  Abstract: Over the last ten years the literature in computer science and philosophy has formulated different criteria of algorithmic fairness. One of the most discussed, classification parity, requires that the erroneous classifications of a predictive algorithm occur with equal frequency for groups picked out by protected characteristics. Despite its intuitive appeal, classification parity has come under attack. Multiple scenarios can be imagined in which - intuitively - a predictive algorithm does not treat any individual unfairly, and yet classification parity is violated. To make progress, we turn to a related principle, equal protection, originally developed in the context of criminal justice. Key to equal protection is equalizing the risks of erroneous classifications (in a sense to be specified) as opposed to equalizing the rates of erroneous classifications. We show that equal protection avoids many of the counterexamples to classificati
    
[^5]: 跨域动态链接预测的一种图模型

    One Graph Model for Cross-domain Dynamic Link Prediction

    [https://arxiv.org/abs/2402.02168](https://arxiv.org/abs/2402.02168)

    DyExpert是一种用于跨域链接预测的动态图模型，通过明确建模历史演化过程并结合链接预测，它可以学习特定下游图的演化模式，并在各个领域上取得了最先进的性能。

    

    本研究提出了DyExpert，一种用于跨域链接预测的动态图模型。它可以明确地建模历史演化过程，学习特定下游图的演化模式，并进而进行特定模式的链接预测。DyExpert采用了解码器优化的transformer，并通过结合演化建模和链接预测的“条件链接生成”实现了高效的并行训练和推断。DyExpert在包含6百万个动态边的广泛动态图上进行训练。在八个未训练的图上进行了大量实验，结果显示DyExpert在跨域链接预测中取得了最先进的性能。与相同设置下的先进基准相比，DyExpert在八个图上的平均精确度提高了11.40％。更令人印象深刻的是，在六个未训练的图上，它超过了八个先进基线的全监督性能。

    This work proposes DyExpert, a dynamic graph model for cross-domain link prediction. It can explicitly model historical evolving processes to learn the evolution pattern of a specific downstream graph and subsequently make pattern-specific link predictions. DyExpert adopts a decode-only transformer and is capable of efficiently parallel training and inference by \textit{conditioned link generation} that integrates both evolution modeling and link prediction. DyExpert is trained by extensive dynamic graphs across diverse domains, comprising 6M dynamic edges. Extensive experiments on eight untrained graphs demonstrate that DyExpert achieves state-of-the-art performance in cross-domain link prediction. Compared to the advanced baseline under the same setting, DyExpert achieves an average of 11.40% improvement Average Precision across eight graphs. More impressive, it surpasses the fully supervised performance of 8 advanced baselines on 6 untrained graphs.
    
[^6]: PERP: 在LLMs时代重新思考修剪-重新训练范式

    PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs

    [https://arxiv.org/abs/2312.15230](https://arxiv.org/abs/2312.15230)

    本研究中，通过仅更新少部分高度表达力的参数，我们挑战了全参数重新训练的做法，在修剪后恢复或甚至提升了性能。PERP方法显著减少了计算量和存储需求。

    

    神经网络可以通过修剪实现高效压缩，显著减少存储和计算需求同时保持预测性能。像迭代幅值修剪（IMP，Han等，2015）这样的简单而有效的方法可以去除不重要的参数，并需要昂贵的重新训练过程以在修剪后恢复性能。然而，随着大型语言模型（LLMs）的兴起，由于内存和计算限制，完全重新训练变得不可行。在本研究中，我们挑战了重新训练所有参数的做法，通过证明只更新少部分高度表达力的参数通常足以恢复甚至提高性能。令人惊讶的是，仅重新训练GPT-结构的0.27%-0.35%的参数即可在不同稀疏水平上实现与一次性IMP相当的性能。我们的方法，即修剪后参数高效重新训练（PERP），大大减少了计算量。

    Neural Networks can be efficiently compressed through pruning, significantly reducing storage and computational demands while maintaining predictive performance. Simple yet effective methods like Iterative Magnitude Pruning (IMP, Han et al., 2015) remove less important parameters and require a costly retraining procedure to recover performance after pruning. However, with the rise of Large Language Models (LLMs), full retraining has become infeasible due to memory and compute constraints. In this study, we challenge the practice of retraining all parameters by demonstrating that updating only a small subset of highly expressive parameters is often sufficient to recover or even improve performance compared to full retraining. Surprisingly, retraining as little as 0.27%-0.35% of the parameters of GPT-architectures achieves comparable performance to One Shot IMP across various sparsity levels. Our approach, Parameter-Efficient Retraining after Pruning (PERP), drastically reduces compute a
    
[^7]: 《基于机器/深度学习的软件工程研究中可解释性的系统文献综述》

    A Systematic Literature Review on Explainability for Machine/Deep Learning-based Software Engineering Research. (arXiv:2401.14617v1 [cs.SE])

    [http://arxiv.org/abs/2401.14617](http://arxiv.org/abs/2401.14617)

    本文通过对机器/深度学习的软件工程领域中可解释性的系统文献综述，总结了XAI技术在软件工程中的应用情况，旨在提高AI模型的可解释性以解决实际部署中的不确定性和风险问题。

    

    人工智能算法，特别是机器学习和深度学习，在软件工程领域取得了显著的成就，并得到了广泛的应用，但由于它们的黑盒特性，这些具有潜力的AI驱动的软件工程模型离实际部署还有很大的差距。这种缺乏可解释性对于在关键任务中应用这些模型，如漏洞检测，决策透明性至关重要，却带来了不必要的风险。本文通过对SE领域中旨在提高AI模型可解释性的方法进行系统文献综述来阐明这个跨学科领域。该综述覆盖了SE和AI学术会议和期刊中出现的研究，涵盖了21个独特的SE任务的63篇论文。基于三个关键的研究问题，我们旨在总结XAI技术在SE任务中的应用情况。

    The remarkable achievements of Artificial Intelligence (AI) algorithms, particularly in Machine Learning (ML) and Deep Learning (DL), have fueled their extensive deployment across multiple sectors, including Software Engineering (SE). However, due to their black-box nature, these promising AI-driven SE models are still far from being deployed in practice. This lack of explainability poses unwanted risks for their applications in critical tasks, such as vulnerability detection, where decision-making transparency is of paramount importance. This paper endeavors to elucidate this interdisciplinary domain by presenting a systematic literature review of approaches that aim to improve the explainability of AI models within the context of SE. The review canvasses work appearing in the most prominent SE & AI conferences and journals, and spans 63 papers across 21 unique SE tasks. Based on three key Research Questions (RQs), we aim to (1) summarize the SE tasks where XAI techniques have shown s
    
[^8]: HetGPT: 利用预训练异构图神经网络中的提示调整的能力

    HetGPT: Harnessing the Power of Prompt Tuning in Pre-Trained Heterogeneous Graph Neural Networks. (arXiv:2310.15318v1 [cs.LG])

    [http://arxiv.org/abs/2310.15318](http://arxiv.org/abs/2310.15318)

    HetGPT是一种预训练异构图神经网络的方法，通过利用提示调整来解决预训练与下游任务之间的不匹配问题。

    

    图表现为表示和分析Web中的复杂模式和丰富信息的自然选择，使得在线页面分类和社交推荐等应用成为可能。然而，当前的“预训练，微调”范式在图机器学习任务中广泛应用，特别是在有限标记节点的情况下，往往存在预训练目标任务与下游任务之间的不匹配问题。这种差距可能导致“负转移”问题，即预训练所获得的知识对下游任务的性能产生不利影响。自然语言处理领域中基于提示的学习的兴起表明了将“预训练，提示”范式应用于图形的潜力，作为一种替代方案。然而，现有的图形提示技术针对的是同质图，忽视了Web图的内在异构性。为了填补这一差距，我们提出了HetGPT，

    Graphs have emerged as a natural choice to represent and analyze the intricate patterns and rich information of the Web, enabling applications such as online page classification and social recommendation. The prevailing "pre-train, fine-tune" paradigm has been widely adopted in graph machine learning tasks, particularly in scenarios with limited labeled nodes. However, this approach often exhibits a misalignment between the training objectives of pretext tasks and those of downstream tasks. This gap can result in the "negative transfer" problem, wherein the knowledge gained from pre-training adversely affects performance in the downstream tasks. The surge in prompt-based learning within Natural Language Processing (NLP) suggests the potential of adapting a "pre-train, prompt" paradigm to graphs as an alternative. However, existing graph prompting techniques are tailored to homogeneous graphs, neglecting the inherent heterogeneity of Web graphs. To bridge this gap, we propose HetGPT, a 
    
[^9]: 证明LLM对抗敌对提示的安全性

    Certifying LLM Safety against Adversarial Prompting. (arXiv:2309.02705v1 [cs.CL])

    [http://arxiv.org/abs/2309.02705](http://arxiv.org/abs/2309.02705)

    本研究提出了首个具有可验证安全保证的框架——消除和检查，用于对抗敌对提示。通过逐个消除标记并使用安全过滤器检查生成的子序列，确保任何敌对修改的有害输入提示都能被正确标识为有害。

    

    为了确保语言模型的输出安全，公开使用的大型语言模型（LLM）引入了所谓的“模型对齐”防护措施。一个对齐的语言模型应该拒绝用户的请求生成有害内容。然而，这种安全措施容易受到敌对提示的攻击，敌对提示包含恶意设计的标记序列，以规避模型的安全防护并导致生成有害内容。在这项工作中，我们介绍了可验证安全保证的第一个对抗敌对提示的框架——消除和检查。我们逐个消除标记，并使用安全过滤器检查生成的子序列。如果安全过滤器检测到任何子序列或输入提示有害，我们的过程将将输入提示标记为有害。这保证了对于某个特定大小的有害输入提示的任何敌对修改也将被标记为有害。我们对抗三种攻击模式：i)敌对后缀，即附加敌对序列…

    Large language models (LLMs) released for public use incorporate guardrails to ensure their output is safe, often referred to as "model alignment." An aligned language model should decline a user's request to produce harmful content. However, such safety measures are vulnerable to adversarial prompts, which contain maliciously designed token sequences to circumvent the model's safety guards and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework to defend against adversarial prompts with verifiable safety guarantees. We erase tokens individually and inspect the resulting subsequences using a safety filter. Our procedure labels the input prompt as harmful if any subsequences or the input prompt are detected as harmful by the filter. This guarantees that any adversarial modification of a harmful prompt up to a certain size is also labeled harmful. We defend against three attack modes: i) adversarial suffix, which appends an adversarial seq
    


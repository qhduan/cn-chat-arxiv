# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [To Label or Not to Label: Hybrid Active Learning for Neural Machine Translation](https://arxiv.org/abs/2403.09259) | 提出了一种用于神经机器翻译的混合主动学习策略HUDS，结合了不确定性和多样性，用于领域自适应的句子选择。 |
| [^2] | [TEncDM: Understanding the Properties of Diffusion Model in the Space of Language Model Encodings](https://arxiv.org/abs/2402.19097) | 通过在语言模型编码空间中训练模型，并使用基于Transformer的解码器以及自我调节，本文提出了名为TEncDM的文本编码扩散模型，在两个文本生成任务上展示了其优越性 |
| [^3] | [SocREval: Large Language Models with the Socratic Method for Reference-Free Reasoning Evaluation.](http://arxiv.org/abs/2310.00074) | 本论文提出了一种称为SocREval的方法，利用GPT-4和苏格拉底方法进行无参考推理评估，以解决当前复杂推理模型评估中遇到的挑战。 |
| [^4] | [Montague semantics and modifier consistency measurement in neural language models.](http://arxiv.org/abs/2212.04310) | 本文提出了一种用于测量神经语言模型合成行为的方法，并从形容词修饰名词短语的角度提出了三个新的合成行为测试。研究结果表明，当前的神经语言模型只在某种程度上符合预期的语言理论。 |

# 详细

[^1]: 是否给数据贴标签：神经机器翻译的混合主动学习

    To Label or Not to Label: Hybrid Active Learning for Neural Machine Translation

    [https://arxiv.org/abs/2403.09259](https://arxiv.org/abs/2403.09259)

    提出了一种用于神经机器翻译的混合主动学习策略HUDS，结合了不确定性和多样性，用于领域自适应的句子选择。

    

    主动学习技术通过从未标记数据中选择更小的代表性子集进行注释，降低了训练神经机器翻译（NMT）模型的标记成本。我们提出了HUDS，这是一种用于NMT领域自适应的混合主动学习策略，将不确定性和多样性相结合，以进行句子选择。

    arXiv:2403.09259v1 Announce Type: new  Abstract: Active learning (AL) techniques reduce labeling costs for training neural machine translation (NMT) models by selecting smaller representative subsets from unlabeled data for annotation. Diversity sampling techniques select heterogeneous instances, while uncertainty sampling methods select instances with the highest model uncertainty. Both approaches have limitations - diversity methods may extract varied but trivial examples, while uncertainty sampling can yield repetitive, uninformative instances. To bridge this gap, we propose HUDS, a hybrid AL strategy for domain adaptation in NMT that combines uncertainty and diversity for sentence selection. HUDS computes uncertainty scores for unlabeled sentences and subsequently stratifies them. It then clusters sentence embeddings within each stratum using k-MEANS and computes diversity scores by distance to the centroid. A weighted hybrid score that combines uncertainty and diversity is then us
    
[^2]: TEncDM: 在语言模型编码空间中理解扩散模型的属性

    TEncDM: Understanding the Properties of Diffusion Model in the Space of Language Model Encodings

    [https://arxiv.org/abs/2402.19097](https://arxiv.org/abs/2402.19097)

    通过在语言模型编码空间中训练模型，并使用基于Transformer的解码器以及自我调节，本文提出了名为TEncDM的文本编码扩散模型，在两个文本生成任务上展示了其优越性

    

    受到扩散模型在各个领域取得成功的启发，许多研究论文提出了将其应用于文本数据的方法。尽管有这些努力，但没有一种方法能够达到大型语言模型的质量。本文对文本扩散模型的关键组件进行了全面分析，并介绍了一种名为Text Encoding Diffusion Model (TEncDM)的新方法。我们在语言模型编码空间中训练我们的模型，而不是通常使用的标记嵌入空间。此外，我们提出使用基于Transformer的解码器，利用上下文信息进行文本重构。我们还分析了自我调节，并发现这会增加模型输出的数量级，从而减少推理阶段的去噪步骤数量。在两个下游文本生成任务QQP和XSum上对TEncDM的评估表明其优越性。

    arXiv:2402.19097v1 Announce Type: new  Abstract: Drawing inspiration from the success of diffusion models in various domains, numerous research papers proposed methods for adapting them to text data. Despite these efforts, none of them has managed to achieve the quality of the large language models. In this paper, we conduct a comprehensive analysis of key components of the text diffusion models and introduce a novel approach named Text Encoding Diffusion Model (TEncDM). Instead of the commonly used token embedding space, we train our model in the space of the language model encodings. Additionally, we propose to use a Transformer-based decoder that utilizes contextual information for text reconstruction. We also analyse self-conditioning and find that it increases the magnitude of the model outputs, allowing the reduction of the number of denoising steps at the inference stage. Evaluation of TEncDM on two downstream text generation tasks, QQP and XSum, demonstrates its superiority ove
    
[^3]: SocREval：使用苏格拉底方法进行无参考推理评估的大规模语言模型

    SocREval: Large Language Models with the Socratic Method for Reference-Free Reasoning Evaluation. (arXiv:2310.00074v1 [cs.CL])

    [http://arxiv.org/abs/2310.00074](http://arxiv.org/abs/2310.00074)

    本论文提出了一种称为SocREval的方法，利用GPT-4和苏格拉底方法进行无参考推理评估，以解决当前复杂推理模型评估中遇到的挑战。

    

    为了全面评估当前模型在复杂推理方面的能力，以可扩展的方式评估它们的逐步推理是至关重要的。现有的基于参考的评估指标依赖于人工注释的推理链来评估模型导出的推理链。然而，这样的“黄金标准”人工编写的推理链可能不是唯一的，并且其获取通常是劳动密集型的。现有的无参考推理指标消除了人工制作推理链的需求作为参考，但通常需要在具有人工推理链的数据集上进行微调，这复杂化了流程并引发了在不同数据集上泛化性的担忧。为了解决这些挑战，我们利用GPT-4自动评估推理链质量，消除了对人工制作参考的需求。利用苏格拉底方法，我们设计了定制化提示来增强无参考推理评估，这就是我们称之为SocREval（苏格拉底方法）的方法。

    To comprehensively assess the capacity of current models for complex reasoning, it is crucial to assess their step-by-step reasoning in a scalable manner. Established reference-based evaluation metrics rely on human-annotated reasoning chains to assess the model-derived chains. However, such ``gold-standard'' human-written reasoning chains may not be unique and their acquisition is often labor-intensive. Existing reference-free reasoning metrics eliminate the need for human-crafted reasoning chains as references, but they typically require fine-tuning on datasets with human-derived reasoning chains, which complicates the process and raises concerns regarding generalizability across diverse datasets. To address these challenges, we harness GPT-4 to automatically evaluate reasoning chain quality, obviating the need for human-crafted references. Leveraging the Socratic method, we devise tailored prompts to enhance reference-free reasoning evaluation, which we term SocREval (Socratic metho
    
[^4]: 蒙塔古语义与神经语言模型中的修饰一致性测量

    Montague semantics and modifier consistency measurement in neural language models. (arXiv:2212.04310v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.04310](http://arxiv.org/abs/2212.04310)

    本文提出了一种用于测量神经语言模型合成行为的方法，并从形容词修饰名词短语的角度提出了三个新的合成行为测试。研究结果表明，当前的神经语言模型只在某种程度上符合预期的语言理论。

    

    在最近几年中，分布式语言表示模型已经取得了巨大的成功。同时，可解释性的需求引发了人们对它们的本质属性和能力的质疑。尤其是，分布式模型在处理自然语言的组合现象时往往不一致，这对它们的安全性和公平性具有重要的影响。尽管如此，目前大多数有关合成性的研究只是针对改善它们在相似性任务上的表现。本研究采取了不同的方法，提出了一种用于测量当代语言模型组成性行为的方法。具体而言，我们关注形容词修饰名词短语中的形容词修饰现象。我们引入了三个灵感来自蒙塔古语意的合成行为测试。我们的实验结果表明，当前的神经语言模型只在某种程度上符合预期的语言理论。

    In recent years, distributional language representation models have demonstrated great practical success. At the same time, the need for interpretability has elicited questions on their intrinsic properties and capabilities. Crucially, distributional models are often inconsistent when dealing with compositional phenomena in natural language, which has significant implications for their safety and fairness. Despite this, most current research on compositionality is directed towards improving their performance on similarity tasks only. This work takes a different approach, and proposes a methodology for measuring compositional behavior in contemporary language models. Specifically, we focus on adjectival modifier phenomena in adjective-noun phrases. We introduce three novel tests of compositional behavior inspired by Montague semantics. Our experimental results indicate that current neural language models behave according to the expected linguistic theories to a limited extent only. This
    


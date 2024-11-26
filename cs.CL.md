# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ZigMa: Zigzag Mamba Diffusion Model](https://arxiv.org/abs/2403.13802) | 本研究提出了一种名为Zigzag Mamba的零参数方法，通过纠正当前Mamba-based视觉方法中对空间连续性的忽视，实现了更好的速度和内存利用，同时在大分辨率视觉数据集上展示了出色的性能。 |
| [^2] | [Reasoning Abilities of Large Language Models: In-Depth Analysis on the Abstraction and Reasoning Corpus](https://arxiv.org/abs/2403.11793) | 使用抽象和推理语料库（ARC）数据集评估大型语言模型的推理和上下文理解能力，结果显示虽然大型语言模型具有较弱的推理能力，但在逻辑连贯性、组合性和效率方面仍然落后，实验结果有助于提出实现人类水平推理的发展路径。 |
| [^3] | [Reawakening knowledge: Anticipatory recovery from catastrophic interference via structured training](https://arxiv.org/abs/2403.09613) | 在结构化环境中依次微调的LLMs表现出预期行为，能够从遗忘中恢复，揭示了在过参数化网络中进行训练的新见解 |
| [^4] | [Emotion Granularity from Text: An Aggregate-Level Indicator of Mental Health](https://arxiv.org/abs/2403.02281) | 提出从社交媒体的文本中计算情绪细粒度的方法，并研究其在心理健康条件中作为指标的有效性 |
| [^5] | [AURA: Natural Language Reasoning for Aleatoric Uncertainty in Rationales](https://arxiv.org/abs/2402.14337) | 提出了在自然语言推理中处理引发模式合理性不确定性的不完美理由的方法，实施了使用熵分数和模型先验信念来指导模型的策略，并在实证中展示了方法相对于敌对理由的稳健性能优势 |
| [^6] | [TinyLLM: Learning a Small Student from Multiple Large Language Models](https://arxiv.org/abs/2402.04616) | TinyLLM是一种从多个大型语言模型中学习小型学生模型的知识蒸馏范式，旨在解决知识多样性有限和缺乏上下文信息等问题，并鼓励学生模型理解答案背后的原理。 |
| [^7] | [Octavius: Mitigating Task Interference in MLLMs via MoE](https://arxiv.org/abs/2311.02684) | 提出了一个名为Octavius的新框架，通过结合MoE和LoRA技术设计了一种新颖的LLM解码器LoRA-MoE，用于多模态学习，实验证明其在各种2D和3D下游任务中具有约20%的改进效果。 |
| [^8] | [Text Classification: A Review, Empirical, and Experimental Evaluation.](http://arxiv.org/abs/2401.12982) | 本论文提出了一种新颖的方法分类法，将文本分类算法层次化地分为精细的类别和具体技术，用以解决现有综述的局限性。 |
| [^9] | [Leveraging Language Models to Detect Greenwashing.](http://arxiv.org/abs/2311.01469) | 本研究引入了一种新的方法，利用语言模型来检测绿色虚假宣传风险。开发了一种量化绿色虚假宣传风险的数学形式，建立了优化的ClimateBERT模型，并进行了结果比较分析。实验表明，我们的方法对于这一任务具有良好的探索方向。 |
| [^10] | [Prompt Injection Attacks and Defenses in LLM-Integrated Applications.](http://arxiv.org/abs/2310.12815) | 本文提出了一个通用框架来形式化提示注入攻击，并系统化防御这种类型的攻击。 |
| [^11] | [Policy-Gradient Training of Language Models for Ranking.](http://arxiv.org/abs/2310.04407) | 该论文提出了一种用于排序的语言模型的策略梯度训练算法Neural PG-RANK，通过将大规模语言模型实例化为Plackett-Luce排名策略，实现了对检索模型的原则性、端到端训练。 |
| [^12] | [Personality Profiling: How informative are social media profiles in predicting personal information?.](http://arxiv.org/abs/2309.13065) | 这项研究探索了利用社交媒体资料预测个人信息的个性化分析模型的准确性和多用途性，并发现支持向量机模型在预测个性类型方面具有最佳准确率，而逻辑回归模型在速度和准确性上表现较好。 |
| [^13] | [Unleashing the Imagination of Text: A Novel Framework for Text-to-image Person Retrieval via Exploring the Power of Words.](http://arxiv.org/abs/2307.09059) | 本研究提出了一个新的框架，通过探索文本中的文字的力量，实现了准确地将抽象的文本描述映射到具体的图像，从而实现了文本到图像的人物检索。 |
| [^14] | [HQP: A Human-Annotated Dataset for Detecting Online Propaganda.](http://arxiv.org/abs/2304.14931) | HQP是一个人工标注的网络宣传检测数据集，与现有的弱标签数据集相比，使用HQP进行训练可以提高44%的准确率。 |
| [^15] | [Deanthropomorphising NLP: Can a Language Model Be Conscious?.](http://arxiv.org/abs/2211.11483) | 本文讨论了关于使用Transformer架构的预训练语言模型LaMDA是否具有意识的说法。作者认为语言模型不可能具有意识，而LaMDA没有比其他类似模型更具先进性。 |

# 详细

[^1]: ZigMa：蜿蜒曼巴扩散模型

    ZigMa: Zigzag Mamba Diffusion Model

    [https://arxiv.org/abs/2403.13802](https://arxiv.org/abs/2403.13802)

    本研究提出了一种名为Zigzag Mamba的零参数方法，通过纠正当前Mamba-based视觉方法中对空间连续性的忽视，实现了更好的速度和内存利用，同时在大分辨率视觉数据集上展示了出色的性能。

    

    扩散模型长期以来一直受到可伸缩性和二次复杂性问题的困扰，特别是在基于变压器的结构内部。在这项研究中，我们旨在利用一种称为曼巴的状态空间模型的长序列建模能力，以扩展其在视觉数据生成中的适用性。首先，我们确定了大多数当前基于曼巴的视觉方法中的一个关键疏忽，即曼巴的扫描方案中缺乏对空间连续性的考虑。其次，基于这一洞察力，我们介绍了一种名为Zigzag Mamba的简单、即插即用、零参数方法，它优于基于曼巴的基线，并表现出比基于变压器的基线更快速和更好的内存利用。最后，我们将Zigzag Mamba集成到随机插值框架中，以研究模型在大分辨率视觉数据集（例如FacesHQ $1024\times 1024$和UCF101，MultiModal-CelebA-HQ）上的可伸缩性。

    arXiv:2403.13802v1 Announce Type: cross  Abstract: The diffusion model has long been plagued by scalability and quadratic complexity issues, especially within transformer-based structures. In this study, we aim to leverage the long sequence modeling capability of a State-Space Model called Mamba to extend its applicability to visual data generation. Firstly, we identify a critical oversight in most current Mamba-based vision methods, namely the lack of consideration for spatial continuity in the scan scheme of Mamba. Secondly, building upon this insight, we introduce a simple, plug-and-play, zero-parameter method named Zigzag Mamba, which outperforms Mamba-based baselines and demonstrates improved speed and memory utilization compared to transformer-based baselines. Lastly, we integrate Zigzag Mamba with the Stochastic Interpolant framework to investigate the scalability of the model on large-resolution visual datasets, such as FacesHQ $1024\times 1024$ and UCF101, MultiModal-CelebA-HQ
    
[^2]: 大型语言模型的推理能力：对抽象和推理语料库的深入分析

    Reasoning Abilities of Large Language Models: In-Depth Analysis on the Abstraction and Reasoning Corpus

    [https://arxiv.org/abs/2403.11793](https://arxiv.org/abs/2403.11793)

    使用抽象和推理语料库（ARC）数据集评估大型语言模型的推理和上下文理解能力，结果显示虽然大型语言模型具有较弱的推理能力，但在逻辑连贯性、组合性和效率方面仍然落后，实验结果有助于提出实现人类水平推理的发展路径。

    

    评估大型语言模型（LLMs）推理能力的现有方法以结果为中心，使得评估推理过程变得困难。我们引入了一种新方法，使用抽象和推理语料库（ARC）数据集以过程为中心的方式评估大型语言模型的推理和上下文理解能力。ARC要求解决问题时具有严谨的逻辑结构，这使得它成为一个能够促进模型推理能力与人类进行比较的基准。实验结果证实，虽然大型语言模型具有较弱的推理能力，但在逻辑连贯性、组合性和效率方面仍然落后。我们的实验突显了LLMs的推理能力，并提出了实现人类水平推理的发展路径。

    arXiv:2403.11793v1 Announce Type: cross  Abstract: The existing methods for evaluating the inference abilities of Large Language Models (LLMs) have been results-centric, making it difficult to assess the inference process. We introduce a new approach using the Abstract and Reasoning Corpus (ARC) dataset to evaluate the inference and contextual understanding abilities of large language models in a process-centric manner. ARC demands rigorous logical structures for problem-solving, making it a benchmark that facilitates the comparison of model inference abilities with humans. Experimental results confirm that while large language models possess weak inference abilities, they still lag in terms of logical coherence, compositionality, and productivity. Our experiments highlight the reasoning capabilities of LLMs, proposing development paths for achieving human-level reasoning.
    
[^3]: 通过结构化训练重新唤醒知识：从灾难性干扰中进行预期性恢复

    Reawakening knowledge: Anticipatory recovery from catastrophic interference via structured training

    [https://arxiv.org/abs/2403.09613](https://arxiv.org/abs/2403.09613)

    在结构化环境中依次微调的LLMs表现出预期行为，能够从遗忘中恢复，揭示了在过参数化网络中进行训练的新见解

    

    我们探讨了神经网络在一个结构化的非独立同分布设置中的训练动态，其中文档以固定重复序列的方式呈现。通常情况下，在一系列文档上训练时，网络会遭受灾难性干扰；然而，我们发现在这种设置下依次微调的LLMs表现出一种奇特且卓越的特性：它们表现出预期的行为，在再次遇到之前的文档时从遗忘中恢复过来。这种行为在架构扩展其参数数量时逐渐出现并变得更加稳健。通过全面的实验和可视化，我们揭示了在结构化环境中训练超参数网络的新见解。

    arXiv:2403.09613v1 Announce Type: cross  Abstract: We explore the training dynamics of neural networks in a structured non-IID setting where documents are presented cyclically in a fixed, repeated sequence. Typically, networks suffer from catastrophic interference when training on a sequence of documents; however, we discover a curious and remarkable property of LLMs fine-tuned sequentially in this setting: they exhibit anticipatory behavior, recovering from the forgetting on documents before encountering them again. The behavior emerges and becomes more robust as the architecture scales up its number of parameters. Through comprehensive experiments and visualizations, we uncover new insights into training over-parameterized networks in structured environments.
    
[^4]: 文本中的情绪细粒度：心理健康的汇总级指标

    Emotion Granularity from Text: An Aggregate-Level Indicator of Mental Health

    [https://arxiv.org/abs/2403.02281](https://arxiv.org/abs/2403.02281)

    提出从社交媒体的文本中计算情绪细粒度的方法，并研究其在心理健康条件中作为指标的有效性

    

    我们在情绪对塑造我们的经历中有共同点，然而，每个人在如何识别、分类和表达情绪方面有很大差异。在心理学中，个体区分情绪概念的能力变化被称为情绪细粒度（通过个体对自己情绪的自我报告来确定）。高情绪细粒度已与更好的心理和身体健康联系在一起；而低情绪细粒度已与应激情绪调节策略和不良健康结果联系在一起。在这项工作中，我们提出从社交媒体中的时间顺序演讲者话语中计算情绪细粒度的计算方法（代替各种偏见的自我报告）。然后我们研究这种文本衍生情绪细粒度措施在作为各种心理健康条件（MHCs）的标记时的有效性。我们建立情绪细粒度的基线措施

    arXiv:2403.02281v1 Announce Type: new  Abstract: We are united in how emotions are central to shaping our experiences; and yet, individuals differ greatly in how we each identify, categorize, and express emotions. In psychology, variation in the ability of individuals to differentiate between emotion concepts is called emotion granularity (determined through self-reports of one's emotions). High emotion granularity has been linked with better mental and physical health; whereas low emotion granularity has been linked with maladaptive emotion regulation strategies and poor health outcomes. In this work, we propose computational measures of emotion granularity derived from temporally-ordered speaker utterances in social media (in lieu of self-reports that suffer from various biases). We then investigate the effectiveness of such text-derived measures of emotion granularity in functioning as markers of various mental health conditions (MHCs). We establish baseline measures of emotion gran
    
[^5]: AURA：自然语言推理中的模式合理性不确定性

    AURA: Natural Language Reasoning for Aleatoric Uncertainty in Rationales

    [https://arxiv.org/abs/2402.14337](https://arxiv.org/abs/2402.14337)

    提出了在自然语言推理中处理引发模式合理性不确定性的不完美理由的方法，实施了使用熵分数和模型先验信念来指导模型的策略，并在实证中展示了方法相对于敌对理由的稳健性能优势

    

    回策背后的理由不仅解释了模型决策，而且提升了语言模型在复杂推理任务上的推理能力。然而，获得无懈可击的理由通常是不可能的。此外，估计理由足够忠实以鼓励模型表现的程度并不是微不足道的。因此，这些推理任务通常迫使模型在不理想的理由下输出正确答案，并且与模型完全有能力的情况相比是次优的。在这项工作中，我们提出了如何应对引发模式合理性不确定性的不完美理由。我们首先用给定理由的熵分数来定义模糊的理由，使用模型先验信念作为信息量。然后根据理由的模糊性来引导模型选择两种不同的推理模型中的一种。我们在实证上论证了我们提出的方法相对于理由的敌对质量产生了稳健的性能优势。

    arXiv:2402.14337v1 Announce Type: new  Abstract: Rationales behind answers not only explain model decisions but boost language models to reason well on complex reasoning tasks. However, obtaining impeccable rationales is often impossible. Besides, it is non-trivial to estimate the degree to which the rationales are faithful enough to encourage model performance. Thus, such reasoning tasks often compel models to output correct answers under undesirable rationales and are sub-optimal compared to what the models are fully capable of. In this work, we propose how to deal with imperfect rationales causing aleatoric uncertainty. We first define the ambiguous rationales with entropy scores of given rationales, using model prior beliefs as informativeness. We then guide models to select one of two different reasoning models according to the ambiguity of rationales. We empirically argue that our proposed method produces robust performance superiority against the adversarial quality of rationale
    
[^6]: TinyLLM: 从多个大型语言模型学习一个小型学生模型

    TinyLLM: Learning a Small Student from Multiple Large Language Models

    [https://arxiv.org/abs/2402.04616](https://arxiv.org/abs/2402.04616)

    TinyLLM是一种从多个大型语言模型中学习小型学生模型的知识蒸馏范式，旨在解决知识多样性有限和缺乏上下文信息等问题，并鼓励学生模型理解答案背后的原理。

    

    将更强大的大型语言模型（LLMs）的推理能力转移到较小的模型上具有吸引力，因为较小的LLMs更灵活，成本更低。在现有的解决方案中，知识蒸馏因其出色的效率和泛化能力而脱颖而出。然而，现有方法存在一些缺点，包括知识多样性有限和缺乏丰富的上下文信息。为了解决这些问题并促进紧凑语言模型的学习，我们提出了TinyLLM，一种从多个大型教师LLMs中学习小型学生LLM的新型知识蒸馏范式。特别地，我们鼓励学生LLM不仅生成正确答案，而且理解这些答案背后的原理。鉴于不同的LLMs具有不同的推理能力，我们引导学生模型吸收来自多个教师LLMs的知识。我们进一步引入了一个上下文示例生成器和一个老师强制模块...

    Transferring the reasoning capability from stronger large language models (LLMs) to smaller ones has been quite appealing, as smaller LLMs are more flexible to deploy with less expense. Among the existing solutions, knowledge distillation stands out due to its outstanding efficiency and generalization. However, existing methods suffer from several drawbacks, including limited knowledge diversity and the lack of rich contextual information. To solve the problems and facilitate the learning of compact language models, we propose TinyLLM, a novel knowledge distillation paradigm to learn a small student LLM from multiple large teacher LLMs. In particular, we encourage the student LLM to not only generate the correct answers but also understand the rationales behind these answers. Given that different LLMs possess diverse reasoning skills, we guide the student model to assimilate knowledge from various teacher LLMs. We further introduce an in-context example generator and a teacher-forcing 
    
[^7]: Octavius：通过MoE减轻MLLM中的任务干扰

    Octavius: Mitigating Task Interference in MLLMs via MoE

    [https://arxiv.org/abs/2311.02684](https://arxiv.org/abs/2311.02684)

    提出了一个名为Octavius的新框架，通过结合MoE和LoRA技术设计了一种新颖的LLM解码器LoRA-MoE，用于多模态学习，实验证明其在各种2D和3D下游任务中具有约20%的改进效果。

    

    最近的研究表明，大型语言模型（LLMs）可以通过指导调整将它们的零-shot泛化能力扩展到多模态学习。随着引入更多的形式和下游任务，负面冲突和干扰可能对性能产生更严重的影响。虽然这种现象在以前的工作中被忽视了，但我们提出了一个名为\mname 的新颖且可扩展的框架，用于与Multimodal Large Language Models（MLLMs）一起进行多模态学习的全面研究和实验。具体来说，我们结合了众所周知的专家混合（MoE）和代表性PEFT技术之一，即LoRA，设计了一种新颖的基于LLM的解码器，称为LoRA-MoE，用于多模态学习。实验结果（约20\%的改进）表明了我们设计在各种2D和3D下游任务中的有效性和多功能性。代码和相应数据集将很快提供。

    arXiv:2311.02684v1 Announce Type: cross  Abstract: Recent studies have demonstrated Large Language Models (LLMs) can extend their zero-shot generalization capabilities to multimodal learning through instruction tuning. As more modalities and downstream tasks are introduced, negative conflicts and interference may have a worse impact on performance. While this phenomenon has been overlooked in previous work, we propose a novel and extensible framework, called \mname, for comprehensive studies and experimentation on multimodal learning with Multimodal Large Language Models (MLLMs). Specifically, we combine the well-known Mixture-of-Experts (MoE) and one of the representative PEFT techniques, \emph{i.e.,} LoRA, designing a novel LLM-based decoder, called LoRA-MoE, for multimodal learning. The experimental results (about 20\% improvement) have shown the effectiveness and versatility of our design in various 2D and 3D downstream tasks. Code and corresponding dataset will be available soon.
    
[^8]: 文本分类：一项回顾、实证和实验评估

    Text Classification: A Review, Empirical, and Experimental Evaluation. (arXiv:2401.12982v1 [cs.CL])

    [http://arxiv.org/abs/2401.12982](http://arxiv.org/abs/2401.12982)

    本论文提出了一种新颖的方法分类法，将文本分类算法层次化地分为精细的类别和具体技术，用以解决现有综述的局限性。

    

    数据的爆炸性和广泛增长使得使用文本分类从大量数据中提取关键信息成为必要。因此，对于经典和深度学习的文本分类方法的研究出现了激增。尽管文献中提出了许多方法，但仍然迫切需要一份全面和最新的综述。现有的综述文章将文本分类算法分为广泛的类别，这可能导致对无关算法的错误分类，以及使用相同度量标准对其质量和行为进行错误评估。为了解决这些局限性，我们的论文引入了一种新颖的方法分类法，将算法层次化地分为精细的类别和具体技术。该分类法包括方法学类别、方法学技术和方法学子技术。我们的研究是首次利用这种方法分类法对算法进行分类的调查。

    The explosive and widespread growth of data necessitates the use of text classification to extract crucial information from vast amounts of data. Consequently, there has been a surge of research in both classical and deep learning text classification methods. Despite the numerous methods proposed in the literature, there is still a pressing need for a comprehensive and up-to-date survey. Existing survey papers categorize algorithms for text classification into broad classes, which can lead to the misclassification of unrelated algorithms and incorrect assessments of their qualities and behaviors using the same metrics. To address these limitations, our paper introduces a novel methodological taxonomy that classifies algorithms hierarchically into fine-grained classes and specific techniques. The taxonomy includes methodology categories, methodology techniques, and methodology sub-techniques. Our study is the first survey to utilize this methodological taxonomy for classifying algorithm
    
[^9]: 利用语言模型检测环保虚假宣传

    Leveraging Language Models to Detect Greenwashing. (arXiv:2311.01469v1 [cs.CL])

    [http://arxiv.org/abs/2311.01469](http://arxiv.org/abs/2311.01469)

    本研究引入了一种新的方法，利用语言模型来检测绿色虚假宣传风险。开发了一种量化绿色虚假宣传风险的数学形式，建立了优化的ClimateBERT模型，并进行了结果比较分析。实验表明，我们的方法对于这一任务具有良好的探索方向。

    

    近年来，气候变化的后果越来越引起公众的关注。因此，企业在可持续发展报告中强调其环保努力以增强公众形象。然而，对此类报告的审核缺乏严格的监管，可能导致绿色虚假宣传。在本研究中，我们引入了一种新的方法来对绿色虚假宣传风险进行训练语言模型。我们的主要贡献包括：开发了一种数学形式来量化绿色虚假宣传风险，提出了一个针对该问题的优化ClimateBERT模型，并进行了结果的比较分析。在一个包含可持续发展报告的测试集上，我们的最佳模型实现了平均准确率86.34%和F1值0.67，表明我们的方法对于这一任务具有探索的良好方向。

    In recent years, climate change repercussions have increasingly captured public interest. Consequently, corporations are emphasizing their environmental efforts in sustainability reports to bolster their public image. Yet, the absence of stringent regulations in review of such reports allows potential greenwashing. In this study, we introduce a novel methodology to train a language model on generated labels for greenwashing risk. Our primary contributions encompass: developing a mathematical formulation to quantify greenwashing risk, a fine-tuned ClimateBERT model for this problem, and a comparative analysis of results. On a test set comprising of sustainability reports, our best model achieved an average accuracy score of 86.34% and F1 score of 0.67, demonstrating that our methods show a promising direction of exploration for this task.
    
[^10]: LLM-集成应用中的提示注入攻击和防御

    Prompt Injection Attacks and Defenses in LLM-Integrated Applications. (arXiv:2310.12815v1 [cs.CR])

    [http://arxiv.org/abs/2310.12815](http://arxiv.org/abs/2310.12815)

    本文提出了一个通用框架来形式化提示注入攻击，并系统化防御这种类型的攻击。

    

    大型语言模型（LLMs）越来越多地用作各种称为LLM-集成应用的实际应用程序的后端。最近的多项研究表明，LLM-集成应用容易受到提示注入攻击的威胁，攻击者可以将恶意指令/数据注入这些应用程序的输入中，以达到攻击者的预期结果。然而，现有的研究仅限于案例研究，缺乏对提示注入攻击及其防御的系统理解。本论文旨在填补这一空白。我们提出了一个通用框架来形式化提示注入攻击，并将研究论文和博客文章中讨论的现有攻击视为我们框架的特例。我们的框架使我们能够通过组合现有攻击设计新的攻击方式。此外，我们还提出了一个系统化提示注入攻击防御的框架。利用我们的框架，我们可以预防和缓解这种类型的攻击。

    Large Language Models (LLMs) are increasingly deployed as the backend for a variety of real-world applications called LLM-Integrated Applications. Multiple recent works showed that LLM-Integrated Applications are vulnerable to prompt injection attacks, in which an attacker injects malicious instruction/data into the input of those applications such that they produce results as the attacker desires. However, existing works are limited to case studies. As a result, the literature lacks a systematic understanding of prompt injection attacks and their defenses. We aim to bridge the gap in this work. In particular, we propose a general framework to formalize prompt injection attacks. Existing attacks, which are discussed in research papers and blog posts, are special cases in our framework. Our framework enables us to design a new attack by combining existing attacks. Moreover, we also propose a framework to systematize defenses against prompt injection attacks. Using our frameworks, we con
    
[^11]: 用于排序的语言模型的策略梯度训练

    Policy-Gradient Training of Language Models for Ranking. (arXiv:2310.04407v1 [cs.CL])

    [http://arxiv.org/abs/2310.04407](http://arxiv.org/abs/2310.04407)

    该论文提出了一种用于排序的语言模型的策略梯度训练算法Neural PG-RANK，通过将大规模语言模型实例化为Plackett-Luce排名策略，实现了对检索模型的原则性、端到端训练。

    

    文本检索在将事实知识纳入到语言处理流程中的决策过程中起着关键作用，从聊天式网页搜索到问答系统。当前最先进的文本检索模型利用预训练的大规模语言模型（LLM）以达到有竞争力的性能，但通过典型的对比损失训练基于LLM的检索器需要复杂的启发式算法，包括选择困难的负样本和使用额外的监督作为学习信号。这种依赖于启发式算法的原因是对比损失本身是启发式的，不能直接优化处理流程末端决策质量的下游指标。为了解决这个问题，我们引入了神经PG-RANK，一种新的训练算法，通过将LLM实例化为Plackett-Luce排名策略，学习排序。神经PG-RANK为检索模型的端到端训练提供了一种原则性方法，作为更大的决策系统的一部分进行训练。

    Text retrieval plays a crucial role in incorporating factual knowledge for decision making into language processing pipelines, ranging from chat-based web search to question answering systems. Current state-of-the-art text retrieval models leverage pre-trained large language models (LLMs) to achieve competitive performance, but training LLM-based retrievers via typical contrastive losses requires intricate heuristics, including selecting hard negatives and using additional supervision as learning signals. This reliance on heuristics stems from the fact that the contrastive loss itself is heuristic and does not directly optimize the downstream metrics of decision quality at the end of the processing pipeline. To address this issue, we introduce Neural PG-RANK, a novel training algorithm that learns to rank by instantiating a LLM as a Plackett-Luce ranking policy. Neural PG-RANK provides a principled method for end-to-end training of retrieval models as part of larger decision systems vi
    
[^12]: 个性化分析：社交媒体资料在预测个人信息方面有多有用？

    Personality Profiling: How informative are social media profiles in predicting personal information?. (arXiv:2309.13065v1 [cs.CL])

    [http://arxiv.org/abs/2309.13065](http://arxiv.org/abs/2309.13065)

    这项研究探索了利用社交媒体资料预测个人信息的个性化分析模型的准确性和多用途性，并发现支持向量机模型在预测个性类型方面具有最佳准确率，而逻辑回归模型在速度和准确性上表现较好。

    

    公司利用个性化分析进行定向广告、政治宣传和疫苗宣传。然而，这些模型的准确性和多用途性仍然相对未知。因此，我们旨在探索人们的在线数字足迹能够被用来分析其迈尔斯-布里格斯人格类型的程度。我们分析和比较了四个模型的结果：逻辑回归、朴素贝叶斯、支持向量机（SVM）和随机森林。我们发现SVM模型在预测某人的完整个性类型方面达到了最佳准确率20.95%。然而，逻辑回归模型的表现只稍微差一些，并且在训练和进行预测时速度更快。我们发现许多标记数据集在社交媒体上呈现出个人特征的严重类别不平衡，包括我们自己的数据集。因此，我们强调需要在报告这些数据集上模型性能时进行仔细考虑。

    Personality profiling has been utilised by companies for targeted advertising, political campaigns and vaccine campaigns. However, the accuracy and versatility of such models still remains relatively unknown. Consequently, we aim to explore the extent to which peoples' online digital footprints can be used to profile their Myers-Briggs personality type. We analyse and compare the results of four models: logistic regression, naive Bayes, support vector machines (SVMs) and random forests. We discover that a SVM model achieves the best accuracy of 20.95% for predicting someones complete personality type. However, logistic regression models perform only marginally worse and are significantly faster to train and perform predictions. We discover that many labelled datasets present substantial class imbalances of personal characteristics on social media, including our own. As a result, we highlight the need for attentive consideration when reporting model performance on these datasets and com
    
[^13]: 文字想象的释放：通过探索文字的力量实现文本到图像的人物检索的新框架

    Unleashing the Imagination of Text: A Novel Framework for Text-to-image Person Retrieval via Exploring the Power of Words. (arXiv:2307.09059v1 [cs.CL])

    [http://arxiv.org/abs/2307.09059](http://arxiv.org/abs/2307.09059)

    本研究提出了一个新的框架，通过探索文本中的文字的力量，实现了准确地将抽象的文本描述映射到具体的图像，从而实现了文本到图像的人物检索。

    

    文本到图像的人物检索的目标是从大型图库中检索与给定文本描述相匹配的人物图像。这个任务的主要挑战在于视觉和文本模态之间信息表示的显著差异。文本模态通过词汇和语法结构传递抽象和精确的信息，而视觉模态通过图像传递具体和直观的信息。为了充分利用文字表示的表达力，准确地将抽象的文本描述映射到具体图像是至关重要的。为了解决这个问题，我们提出了一个新的框架，通过探索句子中的文字的力量，释放了文本到图像人物检索中的文字想象力。具体来说，该框架使用预训练的全面CLIP模型作为图像和文本的双编码器，利用先前的跨模态对齐知识。

    The goal of Text-to-image person retrieval is to retrieve person images from a large gallery that match the given textual descriptions. The main challenge of this task lies in the significant differences in information representation between the visual and textual modalities. The textual modality conveys abstract and precise information through vocabulary and grammatical structures, while the visual modality conveys concrete and intuitive information through images. To fully leverage the expressive power of textual representations, it is essential to accurately map abstract textual descriptions to specific images.  To address this issue, we propose a novel framework to Unleash the Imagination of Text (UIT) in text-to-image person retrieval, aiming to fully explore the power of words in sentences. Specifically, the framework employs the pre-trained full CLIP model as a dual encoder for the images and texts , taking advantage of prior cross-modal alignment knowledge. The Text-guided Imag
    
[^14]: HQP：一份人工标注的用于检测网络宣传的数据集

    HQP: A Human-Annotated Dataset for Detecting Online Propaganda. (arXiv:2304.14931v1 [cs.CL])

    [http://arxiv.org/abs/2304.14931](http://arxiv.org/abs/2304.14931)

    HQP是一个人工标注的网络宣传检测数据集，与现有的弱标签数据集相比，使用HQP进行训练可以提高44%的准确率。

    

    网络宣传对社会的完整性构成了严重威胁。然而，现有的检测网络宣传的数据集存在一个关键限制：它们是使用弱标签进行注释的，可能存在噪音甚至错误。为了解决这一限制，本研究做出了以下贡献：（1）我们提出了一个新的数据集HQP（N=30,000），用于检测网络宣传，具有高质量的标注。据我们所知，这是第一个通过人工注释而创建的用于检测网络宣传的数据集。（2）我们证明了，在使用弱标签进行训练时，最先进的语言模型在检测网络宣传方面失败（AUC：64.03）。相比之下，当使用我们的高质量标签进行训练时，最先进的语言模型可以准确地检测网络宣传（AUC：92.25），提高了约44%。（3）为了解决标注成本问题，我们将我们的工作扩展到了少样本学习。具体来说，我们展示了使用一个小型数据集进行提示式学习的方法。

    Online propaganda poses a severe threat to the integrity of societies. However, existing datasets for detecting online propaganda have a key limitation: they were annotated using weak labels that can be noisy and even incorrect. To address this limitation, our work makes the following contributions: (1) We present \dataset: a novel dataset (N=30,000) for detecting online propaganda with high-quality labels. To the best of our knowledge, \dataset is the first dataset for detecting online propaganda that was created through human annotation. (2) We show empirically that state-of-the-art language models fail in detecting online propaganda when trained with weak labels (AUC: 64.03). In contrast, state-of-the-art language models can accurately detect online propaganda when trained with our high-quality labels (AUC: 92.25), which is an improvement of ~44%. (3) To address the cost of labeling, we extend our work to few-shot learning. Specifically, we show that prompt-based learning using a sm
    
[^15]: Deanthropomorphising NLP：语言模型可以意识到吗？

    Deanthropomorphising NLP: Can a Language Model Be Conscious?. (arXiv:2211.11483v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.11483](http://arxiv.org/abs/2211.11483)

    本文讨论了关于使用Transformer架构的预训练语言模型LaMDA是否具有意识的说法。作者认为语言模型不可能具有意识，而LaMDA没有比其他类似模型更具先进性。

    

    本文旨在对最近有关使用Transformer模型架构的预训练语言模型LaMDA具有意识的说法进行讨论。我们认为这样的语言模型不可能具有意识，而LaMDA并没有比其他类似模型更具先进性。我们通过综合信息理论对Transformer架构进行分析来证明这一点。我们认为这些有意识的说法是NLP报道中使用拟人化语言的更广泛倾向的一部分。无论这些说法的真实性如何，我们认为现在是评估语言建模进展并考虑该任务的伦理影响的适当时机。为了使本文有助于NLP社区以外的读者，我们还提供了一些NLP基础知识的介绍。

    This work is intended as a voice in the discussion over the recent claims that LaMDA, a pretrained language model based on the Transformer model architecture, is sentient. This claim, if confirmed, would have serious ramifications in the Natural Language Processing (NLP) community due to wide-spread use of similar models. However, here we take the position that such a language model cannot be sentient, or conscious, and that LaMDA in particular exhibits no advances over other similar models that would qualify it. We justify this by analysing the Transformer architecture through Integrated Information Theory. We see the claims of consciousness as part of a wider tendency to use anthropomorphic language in NLP reporting. Regardless of the veracity of the claims, we consider this an opportune moment to take stock of progress in language modelling and consider the ethical implications of the task. In order to make this work helpful for readers outside the NLP community, we also present the
    


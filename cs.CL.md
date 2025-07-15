# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An In-depth Evaluation of GPT-4 in Sentence Simplification with Error-based Human Assessment](https://arxiv.org/abs/2403.04963) | 本文深入评估了GPT-4在句子简化中的表现，指出现有自动评估指标和人类评估方法对于大型语言模型的适用性仍有待进一步研究。 |
| [^2] | [PhaseEvo: Towards Unified In-Context Prompt Optimization for Large Language Models](https://arxiv.org/abs/2402.11347) | 该研究提出了PhaseEvo，一个旨在实现提示指令和示例的联合优化的高效自动提示优化框架，结合了LLMs的生成能力和进化算法的全局搜索效率。 |
| [^3] | [Topic Modeling as Multi-Objective Contrastive Optimization](https://arxiv.org/abs/2402.07577) | 该论文介绍了一种新颖的主题建模方法，通过优化对数似然的证据下界和对比学习目标的加权线性组合，将对比主题建模作为一种多目标优化问题，旨在获得能够捕捉共享语义并克服低级别互信息干扰的主题向量集合。 |
| [^4] | [Single Word Change is All You Need: Designing Attacks and Defenses for Text Classifiers](https://arxiv.org/abs/2401.17196) | 本文研究了文本分类中单词扰动的脆弱性问题，并提出了一种新的度量指标以评估分类器的鲁棒性。同时，本文引入了SP-Attack和SP-Defense方法来针对单词扰动进行攻击和防御，实现更高的攻击成功率和更好的句子含义保持。 |
| [^5] | [Cascade Speculative Drafting for Even Faster LLM Inference](https://arxiv.org/abs/2312.11462) | 引入了Cascade Speculative Drafting（CS Drafting）算法，通过垂直级联消除神经模型的自回归生成，通过水平级联优化草稿中的时间分配，从而进一步提高LLM推理效率。 |
| [^6] | [IDEAL: Influence-Driven Selective Annotations Empower In-Context Learners in Large Language Models.](http://arxiv.org/abs/2310.10873) | 本文提出了一种影响驱动的选择性注释方法，用于在大型语言模型中改善上下文学习。该方法通过选择关键的未标记数据子集进行注释，在降低注释成本的同时提高了上下文示例的质量。 |

# 详细

[^1]: 在基于错误的人类评估中深入评估GPT-4在句子简化中的表现

    An In-depth Evaluation of GPT-4 in Sentence Simplification with Error-based Human Assessment

    [https://arxiv.org/abs/2403.04963](https://arxiv.org/abs/2403.04963)

    本文深入评估了GPT-4在句子简化中的表现，指出现有自动评估指标和人类评估方法对于大型语言模型的适用性仍有待进一步研究。

    

    句子简化是一种重写句子以便更易阅读和理解的方法，对于帮助有各种阅读难题的人来说是一种有前途的技术。随着先进大型语言模型（LLMs）的兴起，评估它们在句子简化中的表现变得迫在眉睫。最近的研究利用自动评估指标和人类评估来评估LLMs的简化能力。然而，现有评估方法对LLMs在简化评估中的适用性仍然存在疑问。首先，现有自动指标在LLMs的简化评估中的适用性仍不确定。其次，当前在句子简化中的人类评估方法通常陷入两个极端：要么过于肤浅，无法清晰理解模型的表现，要么过于详细，使注释过程复杂且容易出现不一致性，从而影响评估的可靠性。

    arXiv:2403.04963v1 Announce Type: cross  Abstract: Sentence simplification, which rewrites a sentence to be easier to read and understand, is a promising technique to help people with various reading difficulties. With the rise of advanced large language models (LLMs), evaluating their performance in sentence simplification has become imperative. Recent studies have used both automatic metrics and human evaluations to assess the simplification abilities of LLMs. However, the suitability of existing evaluation methodologies for LLMs remains in question. First, the suitability of current automatic metrics on LLMs' simplification evaluation is still uncertain. Second, current human evaluation approaches in sentence simplification often fall into two extremes: they are either too superficial, failing to offer a clear understanding of the models' performance, or overly detailed, making the annotation process complex and prone to inconsistency, which in turn affects the evaluation's reliabil
    
[^2]: PhaseEvo：面向大型语言模型的统一上下文提示优化

    PhaseEvo: Towards Unified In-Context Prompt Optimization for Large Language Models

    [https://arxiv.org/abs/2402.11347](https://arxiv.org/abs/2402.11347)

    该研究提出了PhaseEvo，一个旨在实现提示指令和示例的联合优化的高效自动提示优化框架，结合了LLMs的生成能力和进化算法的全局搜索效率。

    

    制定大型语言模型（LLMs）的理想提示是一项具有挑战性的任务，需要显著的资源和专业人员的输入。现有工作将优化提示指令和上下文学习示例视为不同问题，导致提示性能次优。本研究通过建立统一的上下文提示优化框架来解决这一局限性，旨在实现提示指令和示例的联合优化。然而，在离散且高维的自然语言空间中制定这种优化引入了收敛和计算效率方面的挑战。为了克服这些问题，我们提出了PhaseEvo，这是一个结合了LLMs的生成能力和进化算法的全局搜索效率的高效自动提示优化框架。我们的框架采用多阶段设计，融合了创新的基于LLMs的变异

    arXiv:2402.11347v1 Announce Type: new  Abstract: Crafting an ideal prompt for Large Language Models (LLMs) is a challenging task that demands significant resources and expert human input. Existing work treats the optimization of prompt instruction and in-context learning examples as distinct problems, leading to sub-optimal prompt performance. This research addresses this limitation by establishing a unified in-context prompt optimization framework, which aims to achieve joint optimization of the prompt instruction and examples. However, formulating such optimization in the discrete and high-dimensional natural language space introduces challenges in terms of convergence and computational efficiency. To overcome these issues, we present PhaseEvo, an efficient automatic prompt optimization framework that combines the generative capability of LLMs with the global search proficiency of evolution algorithms. Our framework features a multi-phase design incorporating innovative LLM-based mut
    
[^3]: 主题建模作为多目标对比优化方法

    Topic Modeling as Multi-Objective Contrastive Optimization

    [https://arxiv.org/abs/2402.07577](https://arxiv.org/abs/2402.07577)

    该论文介绍了一种新颖的主题建模方法，通过优化对数似然的证据下界和对比学习目标的加权线性组合，将对比主题建模作为一种多目标优化问题，旨在获得能够捕捉共享语义并克服低级别互信息干扰的主题向量集合。

    

    最近的表示学习方法通过优化对数似然的证据下界（ELBO）和对比学习目标的加权线性组合来增强神经主题模型。然而，文档级对比学习可能捕捉到低级别的互信息，例如词比例，这会干扰主题建模。此外，ELBO损失旨在记忆输入细节以获得更好的重构质量，而对比损失则试图学习在输入文档之间泛化的主题表示，二者存在潜在冲突。为了解决这些问题，首先我们引入了一种新颖的面向主题向量集合的对比学习方法，以捕捉一组输入文档之间共享的有用语义。其次，我们将对比主题建模明确提出为一个基于梯度的多目标优化问题，目标是实现帕累托平稳解决方案。

    Recent representation learning approaches enhance neural topic models by optimizing the weighted linear combination of the evidence lower bound (ELBO) of the log-likelihood and the contrastive learning objective that contrasts pairs of input documents. However, document-level contrastive learning might capture low-level mutual information, such as word ratio, which disturbs topic modeling. Moreover, there is a potential conflict between the ELBO loss that memorizes input details for better reconstruction quality, and the contrastive loss which attempts to learn topic representations that generalize among input documents. To address these issues, we first introduce a novel contrastive learning method oriented towards sets of topic vectors to capture useful semantics that are shared among a set of input documents. Secondly, we explicitly cast contrastive topic modeling as a gradient-based multi-objective optimization problem, with the goal of achieving a Pareto stationary solution that b
    
[^4]: 一个单词的改变即可：为文本分类器设计攻击与防御策略

    Single Word Change is All You Need: Designing Attacks and Defenses for Text Classifiers

    [https://arxiv.org/abs/2401.17196](https://arxiv.org/abs/2401.17196)

    本文研究了文本分类中单词扰动的脆弱性问题，并提出了一种新的度量指标以评估分类器的鲁棒性。同时，本文引入了SP-Attack和SP-Defense方法来针对单词扰动进行攻击和防御，实现更高的攻击成功率和更好的句子含义保持。

    

    在文本分类中，创建对抗样本意味着在句子中微妙地扰动几个单词而不改变其含义，导致分类器错误分类。令人担忧的是，现有方法生成的对抗样本中有相当部分只改变了一个单词。这种单词扰动的脆弱性代表了分类器的一个重大弱点，恶意用户可以利用它高效地创建大量对抗样本。本文研究了这个问题并作出了以下关键贡献：(1) 我们引入了一种新的度量指标 \r{ho} 来定量评估分类器对于单词扰动的鲁棒性。(2) 我们提出了 SP-Attack，旨在利用单词扰动的脆弱性，实现更高的攻击成功率，更好地保持句子的含义，同时降低与现有对抗方法相比的计算成本。(3) 我们提出了 SP-Defense，旨在改进分类器的抵抗单词扰动的能力，减小攻击效果。

    In text classification, creating an adversarial example means subtly perturbing a few words in a sentence without changing its meaning, causing it to be misclassified by a classifier. A concerning observation is that a significant portion of adversarial examples generated by existing methods change only one word. This single-word perturbation vulnerability represents a significant weakness in classifiers, which malicious users can exploit to efficiently create a multitude of adversarial examples. This paper studies this problem and makes the following key contributions: (1) We introduce a novel metric \r{ho} to quantitatively assess a classifier's robustness against single-word perturbation. (2) We present the SP-Attack, designed to exploit the single-word perturbation vulnerability, achieving a higher attack success rate, better preserving sentence meaning, while reducing computation costs compared to state-of-the-art adversarial methods. (3) We propose SP-Defense, which aims to impro
    
[^5]: 用于更快的LLM推理的级联推测草图

    Cascade Speculative Drafting for Even Faster LLM Inference

    [https://arxiv.org/abs/2312.11462](https://arxiv.org/abs/2312.11462)

    引入了Cascade Speculative Drafting（CS Drafting）算法，通过垂直级联消除神经模型的自回归生成，通过水平级联优化草稿中的时间分配，从而进一步提高LLM推理效率。

    

    引入了增强大型语言模型（LLM）推理效率的级联推测草图，通过较小的模型生成草稿来运作。较大的目标模型然后查看这个草稿以与其输出对齐，目标模型的任何接受都将减少目标模型运行的数量，从而提高效率。然而，在级联推测的草图过程中包括缓慢的自回归生成，并为生成的标记分配相同的时间，而不考虑它们的重要性。这些低效性共同导致级联推测的性能不佳。为了进一步改善LLM推理，我们引入了级联推测草图（CS Drafting），这是一种整合了两种级联类型的推测执行算法。垂直级联从神经模型中消除自回归生成，而水平级联优化了草稿中的时间分配

    arXiv:2312.11462v3 Announce Type: replace-cross  Abstract: Introduced to enhance the efficiency of large language model (LLM) inference, speculative decoding operates by having a smaller model generate a draft. A larger target model then reviews this draft to align with its output, and any acceptance by the target model results in a reduction of the number of the target model runs, ultimately improving efficiency. However, the drafting process in speculative decoding includes slow autoregressive generation and allocates equal time to generating tokens, irrespective of their importance. These inefficiencies collectively contribute to the suboptimal performance of speculative decoding. To further improve LLM inference, we introduce Cascade Speculative Drafting (CS Drafting), a speculative execution algorithm that incorporates two types of cascades. The Vertical Cascade eliminates autoregressive generation from neural models, while the Horizontal Cascade optimizes time allocation in draft
    
[^6]: IDEAL: 强化大型语言模型中上下文学习的影响驱动选择性注释方法

    IDEAL: Influence-Driven Selective Annotations Empower In-Context Learners in Large Language Models. (arXiv:2310.10873v1 [cs.CL])

    [http://arxiv.org/abs/2310.10873](http://arxiv.org/abs/2310.10873)

    本文提出了一种影响驱动的选择性注释方法，用于在大型语言模型中改善上下文学习。该方法通过选择关键的未标记数据子集进行注释，在降低注释成本的同时提高了上下文示例的质量。

    

    上下文学习是一种有前景的范式，它利用上下文示例作为大型语言模型预测的提示。这些提示对于获得强大的性能至关重要。然而，由于这些提示需要从大量注释的示例中进行采样，找到正确的提示可能导致高昂的注释成本。为解决这一挑战，本文引入了一种基于影响驱动的选择性注释方法，旨在在改善上下文示例质量的同时最大程度地降低注释成本。我们的方法的核心是从大规模未标记的数据池中选择一个关键子集进行注释，以用于后续的提示采样。具体地，首先构建一个有向图来表示未标记的数据，然后利用扩散过程量化候选未标记子集的影响力，最后引入一个简单又有效的贪心算法来选择未标记的数据。如果数据提供了最大的影响力，算法就会迭代地选择这些数据。

    In-context learning is a promising paradigm that utilizes in-context examples as prompts for the predictions of large language models. These prompts are crucial for achieving strong performance. However, since the prompts need to be sampled from a large volume of annotated examples, finding the right prompt may result in high annotation costs. To address this challenge, this paper introduces an influence-driven selective annotation method that aims to minimize annotation costs while improving the quality of in-context examples. The essence of our method is to select a pivotal subset from a large-scale unlabeled data pool to annotate for the subsequent sampling of prompts. Specifically, a directed graph is first constructed to represent unlabeled data. Afterward, the influence of candidate unlabeled subsets is quantified with a diffusion process. A simple yet effective greedy algorithm for unlabeled data selection is lastly introduced. It iteratively selects the data if it provides a ma
    


# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Augmenting NER Datasets with LLMs: Towards Automated and Refined Annotation](https://arxiv.org/abs/2404.01334) | 本研究引入了一种新颖的混合标注方法，将人力工作与大型语言模型相结合，旨在提高NER模型的性能，并以成本效益的方式实现这一目标。 |
| [^2] | [What if LLMs Have Different World Views: Simulating Alien Civilizations with LLM-based Agents](https://arxiv.org/abs/2402.13184) | 这项研究引入了“CosmoAgent”，利用LLM模拟人类和外星文明之间的复杂互动，评估和平共存的可行性，并量化评估文明的发展轨迹，同时考虑不同文明之间的巨大多样性。 |
| [^3] | [Text2Data: Low-Resource Data Generation with Textual Control](https://arxiv.org/abs/2402.10941) | Text2Data提出了一种利用未标记数据通过无监督扩散模型来理解基础数据分布的新方法，以解决低资源环境下缺乏文本标签的文本到数据任务中的挑战。 |
| [^4] | [Mathematical Language Models: A Survey](https://arxiv.org/abs/2312.07622) | 该调查论文系统地概述了近年来在数学领域中利用语言模型取得的显著进展，包括对数学LLMs的分类和对超过60个数学数据集的编制，为数学LM领域未来的发展指明了方向。 |
| [^5] | [Exposing Limitations of Language Model Agents in Sequential-Task Compositions on the Web](https://arxiv.org/abs/2311.18751) | 本文介绍了语言模型代理 (LMA) 在多步决策任务上的有希望的范例，在基本任务上具有出色的性能，但在组合任务上表现不佳。通过平衡数据分布，我们训练了一个新模型 HTML-T5++，在现实应用中取得了超越人类的性能，并在新基准测试中实现了最佳零-shot性能。 |
| [^6] | [Preserving the knowledge of long clinical texts using aggregated ensembles of large language models.](http://arxiv.org/abs/2311.01571) | 本文提出了一种使用聚合集成模型的方法来保留长篇临床文本的知识。与以往方法不同，我们将集成学习与文本聚合相结合，并在两个临床预测任务上训练多个大型语言模型。实验证明，我们的方法可以在处理长输入和多样性数据集时提升大型语言模型的性能。 |
| [^7] | [PRD: Peer Rank and Discussion Improve Large Language Model based Evaluations.](http://arxiv.org/abs/2307.02762) | 本研究提出了PRD算法，利用同行评级和讨论改善了基于大型语言模型的评估方法，解决了自我提升和位置偏见等问题。 |
| [^8] | [An investigation of speaker independent phrase break models in End-to-End TTS systems.](http://arxiv.org/abs/2304.04157) | 本文研究了在端到端TTS系统中，加入语调断点预测模型是否有用以及如何衡量其有效性。经过实验验证，使用训练好的语调模型预测断点的故事比未使用预测断点的故事更受欢迎。 |

# 详细

[^1]: 使用LLMs增强NER数据集：迈向自动化和精细化标注

    Augmenting NER Datasets with LLMs: Towards Automated and Refined Annotation

    [https://arxiv.org/abs/2404.01334](https://arxiv.org/abs/2404.01334)

    本研究引入了一种新颖的混合标注方法，将人力工作与大型语言模型相结合，旨在提高NER模型的性能，并以成本效益的方式实现这一目标。

    

    在自然语言处理（NLP）领域，命名实体识别（NER）被认为是一项关键技术，在各种应用中被广泛应用。传统的用于为NER模型标注数据集的方法面临着高成本和数据集质量变化的挑战。本研究介绍了一种新型的混合标注方法，将人力工作与大型语言模型（LLMs）的能力相结合。这种方法不仅旨在改善手动注释中固有的噪音，如遗漏，从而提高NER模型的性能，而且还以一种具有成本效益的方式实现这一目标。此外，通过采用标签混合策略，它解决了LLM-based注释中遇到的类别不平衡问题。通过对多个数据集的分析，这种方法一直表现出比传统注释方法更优异的性能，即使在co

    arXiv:2404.01334v1 Announce Type: new  Abstract: In the field of Natural Language Processing (NLP), Named Entity Recognition (NER) is recognized as a critical technology, employed across a wide array of applications. Traditional methodologies for annotating datasets for NER models are challenged by high costs and variations in dataset quality. This research introduces a novel hybrid annotation approach that synergizes human effort with the capabilities of Large Language Models (LLMs). This approach not only aims to ameliorate the noise inherent in manual annotations, such as omissions, thereby enhancing the performance of NER models, but also achieves this in a cost-effective manner. Additionally, by employing a label mixing strategy, it addresses the issue of class imbalance encountered in LLM-based annotations. Through an analysis across multiple datasets, this method has been consistently shown to provide superior performance compared to traditional annotation methods, even under co
    
[^2]: 如果LLM具有不同的世界观：使用基于LLM的代理模拟外星文明

    What if LLMs Have Different World Views: Simulating Alien Civilizations with LLM-based Agents

    [https://arxiv.org/abs/2402.13184](https://arxiv.org/abs/2402.13184)

    这项研究引入了“CosmoAgent”，利用LLM模拟人类和外星文明之间的复杂互动，评估和平共存的可行性，并量化评估文明的发展轨迹，同时考虑不同文明之间的巨大多样性。

    

    在这项研究中，我们介绍了“CosmoAgent”，这是一个创新的人工智能框架，利用大型语言模型（LLMs）来模拟人类与外星文明之间复杂的交互，特别强调史蒂芬·霍金关于不要随意向宇宙发送无线电信号的谨慎建议。该研究的目标是评估和平共存的可行性，同时考虑可能威胁善意文明的潜在风险。通过采用数学模型和状态转换矩阵，我们的方法定量评估文明的发展轨迹，为在关键增长和饱和点做出未来决策提供见解。此外，本文承认宇宙中潜在生活条件的巨大多样性可能会促进不同文明之间独特的宇宙观、道德准则和世界观。认识到地球上--

    arXiv:2402.13184v1 Announce Type: new  Abstract: In this study, we introduce "CosmoAgent," an innovative artificial intelligence framework utilizing Large Language Models (LLMs) to simulate complex interactions between human and extraterrestrial civilizations, with a special emphasis on Stephen Hawking's cautionary advice about not sending radio signals haphazardly into the universe. The goal is to assess the feasibility of peaceful coexistence while considering potential risks that could threaten well-intentioned civilizations. Employing mathematical models and state transition matrices, our approach quantitatively evaluates the development trajectories of civilizations, offering insights into future decision-making at critical points of growth and saturation. Furthermore, the paper acknowledges the vast diversity in potential living conditions across the universe, which could foster unique cosmologies, ethical codes, and worldviews among various civilizations. Recognizing the Earth-c
    
[^3]: Text2Data：使用文本控制的低资源数据生成

    Text2Data: Low-Resource Data Generation with Textual Control

    [https://arxiv.org/abs/2402.10941](https://arxiv.org/abs/2402.10941)

    Text2Data提出了一种利用未标记数据通过无监督扩散模型来理解基础数据分布的新方法，以解决低资源环境下缺乏文本标签的文本到数据任务中的挑战。

    

    自然语言作为人类与机器无缝交互的一种常见直接控制信号。意识到这一接口的重要性，机器学习社区正在投入大量精力生成与文本指令在语义上一致的数据。虽然在涵盖图像编辑、音频合成、视频生成等领域取得了进展，但低资源领域由于昂贵注释或复杂数据结构（如分子、运动动态和时序）等特点，往往缺乏文本标签。这种不足阻碍了监督学习，从而限制了将先进生成模型应用于文本到数据任务的可能性。为了应对低资源场景中的这些挑战，我们提出了Text2Data，这是一种利用未标记数据通过无监督扩散模型来理解基础数据分布的新方法。

    arXiv:2402.10941v1 Announce Type: cross  Abstract: Natural language serves as a common and straightforward control signal for humans to interact seamlessly with machines. Recognizing the importance of this interface, the machine learning community is investing considerable effort in generating data that is semantically coherent with textual instructions. While strides have been made in text-to-data generation spanning image editing, audio synthesis, video creation, and beyond, low-resource areas characterized by expensive annotations or complex data structures, such as molecules, motion dynamics, and time series, often lack textual labels. This deficiency impedes supervised learning, thereby constraining the application of advanced generative models for text-to-data tasks. In response to these challenges in the low-resource scenario, we propose Text2Data, a novel approach that utilizes unlabeled data to understand the underlying data distribution through an unsupervised diffusion model
    
[^4]: 数学语言模型: 一项调查

    Mathematical Language Models: A Survey

    [https://arxiv.org/abs/2312.07622](https://arxiv.org/abs/2312.07622)

    该调查论文系统地概述了近年来在数学领域中利用语言模型取得的显著进展，包括对数学LLMs的分类和对超过60个数学数据集的编制，为数学LM领域未来的发展指明了方向。

    

    近年来，在数学领域中利用语言模型（LMs），包括预训练语言模型（PLMs）和大规模语言模型（LLMs），取得了显著进展。本文对数学LMs进行了全面调查，系统地从两个不同的视角对重要的研究努力进行了分类：任务和方法论。调查结果显示出大量提出的数学LLMs，进一步划分为指令学习、基于工具的方法、基础CoT技术和高级CoT方法。此外，我们的调查包括编制了60多个数学数据集，包括训练数据集、基准数据集和增强数据集。解决主要挑战，并勾勒数学LM领域未来的发展轨迹，本调查被定位为一个有价值的资源，旨在促进并激励未来的创新。

    arXiv:2312.07622v3 Announce Type: replace  Abstract: In recent years, there has been remarkable progress in leveraging Language Models (LMs), encompassing Pre-trained Language Models (PLMs) and Large-scale Language Models (LLMs), within the domain of mathematics. This paper conducts a comprehensive survey of mathematical LMs, systematically categorizing pivotal research endeavors from two distinct perspectives: tasks and methodologies. The landscape reveals a large number of proposed mathematical LLMs, which are further delineated into instruction learning, tool-based methods, fundamental CoT techniques, and advanced CoT methodologies. In addition, our survey entails the compilation of over 60 mathematical datasets, including training datasets, benchmark datasets, and augmented datasets. Addressing the primary challenges and delineating future trajectories within the field of mathematical LMs, this survey is positioned as a valuable resource, poised to facilitate and inspire future inn
    
[^5]: 在Web上揭示语言模型代理在顺序任务组合中的局限性

    Exposing Limitations of Language Model Agents in Sequential-Task Compositions on the Web

    [https://arxiv.org/abs/2311.18751](https://arxiv.org/abs/2311.18751)

    本文介绍了语言模型代理 (LMA) 在多步决策任务上的有希望的范例，在基本任务上具有出色的性能，但在组合任务上表现不佳。通过平衡数据分布，我们训练了一个新模型 HTML-T5++，在现实应用中取得了超越人类的性能，并在新基准测试中实现了最佳零-shot性能。

    

    最近，语言模型代理(LMA)作为一种在多步决策任务上的有希望的范例出现，通常表现优于人类和其他强化学习代理。尽管有这种希望，但它们在通常涉及任务组合的现实应用中的性能仍未得到充分探索。在这项工作中，我们引入了一个新的基准，叫做CompWoB-反映更现实假设的50个组合性网站自动化任务。我们发现，虽然现有的提示型LMA（gpt-3.5-turbo或gpt-4）在基本任务上实现了94.0％的平均成功率，但在组合任务上降至24.9％的成功率。另一方面，只在基本任务上进行微调的转移性LMA表现出更小的泛化性差距，从85.4％下降到54.8％。通过平衡任务之间的数据分布，我们训练了一个新模型HTML-T5++，在MiniWoB上超过了人类水平的性能（95.2％），并在CompWoB上实现了最佳的零-shot性能（61.5%）。

    Language model agents (LMA) recently emerged as a promising paradigm on muti-step decision making tasks, often outperforming humans and other reinforcement learning agents. Despite the promise, their performance on real-world applications that often involve combinations of tasks is still underexplored. In this work, we introduce a new benchmark, called CompWoB -- 50 new compositional web automation tasks reflecting more realistic assumptions. We show that while existing prompted LMAs (gpt-3.5-turbo or gpt-4) achieve 94.0% average success rate on base tasks, their performance degrades to 24.9% success rate on compositional tasks. On the other hand, transferred LMAs (finetuned only on base tasks) show less generalization gap, dropping from 85.4% to 54.8%. By balancing data distribution across tasks, we train a new model, HTML-T5++, that surpasses human-level performance (95.2%) on MiniWoB, and achieves the best zero-shot performance on CompWoB (61.5%). While these highlight the promise o
    
[^6]: 使用聚合集成模型保留长篇临床文本的知识

    Preserving the knowledge of long clinical texts using aggregated ensembles of large language models. (arXiv:2311.01571v1 [cs.CL])

    [http://arxiv.org/abs/2311.01571](http://arxiv.org/abs/2311.01571)

    本文提出了一种使用聚合集成模型的方法来保留长篇临床文本的知识。与以往方法不同，我们将集成学习与文本聚合相结合，并在两个临床预测任务上训练多个大型语言模型。实验证明，我们的方法可以在处理长输入和多样性数据集时提升大型语言模型的性能。

    

    临床文本，如入院记录、出院小结和进展记录，包含丰富而宝贵的信息，可用于各种临床结果预测任务。然而，将基于BERT的大型语言模型应用于临床文本面临两个主要挑战：输入长度的限制和数据来源的多样性。本文提出了一种新颖的方法，使用聚合集成的大型语言模型来保留长篇临床文本的知识。与以往研究单独使用模型集成或文本聚合方法不同，我们将集成学习与文本聚合相结合，在两个临床结果预测任务（死亡预测和住院天数预测）上训练多个大型语言模型。我们展示了我们的方法可以比基线、独立的集成和聚合效果更好，并且可以在处理长输入和多样性数据集时提高大型语言模型的性能。

    Clinical texts, such as admission notes, discharge summaries, and progress notes, contain rich and valuable information that can be used for various clinical outcome prediction tasks. However, applying large language models, such as BERT-based models, to clinical texts poses two major challenges: the limitation of input length and the diversity of data sources. This paper proposes a novel method to preserve the knowledge of long clinical texts using aggregated ensembles of large language models. Unlike previous studies which use model ensembling or text aggregation methods separately, we combine ensemble learning with text aggregation and train multiple large language models on two clinical outcome tasks: mortality prediction and length of stay prediction. We show that our method can achieve better results than baselines, ensembling, and aggregation individually, and can improve the performance of large language models while handling long inputs and diverse datasets. We conduct extensi
    
[^7]: PRD: 同行评级和讨论改善基于大型语言模型的评估

    PRD: Peer Rank and Discussion Improve Large Language Model based Evaluations. (arXiv:2307.02762v1 [cs.CL])

    [http://arxiv.org/abs/2307.02762](http://arxiv.org/abs/2307.02762)

    本研究提出了PRD算法，利用同行评级和讨论改善了基于大型语言模型的评估方法，解决了自我提升和位置偏见等问题。

    

    如今，评估和比较不同现代大型语言模型（LLMs）生成的回答质量在自动化方面很难。最近的研究建议并主要使用LLMs作为无参考度量衡开放式问题回答的参考指标。更具体地说，他们以被认为是“最强”的LLM作为评估器，对候选模型的答案进行两两比较并提供排名分数。然而，这种直观的方法存在多个问题，例如带来自我提升（青睐自己的答案）和位置偏见。我们从教育领域（Cho and MacArthur, 2011；Walsh, 2014）中汲取见解和教训，改进了基于LLM的评估。具体而言，我们提出了（1）同行评级（PR）算法，该算法考虑每个同行LLM对所有答案对的两两偏好，并输出模型的最终排名；以及（2）同行讨论（PD），在其中我们促使两个LLMs进行讨论并尝试就两个偏好达成共识。

    Nowadays, the quality of responses generated by different modern large language models (LLMs) are hard to evaluate and compare automatically. Recent studies suggest and predominantly use LLMs as a reference-free metric for open-ended question answering. More specifically, they use the recognized "strongest" LLM as the evaluator, which conducts pairwise comparisons of candidate models' answers and provides a ranking score. However, this intuitive method has multiple problems, such as bringing in self-enhancement (favoring its own answers) and positional bias. We draw insights and lessons from the educational domain (Cho and MacArthur, 2011; Walsh, 2014) to improve LLM-based evaluations. Specifically, we propose the (1) peer rank (PR) algorithm that takes into account each peer LLM's pairwise preferences of all answer pairs, and outputs a final ranking of models; and (2) peer discussion (PD), where we prompt two LLMs to discuss and try to reach a mutual agreement on preferences of two an
    
[^8]: 在端到端的TTS系统中，说话人独立语调断点模型的研究

    An investigation of speaker independent phrase break models in End-to-End TTS systems. (arXiv:2304.04157v1 [eess.AS])

    [http://arxiv.org/abs/2304.04157](http://arxiv.org/abs/2304.04157)

    本文研究了在端到端TTS系统中，加入语调断点预测模型是否有用以及如何衡量其有效性。经过实验验证，使用训练好的语调模型预测断点的故事比未使用预测断点的故事更受欢迎。

    

    本文提出了我们对于端到端TTS系统中语调断点预测的研究，研究动机是：（一）在端到端TTS系统中融入明确的语调模型是否有用？（二）如何评估端到端TTS系统的语调模型是否有效？特别地，我们将对儿童故事合成的语境下短语断点预测模型的效用和有效性进行评估，使用的评估指标为听众理解度。我们通过实验听力评估表明，通过使用经过训练的语调模型预测短语断点位置合成的故事比直接合成的故事更受欢迎。

    This paper presents our work on phrase break prediction in the context of end-to-end TTS systems, motivated by the following questions: (i) Is there any utility in incorporating an explicit phrasing model in an end-to-end TTS system?, and (ii) How do you evaluate the effectiveness of a phrasing model in an end-to-end TTS system? In particular, the utility and effectiveness of phrase break prediction models are evaluated in in the context of childrens story synthesis, using listener comprehension. We show by means of perceptual listening evaluations that there is a clear preference for stories synthesized after predicting the location of phrase breaks using a trained phrasing model, over stories directly synthesized without predicting the location of phrase breaks.
    


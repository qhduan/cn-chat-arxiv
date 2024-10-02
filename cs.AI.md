# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Paths to Equilibrium in Normal-Form Games](https://arxiv.org/abs/2403.18079) | 本文研究在多智体强化学习中的策略序列，探讨满足特定约束的策略路径，即满足路径，对于构建终止于均衡策略的路径具有重要意义。 |
| [^2] | [BlendScape: Enabling Unified and Personalized Video-Conferencing Environments through Generative AI](https://arxiv.org/abs/2403.13947) | BlendScape通过生成式人工智能技术实现了用户定制视频会议环境，支持灵活的任务空间表示和多模交互技术，使用户能够快速表达设计意图并设想未来会议中的协作结构。 |
| [^3] | [How Far Are We on the Decision-Making of LLMs? Evaluating LLMs' Gaming Ability in Multi-Agent Environments](https://arxiv.org/abs/2403.11807) | 通过博弈论视角评估LLMs的决策能力，结果表明GPT-3.5在稳健性方面表现良好，但泛化能力有限，而GPT-4则优于其他模型。 |
| [^4] | [GEAR: An Efficient KV Cache Compression Recipefor Near-Lossless Generative Inference of LLM](https://arxiv.org/abs/2403.05527) | GEAR提出了一种高效的KV缓存压缩框架，实现几乎无损的高比率压缩，用于解决大型语言模型推断中因缓存需求增长而导致的记忆绑定问题和性能下降。 |
| [^5] | [How Far Are We from Intelligent Visual Deductive Reasoning?](https://arxiv.org/abs/2403.04732) | 目前的视觉语言模型在文本推理方面表现出色，但在视觉演绎推理方面仍存在较大差距和盲点。 |
| [^6] | [Information-based Transductive Active Learning](https://arxiv.org/abs/2402.15898) | ITL是一种基于信息的转导式学习方法，可以在现实世界设置中自适应采样，以最大化关于指定预测目标的信息获取，并在少样本微调和安全贝叶斯优化应用中显著优于最先进技术。 |
| [^7] | [Shrub of a thousand faces: an individual segmentation from satellite images using deep learning](https://arxiv.org/abs/2401.17985) | 该研究提出了一种新方法，利用深度学习模型从卫星图像中个别描绘位于西班牙内华达山脉林线上的杜松灌木。研究通过整合遥感影像和实例分割模型，解决了检测和描绘自然对象复杂生长模式的挑战。 |
| [^8] | [Mitigating Biases with Diverse Ensembles and Diffusion Models](https://arxiv.org/abs/2311.16176) | 通过利用扩散概率模型（DPMs）生成新特征组合的图像，可以在集成模型中增加模型多样性，并减轻捷径偏见，而无需额外监督信号。 |
| [^9] | [Analyzing the Quality Attributes of AI Vision Models in Open Repositories Under Adversarial Attacks.](http://arxiv.org/abs/2401.12261) | 本文提出了一种综合流程，用于对AI视觉模型在开放存储库中的质量属性进行分析，尤其是在面对对抗攻击时的表现。我们展示了一个涉及六个计算机视觉模型的评估场景，以评估准确性、鲁棒性、解释效用和开销。 |
| [^10] | [Universal Vulnerabilities in Large Language Models: In-context Learning Backdoor Attacks.](http://arxiv.org/abs/2401.05949) | 本研究发现上下文学习范式在大型语言模型中存在漏洞，攻击者可以通过污染示范上下文来操控模型行为，而无需进行微调。这项研究设计了一种名为ICLAttack的后门攻击方法，可以通过污染示范样本和提示来使模型按照预定义的意图行事。 |
| [^11] | [Closed-Form Interpretation of Neural Network Classifiers with Symbolic Regression Gradients.](http://arxiv.org/abs/2401.04978) | 本文提出了一种解释神经网络分类器的闭式表示的方法，使其适用于自动化科学发现。这种方法通过将神经网络嵌入到一组基于相同量的等价类中，并通过找到该等价类与符号回归搜索空间中的方程的交集来解释神经网络。 |
| [^12] | [Quantifying degeneracy in singular models via the learning coefficient.](http://arxiv.org/abs/2308.12108) | 这项工作介绍了一种称为学习系数的量，用于精确量化深度神经网络中的退化程度，该方法能够区分不同参数区域的退化顺序，并揭示了随机优化器对临界点的归纳偏好。 |
| [^13] | [SDC-HSDD-NDSA: Structure Detecting Cluster by Hierarchical Secondary Directed Differential with Normalized Density and Self-Adaption.](http://arxiv.org/abs/2307.00677) | 本文提出了一种基于密度的聚类算法，能够检测到高密度区域中的结构，具有先前算法所不具备的能力。 |

# 详细

[^1]: 正态形式博弈中的均衡路径

    Paths to Equilibrium in Normal-Form Games

    [https://arxiv.org/abs/2403.18079](https://arxiv.org/abs/2403.18079)

    本文研究在多智体强化学习中的策略序列，探讨满足特定约束的策略路径，即满足路径，对于构建终止于均衡策略的路径具有重要意义。

    

    在多智体强化学习（MARL）中，智体会反复在时间上交互，并随着新数据的到来修订他们的策略，从而产生一系列策略概况。本文研究满足一种由强化学习中政策更新启发的成对约束的策略序列，其中在第 $t$ 期最优应答的智体在下一期 $t+1$ 不会改变其策略。这种约束仅要求优化智体不更改策略，但并不以任何方式限制其他非最优化智体，因此允许探索。具有此属性的序列被称为满足路径，并在许多 MARL 算法中自然出现。关于战略动态的一个基本问题是：对于给定的博弈和初始策略概况，是否总可以构建一个终止于均衡策略的满足路径？这个问题的解决对应着一些重要含义。

    arXiv:2403.18079v1 Announce Type: cross  Abstract: In multi-agent reinforcement learning (MARL), agents repeatedly interact across time and revise their strategies as new data arrives, producing a sequence of strategy profiles. This paper studies sequences of strategies satisfying a pairwise constraint inspired by policy updating in reinforcement learning, where an agent who is best responding in period $t$ does not switch its strategy in the next period $t+1$. This constraint merely requires that optimizing agents do not switch strategies, but does not constrain the other non-optimizing agents in any way, and thus allows for exploration. Sequences with this property are called satisficing paths, and arise naturally in many MARL algorithms. A fundamental question about strategic dynamics is such: for a given game and initial strategy profile, is it always possible to construct a satisficing path that terminates at an equilibrium strategy? The resolution of this question has implication
    
[^2]: BlendScape: 通过生成式人工智能实现统一和个性化视频会议环境

    BlendScape: Enabling Unified and Personalized Video-Conferencing Environments through Generative AI

    [https://arxiv.org/abs/2403.13947](https://arxiv.org/abs/2403.13947)

    BlendScape通过生成式人工智能技术实现了用户定制视频会议环境，支持灵活的任务空间表示和多模交互技术，使用户能够快速表达设计意图并设想未来会议中的协作结构。

    

    当今的视频会议工具支持丰富的专业和社交活动，但它们的通用、基于网格的环境无法轻松地适应分布式合作者的各种需求。为了实现最终用户定制，我们开发了BlendScape，这是一个系统，通过利用AI图像生成技术，允许会议参与者构建根据他们的协作背景定制的视频会议环境。BlendScape通过将用户的物理或虚拟背景融合到统一环境中支持任务空间的灵活表示，并实现了多模交互技术以引导生成过程。通过与15名最终用户的评估，我们调查了他们对工作和社交场景的定制偏好。参与者可以快速表达他们对BlendScape的设计意图，并设想使用该系统来组织未来会议中的协作，但也遇到了挑战。

    arXiv:2403.13947v1 Announce Type: cross  Abstract: Today's video-conferencing tools support a rich range of professional and social activities, but their generic, grid-based environments cannot be easily adapted to meet the varying needs of distributed collaborators. To enable end-user customization, we developed BlendScape, a system for meeting participants to compose video-conferencing environments tailored to their collaboration context by leveraging AI image generation techniques. BlendScape supports flexible representations of task spaces by blending users' physical or virtual backgrounds into unified environments and implements multimodal interaction techniques to steer the generation. Through an evaluation with 15 end-users, we investigated their customization preferences for work and social scenarios. Participants could rapidly express their design intentions with BlendScape and envisioned using the system to structure collaboration in future meetings, but experienced challenge
    
[^3]: LLM的决策水平在多智能体环境中的评估究竟如何？

    How Far Are We on the Decision-Making of LLMs? Evaluating LLMs' Gaming Ability in Multi-Agent Environments

    [https://arxiv.org/abs/2403.11807](https://arxiv.org/abs/2403.11807)

    通过博弈论视角评估LLMs的决策能力，结果表明GPT-3.5在稳健性方面表现良好，但泛化能力有限，而GPT-4则优于其他模型。

    

    决策是一个复杂的任务，需要各种能力，为评估大型语言模型（LLMs）提供了一个极好的框架。我们的研究通过博弈论的视角探究LLMs的决策能力。我们专注于支持多个智能体同时参与的游戏，引入了我们的框架GAMA-Bench，包括八个经典的多智能体游戏。我们设计了一个评分方案，定量评估模型在这些游戏中的表现。通过GAMA-Bench，我们研究了LLMs的稳健性、泛化能力和增强策略。结果显示，虽然GPT-3.5表现出令人满意的稳健性，但其泛化能力相对有限。然而，通过一些方法如“思维链”，其性能可以得到提高。此外，我们对各种LLMs进行评估，发现GPT-4胜过其他模型。

    arXiv:2403.11807v1 Announce Type: new  Abstract: Decision-making, a complicated task requiring various types of abilities, presents an excellent framework for assessing Large Language Models (LLMs). Our research investigates LLMs' decision-making capabilities through the lens of a well-established field, Game Theory. We focus specifically on games that support the participation of more than two agents simultaneously. Subsequently, we introduce our framework, GAMA-Bench, including eight classical multi-agent games. We design a scoring scheme to assess a model's performance in these games quantitatively. Through GAMA-Bench, we investigate LLMs' robustness, generalizability, and enhancement strategies. Results reveal that while GPT-3.5 shows satisfying robustness, its generalizability is relatively limited. However, its performance can be improved through approaches such as Chain-of-Thought. Additionally, we conduct evaluations across various LLMs and find that GPT-4 outperforms other mod
    
[^4]: GEAR: 一种用于几乎无损生成推断大型语言模型的高效KV缓存压缩方案

    GEAR: An Efficient KV Cache Compression Recipefor Near-Lossless Generative Inference of LLM

    [https://arxiv.org/abs/2403.05527](https://arxiv.org/abs/2403.05527)

    GEAR提出了一种高效的KV缓存压缩框架，实现几乎无损的高比率压缩，用于解决大型语言模型推断中因缓存需求增长而导致的记忆绑定问题和性能下降。

    

    关键-值（KV）缓存已成为加快大型语言模型（LLMs）推断生成速度的事实标准。然而，随着序列长度增加而增长的缓存需求已将LLM推断转变为一个记忆绑定问题，显著地限制了系统吞吐量。现有方法依赖于丢弃不重要的标记或均匀量化所有条目。然而，这种方法往往会产生较高的近似误差来表示压缩后的矩阵。自回归解码过程进一步增加了每个步骤的误差，导致模型生成中的重大偏差和性能恶化。为了解决这一挑战，我们提出了GEAR，一种高效的KV缓存压缩框架，实现几乎无损的高压缩比。

    arXiv:2403.05527v1 Announce Type: cross  Abstract: Key-value (KV) caching has become the de-facto to accelerate generation speed for large language models (LLMs) inference. However, the growing cache demand with increasing sequence length has transformed LLM inference to be a memory bound problem, significantly constraining the system throughput. Existing methods rely on dropping unimportant tokens or quantizing all entries uniformly. Such methods, however, often incur high approximation errors to represent the compressed matrices. The autoregressive decoding process further compounds the error of each step, resulting in critical deviation in model generation and deterioration of performance. To tackle this challenge, we propose GEAR, an efficient KV cache compression framework that achieves near-lossless high-ratio compression. GEAR first applies quantization to majority of entries of similar magnitudes to ultra-low precision. It then employs a low rank matrix to approximate the quant
    
[^5]: 我们距离智能视觉演绎推理还有多远？

    How Far Are We from Intelligent Visual Deductive Reasoning?

    [https://arxiv.org/abs/2403.04732](https://arxiv.org/abs/2403.04732)

    目前的视觉语言模型在文本推理方面表现出色，但在视觉演绎推理方面仍存在较大差距和盲点。

    

    最近，诸如GPT-4V之类的视觉语言模型（VLM）在各种视觉语言任务上取得了巨大进展。我们深入探讨了基于视觉的演绎推理，这是一个更复杂但不太被探索的领域，并发现了当前领先的VLM中以前未暴露的盲点。具体来说，我们利用瑞文渐进矩阵（RPM）来评估VLM在仅依靠视觉线索进行多跳关系和演绎推理的能力。我们对几种流行的VLM进行了全面评估，采用了标准策略，如上下文学习、自我一致性和思维链（CoT），在三个不同的数据集上进行了评估，包括Mensa智商测试、智商测试和RAVEN。结果表明，尽管LLM在基于文本的推理方面具有令人印象深刻的能力，但我们在视觉演绎推理方面仍有很大的差距。

    arXiv:2403.04732v1 Announce Type: new  Abstract: Vision-Language Models (VLMs) such as GPT-4V have recently demonstrated incredible strides on diverse vision language tasks. We dig into vision-based deductive reasoning, a more sophisticated but less explored realm, and find previously unexposed blindspots in the current SOTA VLMs. Specifically, we leverage Raven's Progressive Matrices (RPMs), to assess VLMs' abilities to perform multi-hop relational and deductive reasoning relying solely on visual clues. We perform comprehensive evaluations of several popular VLMs employing standard strategies such as in-context learning, self-consistency, and Chain-of-thoughts (CoT) on three diverse datasets, including the Mensa IQ test, IntelligenceTest, and RAVEN. The results reveal that despite the impressive capabilities of LLMs in text-based reasoning, we are still far from achieving comparable proficiency in visual deductive reasoning. We found that certain standard strategies that are effective
    
[^6]: 基于信息的转导式主动学习

    Information-based Transductive Active Learning

    [https://arxiv.org/abs/2402.15898](https://arxiv.org/abs/2402.15898)

    ITL是一种基于信息的转导式学习方法，可以在现实世界设置中自适应采样，以最大化关于指定预测目标的信息获取，并在少样本微调和安全贝叶斯优化应用中显著优于最先进技术。

    

    我们将主动学习推广到解决现实世界中采样受限于可访问域的情况，而预测目标可能位于这个域之外。为此，我们提出了ITL，即基于信息的转导式学习，一种自适应采样的方法，旨在最大化关于指定预测目标的信息获取。在一般正则性假设下，我们展示了ITL收敛到可从可访问数据中获得的最小可能不确定性。我们在两个关键应用中展示了ITL：大型神经网络的少样本微调和安全贝叶斯优化，在两种情况下，ITL明显优于最先进技术。

    arXiv:2402.15898v1 Announce Type: cross  Abstract: We generalize active learning to address real-world settings where sampling is restricted to an accessible region of the domain, while prediction targets may lie outside this region. To this end, we propose ITL, short for information-based transductive learning, an approach which samples adaptively to maximize the information gained about specified prediction targets. We show, under general regularity assumptions, that ITL converges uniformly to the smallest possible uncertainty obtainable from the accessible data. We demonstrate ITL in two key applications: Few-shot fine-tuning of large neural networks and safe Bayesian optimization, and in both cases, ITL significantly outperforms the state-of-the-art.
    
[^7]: 千面灌木：利用深度学习从卫星图像中进行个体分割

    Shrub of a thousand faces: an individual segmentation from satellite images using deep learning

    [https://arxiv.org/abs/2401.17985](https://arxiv.org/abs/2401.17985)

    该研究提出了一种新方法，利用深度学习模型从卫星图像中个别描绘位于西班牙内华达山脉林线上的杜松灌木。研究通过整合遥感影像和实例分割模型，解决了检测和描绘自然对象复杂生长模式的挑战。

    

    监测长寿灌木（如常见的杜松）的分布和大小结构可以用于估计气候变化对高山和高纬度生态系统的长期影响。历史航空超高分辨率影像提供了一种回顾性工具，可以高精度监测灌木的生长和分布。目前，深度学习模型在检测和描绘具有明确定义形状的对象的轮廓方面取得了令人印象深刻的结果。然而，将这些模型适应于检测表达复杂生长模式的自然对象（例如杜松）仍然是一项具有挑战性的任务。本研究提出了一种新方法，利用遥感RGB影像与基于Mask R-CNN的实例分割模型相结合，个别描绘西班牙内华达山脉林线上的杜松灌木。在这项研究中，我们提出了一种新的数据构造设计，其中使用的是光解译数据（PI）和实地调查数据（FW）分别进行操作。

    Monitoring the distribution and size structure of long-living shrubs, such as Juniperus communis, can be used to estimate the long-term effects of climate change on high-mountain and high latitude ecosystems. Historical aerial very-high resolution imagery offers a retrospective tool to monitor shrub growth and distribution at high precision. Currently, deep learning models provide impressive results for detecting and delineating the contour of objects with defined shapes. However, adapting these models to detect natural objects that express complex growth patterns, such as junipers, is still a challenging task.   This research presents a novel approach that leverages remotely sensed RGB imagery in conjunction with Mask R-CNN-based instance segmentation models to individually delineate Juniperus shrubs above the treeline in Sierra Nevada (Spain). In this study, we propose a new data construction design that consists in using photo interpreted (PI) and field work (FW) data to respectivel
    
[^8]: 通过多样化合成和扩散模型减轻偏见

    Mitigating Biases with Diverse Ensembles and Diffusion Models

    [https://arxiv.org/abs/2311.16176](https://arxiv.org/abs/2311.16176)

    通过利用扩散概率模型（DPMs）生成新特征组合的图像，可以在集成模型中增加模型多样性，并减轻捷径偏见，而无需额外监督信号。

    

    数据中的虚假相关性，即多个线索可以预测目标标签，常常导致一种称为捷径偏见的现象，即模型依赖于错误的、易学的线索，而忽略可靠的线索。在这项工作中，我们提出了一种利用扩散概率模型（DPMs）的集成多样化框架，用于减轻捷径偏见。我们展示了在特定的训练间隔中，DPMs可以生成具有新特征组合的图像，即使在显示相关输入特征的样本上进行训练。我们利用这一关键属性通过集成不一致性生成合成反事实来增加模型的多样性。我们展示了DPM引导的多样化足以消除对主要捷径线索的依赖，无需额外的监督信号。我们进一步在几个多样化目标上在实证上量化其有效性，并最终展示了改进的泛化性能。

    arXiv:2311.16176v2 Announce Type: replace-cross  Abstract: Spurious correlations in the data, where multiple cues are predictive of the target labels, often lead to a phenomenon known as shortcut bias, where a model relies on erroneous, easy-to-learn cues while ignoring reliable ones. In this work, we propose an ensemble diversification framework exploiting Diffusion Probabilistic Models (DPMs) for shortcut bias mitigation. We show that at particular training intervals, DPMs can generate images with novel feature combinations, even when trained on samples displaying correlated input features. We leverage this crucial property to generate synthetic counterfactuals to increase model diversity via ensemble disagreement. We show that DPM-guided diversification is sufficient to remove dependence on primary shortcut cues, without a need for additional supervised signals. We further empirically quantify its efficacy on several diversification objectives, and finally show improved generalizati
    
[^9]: 在开放存储库中分析AI视觉模型的质量属性及其在对抗攻击下的表现

    Analyzing the Quality Attributes of AI Vision Models in Open Repositories Under Adversarial Attacks. (arXiv:2401.12261v1 [cs.CR])

    [http://arxiv.org/abs/2401.12261](http://arxiv.org/abs/2401.12261)

    本文提出了一种综合流程，用于对AI视觉模型在开放存储库中的质量属性进行分析，尤其是在面对对抗攻击时的表现。我们展示了一个涉及六个计算机视觉模型的评估场景，以评估准确性、鲁棒性、解释效用和开销。

    

    随着AI模型的快速发展，它们经常发布到开放存储库中，如HuggingFace。在将它们集成到生产开发生命周期之前，对这些模型进行质量保证验证是至关重要的。除了评估平衡准确性和计算成本方面的效率外，对抗攻击可能对AI模型的鲁棒性和可解释性构成威胁。同时，可解释性AI（XAI）应用近似输入到输出的算法来识别贡献特征。对抗扰动可能会降低需要进一步研究的XAI解释的效用。在本文中，我们提出了一种综合流程，用于下游评估任务，包括验证AI模型的准确性，使用基准扰动评估鲁棒性，比较解释效用以及评估开销。我们展示了一个涉及六个计算机视觉模型的评估场景，其中包括基于CNN和Transformer的模型。

    As AI models rapidly evolve, they are frequently released to open repositories, such as HuggingFace. It is essential to perform quality assurance validation on these models before integrating them into the production development lifecycle. In addition to evaluating efficiency in terms of balanced accuracy and computing costs, adversarial attacks are potential threats to the robustness and explainability of AI models. Meanwhile, XAI applies algorithms that approximate inputs to outputs post-hoc to identify the contributing features. Adversarial perturbations may also degrade the utility of XAI explanations that require further investigation. In this paper, we present an integrated process designed for downstream evaluation tasks, including validating AI model accuracy, evaluating robustness with benchmark perturbations, comparing explanation utility, and assessing overhead. We demonstrate an evaluation scenario involving six computer vision models, which include CNN-based, Transformer-b
    
[^10]: 大型语言模型中的通用漏洞：上下文学习后门攻击

    Universal Vulnerabilities in Large Language Models: In-context Learning Backdoor Attacks. (arXiv:2401.05949v1 [cs.CL])

    [http://arxiv.org/abs/2401.05949](http://arxiv.org/abs/2401.05949)

    本研究发现上下文学习范式在大型语言模型中存在漏洞，攻击者可以通过污染示范上下文来操控模型行为，而无需进行微调。这项研究设计了一种名为ICLAttack的后门攻击方法，可以通过污染示范样本和提示来使模型按照预定义的意图行事。

    

    上下文学习是一种在预训练和微调之间弥合差距的范式，在几个自然语言处理任务中展现了高效性，特别是在少样本设置中。与传统的微调方法不同，上下文学习能够适应未见过的任务而无需更新任何参数。尽管被广泛应用，上下文学习仍然容易受到恶意攻击。本研究提出了对这一范式的安全性问题的关切。我们的研究表明，攻击者可以通过污染示范上下文来操控大型语言模型的行为，而无需对模型进行微调。具体来说，我们设计了一种新的后门攻击方法，命名为ICLAttack，针对基于上下文学习的大型语言模型。我们的方法包括两种类型的攻击：污染示范样本和污染提示，可以使模型按照预定义的意图行事。ICLAttack不需要额外的微调。

    In-context learning, a paradigm bridging the gap between pre-training and fine-tuning, has demonstrated high efficacy in several NLP tasks, especially in few-shot settings. Unlike traditional fine-tuning methods, in-context learning adapts pre-trained models to unseen tasks without updating any parameters. Despite being widely applied, in-context learning is vulnerable to malicious attacks. In this work, we raise security concerns regarding this paradigm. Our studies demonstrate that an attacker can manipulate the behavior of large language models by poisoning the demonstration context, without the need for fine-tuning the model. Specifically, we have designed a new backdoor attack method, named ICLAttack, to target large language models based on in-context learning. Our method encompasses two types of attacks: poisoning demonstration examples and poisoning prompts, which can make models behave in accordance with predefined intentions. ICLAttack does not require additional fine-tuning 
    
[^11]: 用符号回归梯度解释神经网络分类器的闭式表示

    Closed-Form Interpretation of Neural Network Classifiers with Symbolic Regression Gradients. (arXiv:2401.04978v1 [cs.LG])

    [http://arxiv.org/abs/2401.04978](http://arxiv.org/abs/2401.04978)

    本文提出了一种解释神经网络分类器的闭式表示的方法，使其适用于自动化科学发现。这种方法通过将神经网络嵌入到一组基于相同量的等价类中，并通过找到该等价类与符号回归搜索空间中的方程的交集来解释神经网络。

    

    我提出了一个统一的框架来解释神经网络分类器，以实现自动科学发现。与基于神经网络的回归不同，对于分类而言，即使神经网络本身的分类基于可以表示为闭式方程的量，也一般无法找到从神经网络到符号方程的一对一映射。在本文中，我将训练好的神经网络嵌入到一个等价类中，这个等价类的分类函数的决策都基于相同的量。我通过找到这个等价类与由符号回归搜索空间定义的可读的方程的交集来解释神经网络。这种方法不限于分类器或完整的神经网络，还可以应用于隐藏层或潜在空间中的任意神经元，或简化解释神经网络回归器的过程。

    I introduce a unified framework for interpreting neural network classifiers tailored toward automated scientific discovery. In contrast to neural network-based regression, for classification, it is in general impossible to find a one-to-one mapping from the neural network to a symbolic equation even if the neural network itself bases its classification on a quantity that can be written as a closed-form equation. In this paper, I embed a trained neural network into an equivalence class of classifying functions that base their decisions on the same quantity. I interpret neural networks by finding an intersection between this equivalence class and human-readable equations defined by the search space of symbolic regression. The approach is not limited to classifiers or full neural networks and can be applied to arbitrary neurons in hidden layers or latent spaces or to simplify the process of interpreting neural network regressors.
    
[^12]: 通过学习系数量化奇异模型中的退化情况

    Quantifying degeneracy in singular models via the learning coefficient. (arXiv:2308.12108v1 [stat.ML])

    [http://arxiv.org/abs/2308.12108](http://arxiv.org/abs/2308.12108)

    这项工作介绍了一种称为学习系数的量，用于精确量化深度神经网络中的退化程度，该方法能够区分不同参数区域的退化顺序，并揭示了随机优化器对临界点的归纳偏好。

    

    深度神经网络(DNN)是具有复杂退化的奇异统计模型。本文阐述了一种称为学习系数的量，它在奇异学习理论中精确地量化了深度神经网络的退化程度。重要的是，我们将证明DNN中的退化不能仅通过计算“平坦”方向的数量来解释。我们提出了一种基于随机梯度 Langevin 动力学的局部学习系数的计算可扩展近似方法。为了验证我们的方法，我们在已知理论值的低维模型上演示了其准确性。重要的是，局部学习系数能够正确恢复感兴趣参数区域之间退化的顺序。对MNIST的实验证明，局部学习系数可以揭示随机优化器对更退化或不太退化的临界点的归纳偏好。

    Deep neural networks (DNN) are singular statistical models which exhibit complex degeneracies. In this work, we illustrate how a quantity known as the \emph{learning coefficient} introduced in singular learning theory quantifies precisely the degree of degeneracy in deep neural networks. Importantly, we will demonstrate that degeneracy in DNN cannot be accounted for by simply counting the number of "flat" directions. We propose a computationally scalable approximation of a localized version of the learning coefficient using stochastic gradient Langevin dynamics. To validate our approach, we demonstrate its accuracy in low-dimensional models with known theoretical values. Importantly, the local learning coefficient can correctly recover the ordering of degeneracy between various parameter regions of interest. An experiment on MNIST shows the local learning coefficient can reveal the inductive bias of stochastic opitmizers for more or less degenerate critical points.
    
[^13]: SDC-HSDD-NDSA: 使用层次次级导向差异和归一化密度自适应的结构检测聚类算法

    SDC-HSDD-NDSA: Structure Detecting Cluster by Hierarchical Secondary Directed Differential with Normalized Density and Self-Adaption. (arXiv:2307.00677v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2307.00677](http://arxiv.org/abs/2307.00677)

    本文提出了一种基于密度的聚类算法，能够检测到高密度区域中的结构，具有先前算法所不具备的能力。

    

    基于密度的聚类算法是最受欢迎的聚类算法之一，因为它能够识别任意形状的聚类，只要不同的高密度聚类之间有低密度区域分隔。然而，通过低密度区域将聚类分隔开的要求并不是微不足道的，因为高密度区域可能具有不同的结构，应该被聚类到不同的组中。这种情况说明了我们已知的所有先前基于密度的聚类算法的主要缺陷--无法检测高密度聚类中的结构。因此，本文旨在提供一种基于密度的聚类方案，既具有先前方法的能力，又能够检测到高密度区域中未被低密度区分开的结构。该算法采用层次次级导向差异、层次化、归一化密度以及自适应系数，因此被称为结构检测聚类算法。

    Density-based clustering could be the most popular clustering algorithm since it can identify clusters of arbitrary shape as long as different (high-density) clusters are separated by low-density regions. However, the requirement of the separateness of clusters by low-density regions is not trivial since a high-density region might have different structures which should be clustered into different groups. Such a situation demonstrates the main flaw of all previous density-based clustering algorithms we have known--structures in a high-density cluster could not be detected. Therefore, this paper aims to provide a density-based clustering scheme that not only has the ability previous ones have but could also detect structures in a high-density region not separated by low-density ones. The algorithm employs secondary directed differential, hierarchy, normalized density, as well as the self-adaption coefficient, and thus is called Structure Detecting Cluster by Hierarchical Secondary Direc
    


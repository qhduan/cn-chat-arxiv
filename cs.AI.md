# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Knowing Your Nonlinearities: Shapley Interactions Reveal the Underlying Structure of Data](https://arxiv.org/abs/2403.13106) | 该论文使用Shapley Taylor互动指数（STII）分析了底层数据结构对各种模态、任务和架构中模型表征的影响，发现了语言模型和语音模型中的新颖现象，并展示了特征交互如何直观反映对象边界。 |
| [^2] | [DREsS: Dataset for Rubric-based Essay Scoring on EFL Writing](https://arxiv.org/abs/2402.16733) | 本文发布了一个大型标准数据集DREsS，用于基于评分标准的自动作文评分，在提出了一种基于破坏的作文增强策略CASE后，这个数据集的基线结果提高了45.44％。 |
| [^3] | [A Comparative Study of Conventional and Tripolar EEG for High-Performance Reach-to-Grasp BCI Systems](https://arxiv.org/abs/2402.09448) | 比较传统EEG与三极EEG在高性能到颤抓握BCI系统中的有效性，包括信噪比、空间分辨率、ERPs和小波时频分析。 |
| [^4] | [Forecasting high-impact research topics via machine learning on evolving knowledge graphs](https://arxiv.org/abs/2402.08640) | 通过机器学习预测未发布研究想法的影响力，我们使用一个由超过2100万篇科学论文构建的演化知识图谱，结合论文内容和历史引用的信息，高准确度预测未来的演化网络动态和新的研究方向的影响力。 |
| [^5] | [Average-Case Analysis of Iterative Voting](https://arxiv.org/abs/2402.08144) | 这项工作通过分析代理人偏好分布的平均情况，扩展了迭代投票模型的效果分析。并且区分了迭代多数制何时改善或降低渐近福利。 |
| [^6] | [Learning Concepts Definable in First-Order Logic with Counting](https://arxiv.org/abs/1909.03820) | 该研究将一阶逻辑与计数符号相结合，证明了可以在多对数度结构下以次线性时间一致学习可定义的分类器，为包含数值方面的机器学习扩展学习框架迈出了第一步。 |
| [^7] | [Generating Likely Counterfactuals Using Sum-Product Networks.](http://arxiv.org/abs/2401.14086) | 由于用户需求和最近的法规要求，需要对AI系统所做出的决策进行解释。本论文提出了一种使用Sum-Product Networks模拟寻找高可能性反事实推理的系统，该系统能够提供满足多个常见要求的最佳解释。 |
| [^8] | [Pointwise-in-Time Explanation for Linear Temporal Logic Rules.](http://arxiv.org/abs/2306.13956) | 本文提出了一个可以评估给定路径规划中特定时间点上的单个线性时间逻辑(LTL)约束的相关性和状态的框架，可以用于在离散时间、离散空间中执行有限计划的代理任务中，为用户提供时间点解释和规则参数状态的洞察力。 |
| [^9] | [Effective Data Augmentation With Diffusion Models.](http://arxiv.org/abs/2302.07944) | 本文提出了一种利用预训练文本至图像扩散模型参数化的图像到图像转换方法，用于解决数据增强的多样性不足问题，并能够泛化到新视觉概念，从而提高了少样本图像分类和图像识别的性能。 |

# 详细

[^1]: 认识你的非线性：Shapley互动揭示数据的潜在结构

    Knowing Your Nonlinearities: Shapley Interactions Reveal the Underlying Structure of Data

    [https://arxiv.org/abs/2403.13106](https://arxiv.org/abs/2403.13106)

    该论文使用Shapley Taylor互动指数（STII）分析了底层数据结构对各种模态、任务和架构中模型表征的影响，发现了语言模型和语音模型中的新颖现象，并展示了特征交互如何直观反映对象边界。

    

    测量非线性特征交互是理解许多模型中复杂归因模式的一种已建立的方法。本文使用Shapley Taylor互动指数（STII）来分析底层数据结构对多种模态、任务和架构中模型表征的影响。在考虑掩码和自回归语言模型（MLMs和ALMs）中的语言结构时，我们发现STII在惯用表达中增加，MLMs随句法距离扩展STII，更多地依赖语法在其非线性结构中相比ALMs。我们的语音模型研究反映了口腔张开程度决定音素根据上下文变化的数量的原则。最后，我们研究图像分类器并说明特征交互直观反映对象边界。我们广泛的结果展示了跨学科工作和领域之间的益处。

    arXiv:2403.13106v1 Announce Type: cross  Abstract: Measuring nonlinear feature interaction is an established approach to understanding complex patterns of attribution in many models. In this paper, we use Shapley Taylor interaction indices (STII) to analyze the impact of underlying data structure on model representations in a variety of modalities, tasks, and architectures. Considering linguistic structure in masked and auto-regressive language models (MLMs and ALMs), we find that STII increases within idiomatic expressions and that MLMs scale STII with syntactic distance, relying more on syntax in their nonlinear structure than ALMs do. Our speech model findings reflect the phonetic principal that the openness of the oral cavity determines how much a phoneme varies based on its context. Finally, we study image classifiers and illustrate that feature interactions intuitively reflect object boundaries. Our wide range of results illustrates the benefits of interdisciplinary work and doma
    
[^2]: DREsS: 英语作为外语写作基于评分标准的数据集

    DREsS: Dataset for Rubric-based Essay Scoring on EFL Writing

    [https://arxiv.org/abs/2402.16733](https://arxiv.org/abs/2402.16733)

    本文发布了一个大型标准数据集DREsS，用于基于评分标准的自动作文评分，在提出了一种基于破坏的作文增强策略CASE后，这个数据集的基线结果提高了45.44％。

    

    自动化作文评分（AES）是英语作为外语写作教育中一种有用的工具，为学生和教师提供实时作文评分。然而，先前的AES模型是在与EFL写作教育实际场景不相关的作文和分数上进行训练的，并且通常由于缺乏适当的数据集而提供单一的整体评分。在本文中，我们发布了DREsS，这是一个用于基于评分标准的自动作文评分的大型标准数据集。DREsS包括三个子数据集：DREsS_New，DREsS_Std.和DREsS_CASE。我们收集了DREsS_New，这是一个由EFL本科生撰写并由英语教育专家评分的真实课堂数据集。我们还将现有的基于评分标准的作文评分数据集标准化为DREsS_Std。我们提出了一个名为CASE的基于破坏的作文增强策略，用于生成20K个DREsS_CASE的合成样本，并将基线结果提高了45.44％。

    arXiv:2402.16733v1 Announce Type: new  Abstract: Automated essay scoring (AES) is a useful tool in English as a Foreign Language (EFL) writing education, offering real-time essay scores for students and instructors. However, previous AES models were trained on essays and scores irrelevant to the practical scenarios of EFL writing education and usually provided a single holistic score due to the lack of appropriate datasets. In this paper, we release DREsS, a large-scale, standard dataset for rubric-based automated essay scoring. DREsS comprises three sub-datasets: DREsS_New, DREsS_Std., and DREsS_CASE. We collect DREsS_New, a real-classroom dataset with 1.7K essays authored by EFL undergraduate students and scored by English education experts. We also standardize existing rubric-based essay scoring datasets as DREsS_Std. We suggest CASE, a corruption-based augmentation strategy for essays, which generates 20K synthetic samples of DREsS_CASE and improves the baseline results by 45.44%. 
    
[^3]: 普通EEG与三极EEG在高性能到颤抓握BCI系统中的比较研究

    A Comparative Study of Conventional and Tripolar EEG for High-Performance Reach-to-Grasp BCI Systems

    [https://arxiv.org/abs/2402.09448](https://arxiv.org/abs/2402.09448)

    比较传统EEG与三极EEG在高性能到颤抓握BCI系统中的有效性，包括信噪比、空间分辨率、ERPs和小波时频分析。

    

    本研究旨在比较传统EEG与三极EEG在提升运动障碍个体的BCI应用方面的有效性。重点是解读和解码各种抓握动作，如力握和精确握持。目标是确定哪种EEG技术在处理和翻译与抓握相关的脑电信号方面更为有效。研究涉及对十名健康参与者进行实验，参与者进行了两种不同的握持运动：力握和精确握持，无运动条件作为基线。我们的研究在解码抓握动作方面对EEG和三极EEG进行了全面比较。该比较涵盖了几个关键参数，包括信噪比（SNR）、通过功能连接的空间分辨率、ERPs和小波时频分析。此外，我们的研究还涉及从...

    arXiv:2402.09448v1 Announce Type: cross  Abstract: This study aims to enhance BCI applications for individuals with motor impairments by comparing the effectiveness of tripolar EEG (tEEG) with conventional EEG. The focus is on interpreting and decoding various grasping movements, such as power grasp and precision grasp. The goal is to determine which EEG technology is more effective in processing and translating grasp related neural signals. The approach involved experimenting on ten healthy participants who performed two distinct grasp movements: power grasp and precision grasp, with a no movement condition serving as the baseline. Our research presents a thorough comparison between EEG and tEEG in decoding grasping movements. This comparison spans several key parameters, including signal to noise ratio (SNR), spatial resolution via functional connectivity, ERPs, and wavelet time frequency analysis. Additionally, our study involved extracting and analyzing statistical features from th
    
[^4]: 通过机器学习在不断演化的知识图谱上预测高影响力的研究主题

    Forecasting high-impact research topics via machine learning on evolving knowledge graphs

    [https://arxiv.org/abs/2402.08640](https://arxiv.org/abs/2402.08640)

    通过机器学习预测未发布研究想法的影响力，我们使用一个由超过2100万篇科学论文构建的演化知识图谱，结合论文内容和历史引用的信息，高准确度预测未来的演化网络动态和新的研究方向的影响力。

    

    科学出版物的指数增长对人类研究者构成了严峻挑战。它迫使研究者将注意力集中在更狭窄的子领域上，使得发现其他领域的新颖且有影响力的研究想法和合作变得困难。虽然有办法预测科学论文未来的引用次数，但通常需要等到研究完成并且论文写成后才能进行评估，这样就错过了想法构思的早期阶段。在本文中，我们展示了如何预测从未被研究者发布的想法的影响力。为此，我们开发了一个大型的演化知识图谱，其中包含超过2100万篇科学论文。它结合了从论文内容中创建的语义网络和从历史引用中创建的影响网络。利用机器学习，我们可以高准确度地预测演化网络的动态情况，从而预测新的研究方向的影响力。我们预期这种能力将有助于研究者发现具有高影响力的研究主题。

    The exponential growth in scientific publications poses a severe challenge for human researchers. It forces attention to more narrow sub-fields, which makes it challenging to discover new impactful research ideas and collaborations outside one's own field. While there are ways to predict a scientific paper's future citation counts, they need the research to be finished and the paper written, usually assessing impact long after the idea was conceived. Here we show how to predict the impact of onsets of ideas that have never been published by researchers. For that, we developed a large evolving knowledge graph built from more than 21 million scientific papers. It combines a semantic network created from the content of the papers and an impact network created from the historic citations of papers. Using machine learning, we can predict the dynamic of the evolving network into the future with high accuracy, and thereby the impact of new research directions. We envision that the ability to 
    
[^5]: 迭代投票的平均情况分析

    Average-Case Analysis of Iterative Voting

    [https://arxiv.org/abs/2402.08144](https://arxiv.org/abs/2402.08144)

    这项工作通过分析代理人偏好分布的平均情况，扩展了迭代投票模型的效果分析。并且区分了迭代多数制何时改善或降低渐近福利。

    

    迭代投票是社会选择中重复战略决策的自然模型，当代理可以在最终确定群体决策之前更新他们的投票时。之前的研究通过对无序文化下代理人偏好的最坏情况和平均情况表现进行分析，通过改进安纳基价格来分析迭代多数制对平衡点选出的结果福利的有效性。然而，之前的分析只研究了在代理人偏好通过无偏文化分布的最坏情况和平均情况下的性能。本研究将平均情况分析扩展到更广泛的分布类，并区分出迭代多数制何时改善或降低渐近福利。

    Iterative voting is a natural model of repeated strategic decision-making in social choice when agents have the opportunity to update their votes prior to finalizing the group decision. Prior work has analyzed the efficacy of iterative plurality on the welfare of the chosen outcome at equilibrium, relative to the truthful vote profile, via an adaptation of the price of anarchy. However, prior analyses have only studied the worst-case and average-case performances when agents' preferences are distributed by the impartial culture. This work extends average-case analyses to a wider class of distributions and distinguishes when iterative plurality improves or degrades asymptotic welfare.
    
[^6]: 用计数符号的一阶逻辑定义的概念的学习

    Learning Concepts Definable in First-Order Logic with Counting

    [https://arxiv.org/abs/1909.03820](https://arxiv.org/abs/1909.03820)

    该研究将一阶逻辑与计数符号相结合，证明了可以在多对数度结构下以次线性时间一致学习可定义的分类器，为包含数值方面的机器学习扩展学习框架迈出了第一步。

    

    我们研究了在Grohe和Tur\'an引入的逻辑框架下的关系背景结构上的布尔分类问题。众所周知(Grohe和Ritzert, LICS 2017)，在多对数度结构上的一阶逻辑可定义的分类器可以在次线性时间内学习，其中结构的度和运行时间是以结构的大小为单位来衡量的。我们将结果推广到了由Kuske和Schweikardt(LICS 2017)引入的带计数的一阶逻辑FOCN，它作为一个广泛推广各种计数逻辑的表现逻辑。具体来说，我们证明了可以在多对数度结构类上定义的FOCN中的分类器可以在次线性时间内一致地学习。这可以看作是将学习框架扩展以包含机器学习的数值方面的第一步。我们将这一结果扩展到了无视的概率

    arXiv:1909.03820v2 Announce Type: replace-cross  Abstract: We study Boolean classification problems over relational background structures in the logical framework introduced by Grohe and Tur\'an (TOCS 2004). It is known (Grohe and Ritzert, LICS 2017) that classifiers definable in first-order logic over structures of polylogarithmic degree can be learned in sublinear time, where the degree of the structure and the running time are measured in terms of the size of the structure. We generalise the results to the first-order logic with counting FOCN, which was introduced by Kuske and Schweikardt (LICS 2017) as an expressive logic generalising various other counting logics. Specifically, we prove that classifiers definable in FOCN over classes of structures of polylogarithmic degree can be consistently learned in sublinear time. This can be seen as a first step towards extending the learning framework to include numerical aspects of machine learning. We extend the result to agnostic probabl
    
[^7]: 使用Sum-Product Networks生成可能的反事实推理

    Generating Likely Counterfactuals Using Sum-Product Networks. (arXiv:2401.14086v1 [cs.AI])

    [http://arxiv.org/abs/2401.14086](http://arxiv.org/abs/2401.14086)

    由于用户需求和最近的法规要求，需要对AI系统所做出的决策进行解释。本论文提出了一种使用Sum-Product Networks模拟寻找高可能性反事实推理的系统，该系统能够提供满足多个常见要求的最佳解释。

    

    由于用户需求和最近的法规（GDPR、AI法案），需要解释AI系统所做出的决策。这些决策往往只能在事后解释，反事实推理成为常见的解释方式。什么构成了最佳的反事实解释必须考虑多个方面，其中“样本距离”是最常见的。我们认为，这一要求经常会导致不太可能且因此价值有限的解释。在这里，我们提出了一个能够提供高可能性解释的系统。我们展示了使用混合整数优化（MIO）模拟寻找满足反事实推理的许多常见要求的最有可能解释。在此过程中，我们提出了Sum-Product Network（SPN）的MIO表达，并使用SPN估计反事实的可能性，这对独立的兴趣也有用。与生成反事实解释的几种方法进行数值比较。

    Due to user demand and recent regulation (GDPR, AI Act), decisions made by AI systems need to be explained. These decisions are often explainable only post hoc, where counterfactual explanations are popular. The question of what constitutes the best counterfactual explanation must consider multiple aspects, where "distance from the sample" is the most common. We argue that this requirement frequently leads to explanations that are unlikely and, therefore, of limited value. Here, we present a system that provides high-likelihood explanations. We show that the search for the most likely explanations satisfying many common desiderata for counterfactual explanations can be modeled using mixed-integer optimization (MIO). In the process, we propose an MIO formulation of a Sum-Product Network (SPN) and use the SPN to estimate the likelihood of a counterfactual, which can be of independent interest. A numerical comparison against several methods for generating counterfactual explanations is pr
    
[^8]: 线性时间逻辑规则的时间点解释框架

    Pointwise-in-Time Explanation for Linear Temporal Logic Rules. (arXiv:2306.13956v1 [cs.AI])

    [http://arxiv.org/abs/2306.13956](http://arxiv.org/abs/2306.13956)

    本文提出了一个可以评估给定路径规划中特定时间点上的单个线性时间逻辑(LTL)约束的相关性和状态的框架，可以用于在离散时间、离散空间中执行有限计划的代理任务中，为用户提供时间点解释和规则参数状态的洞察力。

    

    本文介绍了一个框架来评估给定路径规划中特定时间点上的单个线性时间逻辑(LTL)约束的相关性，这个任务被我们称为“时间点解释”。我们开发了一个包含状态评估算法的框架，适用于在Kripke结构可表达的离散时间、离散空间中执行有限计划的代理。在给定的结构上和已知约束代理的一组LTL规则的计划中，该算法针对两种类型的用户查询响应地生成解释。对于所选的查询时间，解释识别哪些规则是活动的，哪些规则刚刚被满足，哪些规则是不活动的，其中框架状态标准是正式和直观地定义的。解释还可以包括单个规则参数的状态，以提供进一步的洞察力。在本文中，我们系统地介绍了这个新颖的框架，并提供了其实现的示例。

    This work introduces a framework to assess the relevance of individual linear temporal logic (LTL) constraints at specific times in a given path plan, a task we refer to as "pointwise-in-time" explanation. We develop this framework, featuring a status assessment algorithm, for agents which execute finite plans in a discrete-time, discrete-space setting expressible via a Kripke structure. Given a plan on this structure and a set of LTL rules which are known to constrain the agent, the algorithm responds to two types of user queries to produce explanation. For the selected query time, explanations identify which rules are active, which have just been satisfied, and which are inactive, where the framework status criteria are formally and intuitively defined. Explanations may also include the status of individual rule arguments to provide further insight. In this paper, we systematically present this novel framework and provide an example of its implementation.
    
[^9]: 利用扩散模型进行有效的数据增强

    Effective Data Augmentation With Diffusion Models. (arXiv:2302.07944v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.07944](http://arxiv.org/abs/2302.07944)

    本文提出了一种利用预训练文本至图像扩散模型参数化的图像到图像转换方法，用于解决数据增强的多样性不足问题，并能够泛化到新视觉概念，从而提高了少样本图像分类和图像识别的性能。

    

    数据增强是深度学习中最常见的工具之一，支撑着最近包括分类、生成模型和表示学习在内的许多进展。然而，当前的增强方法在数据的关键语义轴上缺乏多样性，缺乏改变高级语义属性（如场景中的动物种类）以增强数据多样性的方法。本文提出了一种利用预训练文本至图像扩散模型参数化的图像到图像转换来解决数据增强多样性不足问题的方法。我们的方法利用现成的扩散模型编辑图像，改变它们的语义，能够泛化到仅用少量标记示例得到的新视觉概念。我们在少样本图像分类任务和真实世界的杂草识别任务中评估了我们的方法，并观察到......

    Data augmentation is one of the most prevalent tools in deep learning, underpinning many recent advances, including those from classification, generative models, and representation learning. The standard approach to data augmentation combines simple transformations like rotations and flips to generate new images from existing ones. However, these new images lack diversity along key semantic axes present in the data. Current augmentations cannot alter the high-level semantic attributes, such as animal species present in a scene, to enhance the diversity of data. We address the lack of diversity in data augmentation with image-to-image transformations parameterized by pre-trained text-to-image diffusion models. Our method edits images to change their semantics using an off-the-shelf diffusion model, and generalizes to novel visual concepts from a few labelled examples. We evaluate our approach on few-shot image classification tasks, and on a real-world weed recognition task, and observe 
    


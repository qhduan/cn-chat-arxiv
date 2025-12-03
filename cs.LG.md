# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ContourDiff: Unpaired Image Translation with Contour-Guided Diffusion Models](https://arxiv.org/abs/2403.10786) | ContourDiff是一种新颖的框架，利用图像的领域不变解剖轮廓表示，旨在帮助准确翻译医学图像并保持其解剖准确性。 |
| [^2] | [PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs](https://arxiv.org/abs/2312.15230) | 本研究中，通过仅更新少部分高度表达力的参数，我们挑战了全参数重新训练的做法，在修剪后恢复或甚至提升了性能。PERP方法显著减少了计算量和存储需求。 |
| [^3] | [Towards Inferring Users' Impressions of Robot Performance in Navigation Scenarios.](http://arxiv.org/abs/2310.11590) | 本研究拟通过非语言行为提示和机器学习技术预测人们对机器人行为印象，并提供了一个数据集和分析结果，发现在导航场景中，空间特征是最关键的信息。 |
| [^4] | [Combining Survival Analysis and Machine Learning for Mass Cancer Risk Prediction using EHR data.](http://arxiv.org/abs/2309.15039) | 该论文介绍了一种利用 EHR 数据进行大规模肿瘤风险预测的新方法，其创新之处在于只需利用历史的医疗服务代码和诊断信息来实现最小化的数据需求，通过将存活分析和机器学习相结合，可以在大规模应用中实现对患者癌症风险的个性化评估。 |
| [^5] | [Data Augmentation in the Underparameterized and Overparameterized Regimes.](http://arxiv.org/abs/2202.09134) | 这项研究提供了数据增强如何影响估计的方差和极限分布的确切量化结果，发现数据增强可能会增加估计的不确定性，并且其效果取决于多个因素。同时，该研究还通过随机转换的高维随机向量的函数的极限定理进行了证明。 |

# 详细

[^1]: ContourDiff：带轮廓引导扩散模型的无配对图像翻译

    ContourDiff: Unpaired Image Translation with Contour-Guided Diffusion Models

    [https://arxiv.org/abs/2403.10786](https://arxiv.org/abs/2403.10786)

    ContourDiff是一种新颖的框架，利用图像的领域不变解剖轮廓表示，旨在帮助准确翻译医学图像并保持其解剖准确性。

    

    准确地在不同模态之间翻译医学图像（例如从CT到MRI）对于许多临床和机器学习应用至关重要。本文提出了一种名为ContourDiff的新框架，该框架利用图像的领域不变解剖轮廓表示。这些表示易于从图像中提取，但对其解剖内容形成精确的空间约束。我们引入一种扩散模型，将来自任意输入领域的图像的轮廓表示转换为输出领域中的图像。

    arXiv:2403.10786v1 Announce Type: cross  Abstract: Accurately translating medical images across different modalities (e.g., CT to MRI) has numerous downstream clinical and machine learning applications. While several methods have been proposed to achieve this, they often prioritize perceptual quality with respect to output domain features over preserving anatomical fidelity. However, maintaining anatomy during translation is essential for many tasks, e.g., when leveraging masks from the input domain to develop a segmentation model with images translated to the output domain. To address these challenges, we propose ContourDiff, a novel framework that leverages domain-invariant anatomical contour representations of images. These representations are simple to extract from images, yet form precise spatial constraints on their anatomical content. We introduce a diffusion model that converts contour representations of images from arbitrary input domains into images in the output domain of in
    
[^2]: PERP: 在LLMs时代重新思考修剪-重新训练范式

    PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs

    [https://arxiv.org/abs/2312.15230](https://arxiv.org/abs/2312.15230)

    本研究中，通过仅更新少部分高度表达力的参数，我们挑战了全参数重新训练的做法，在修剪后恢复或甚至提升了性能。PERP方法显著减少了计算量和存储需求。

    

    神经网络可以通过修剪实现高效压缩，显著减少存储和计算需求同时保持预测性能。像迭代幅值修剪（IMP，Han等，2015）这样的简单而有效的方法可以去除不重要的参数，并需要昂贵的重新训练过程以在修剪后恢复性能。然而，随着大型语言模型（LLMs）的兴起，由于内存和计算限制，完全重新训练变得不可行。在本研究中，我们挑战了重新训练所有参数的做法，通过证明只更新少部分高度表达力的参数通常足以恢复甚至提高性能。令人惊讶的是，仅重新训练GPT-结构的0.27%-0.35%的参数即可在不同稀疏水平上实现与一次性IMP相当的性能。我们的方法，即修剪后参数高效重新训练（PERP），大大减少了计算量。

    Neural Networks can be efficiently compressed through pruning, significantly reducing storage and computational demands while maintaining predictive performance. Simple yet effective methods like Iterative Magnitude Pruning (IMP, Han et al., 2015) remove less important parameters and require a costly retraining procedure to recover performance after pruning. However, with the rise of Large Language Models (LLMs), full retraining has become infeasible due to memory and compute constraints. In this study, we challenge the practice of retraining all parameters by demonstrating that updating only a small subset of highly expressive parameters is often sufficient to recover or even improve performance compared to full retraining. Surprisingly, retraining as little as 0.27%-0.35% of the parameters of GPT-architectures achieves comparable performance to One Shot IMP across various sparsity levels. Our approach, Parameter-Efficient Retraining after Pruning (PERP), drastically reduces compute a
    
[^3]: 探索在导航场景下推断用户对机器人性能的印象

    Towards Inferring Users' Impressions of Robot Performance in Navigation Scenarios. (arXiv:2310.11590v1 [cs.RO])

    [http://arxiv.org/abs/2310.11590](http://arxiv.org/abs/2310.11590)

    本研究拟通过非语言行为提示和机器学习技术预测人们对机器人行为印象，并提供了一个数据集和分析结果，发现在导航场景中，空间特征是最关键的信息。

    

    人们对机器人性能的印象通常通过调查问卷来衡量。作为一种更可扩展且成本效益更高的替代方案，我们研究了使用非语言行为提示和机器学习技术预测人们对机器人行为印象的可能性。为此，我们首先提供了SEAN TOGETHER数据集，该数据集包括在虚拟现实模拟中人与移动机器人相互作用的观察结果，以及用户对机器人性能的5点量表评价。其次，我们对人类和监督学习技术如何基于不同的观察类型（例如面部、空间和地图特征）来预测感知到的机器人性能进行了分析。我们的结果表明，仅仅面部表情就能提供关于人们对机器人性能印象的有用信息；但在我们测试的导航场景中，空间特征是这种推断任务最关键的信息。

    Human impressions of robot performance are often measured through surveys. As a more scalable and cost-effective alternative, we study the possibility of predicting people's impressions of robot behavior using non-verbal behavioral cues and machine learning techniques. To this end, we first contribute the SEAN TOGETHER Dataset consisting of observations of an interaction between a person and a mobile robot in a Virtual Reality simulation, together with impressions of robot performance provided by users on a 5-point scale. Second, we contribute analyses of how well humans and supervised learning techniques can predict perceived robot performance based on different combinations of observation types (e.g., facial, spatial, and map features). Our results show that facial expressions alone provide useful information about human impressions of robot performance; but in the navigation scenarios we tested, spatial features are the most critical piece of information for this inference task. Als
    
[^4]: 结合存活分析和机器学习利用电子健康记录数据进行肿瘤风险预测

    Combining Survival Analysis and Machine Learning for Mass Cancer Risk Prediction using EHR data. (arXiv:2309.15039v1 [cs.LG])

    [http://arxiv.org/abs/2309.15039](http://arxiv.org/abs/2309.15039)

    该论文介绍了一种利用 EHR 数据进行大规模肿瘤风险预测的新方法，其创新之处在于只需利用历史的医疗服务代码和诊断信息来实现最小化的数据需求，通过将存活分析和机器学习相结合，可以在大规模应用中实现对患者癌症风险的个性化评估。

    

    纯粹的医学肿瘤筛查方法通常费用高昂、耗时长，并且仅适用于大规模应用。先进的人工智能（AI）方法在癌症检测方面发挥了巨大作用，但需要特定或深入的医学数据。这些方面影响了癌症筛查方法的大规模实施。因此，基于已有的电子健康记录（EHR）数据对患者进行大规模个性化癌症风险评估应用AI方法是一种颠覆性的改变。本文提出了一种利用EHR数据进行大规模肿瘤风险预测的新方法。与其他方法相比，我们的方法通过最小的数据贪婪策略脱颖而出，仅需要来自EHR的医疗服务代码和诊断历史。我们将问题形式化为二分类问题。该数据集包含了175441名不记名的患者（其中2861名被诊断为癌症）。作为基准，我们实现了一个基于循环神经网络（RNN）的解决方案。我们提出了一种方法，将存活分析和机器学习相结合，

    Purely medical cancer screening methods are often costly, time-consuming, and weakly applicable on a large scale. Advanced Artificial Intelligence (AI) methods greatly help cancer detection but require specific or deep medical data. These aspects affect the mass implementation of cancer screening methods. For these reasons, it is a disruptive change for healthcare to apply AI methods for mass personalized assessment of the cancer risk among patients based on the existing Electronic Health Records (EHR) volume.  This paper presents a novel method for mass cancer risk prediction using EHR data. Among other methods, our one stands out by the minimum data greedy policy, requiring only a history of medical service codes and diagnoses from EHR. We formulate the problem as a binary classification. This dataset contains 175 441 de-identified patients (2 861 diagnosed with cancer). As a baseline, we implement a solution based on a recurrent neural network (RNN). We propose a method that combine
    
[^5]: 在欠参数化和过参数化的模式中的数据增强

    Data Augmentation in the Underparameterized and Overparameterized Regimes. (arXiv:2202.09134v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2202.09134](http://arxiv.org/abs/2202.09134)

    这项研究提供了数据增强如何影响估计的方差和极限分布的确切量化结果，发现数据增强可能会增加估计的不确定性，并且其效果取决于多个因素。同时，该研究还通过随机转换的高维随机向量的函数的极限定理进行了证明。

    

    我们提供了确切量化数据增强如何影响估计的方差和极限分布的结果，并详细分析了几个具体模型。结果证实了机器学习实践中的一些观察，但也得出了意外的发现：数据增强可能会增加而不是减少估计的不确定性，比如经验预测风险。它可以充当正则化器，但在某些高维问题中却无法实现，并且可能会改变经验风险的双重下降峰值。总的来说，分析表明数据增强被赋予的几个属性要么是真的，要么是假的，而是取决于多个因素的组合-特别是数据分布，估计器的属性以及样本大小，增强数量和维数的相互作用。我们的主要理论工具是随机转换的高维随机向量的函数的极限定理。

    We provide results that exactly quantify how data augmentation affects the variance and limiting distribution of estimates, and analyze several specific models in detail. The results confirm some observations made in machine learning practice, but also lead to unexpected findings: Data augmentation may increase rather than decrease the uncertainty of estimates, such as the empirical prediction risk. It can act as a regularizer, but fails to do so in certain high-dimensional problems, and it may shift the double-descent peak of an empirical risk. Overall, the analysis shows that several properties data augmentation has been attributed with are not either true or false, but rather depend on a combination of factors -- notably the data distribution, the properties of the estimator, and the interplay of sample size, number of augmentations, and dimension. Our main theoretical tool is a limit theorem for functions of randomly transformed, high-dimensional random vectors. The proof draws on 
    


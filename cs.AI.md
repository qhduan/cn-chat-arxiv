# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Brain-grounding of semantic vectors improves neural decoding of visual stimuli](https://arxiv.org/abs/2403.15176) | 提出了一种表示学习框架，称为语义向量的脑接地，通过微调预训练的特征向量，使其更好地与人类大脑中视觉刺激的神经表示对齐。 |
| [^2] | [PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs](https://arxiv.org/abs/2312.15230) | 本研究中，通过仅更新少部分高度表达力的参数，我们挑战了全参数重新训练的做法，在修剪后恢复或甚至提升了性能。PERP方法显著减少了计算量和存储需求。 |
| [^3] | [Computational Copyright: Towards A Royalty Model for Music Generative AI](https://arxiv.org/abs/2312.06646) | 本文旨在解决音乐生成AI领域中的版权问题，提出了一种用于AI音乐生成平台的版税模型，并探讨了对AI生成音乐进行版权归因的算法解决方案。 |
| [^4] | [Combining Survival Analysis and Machine Learning for Mass Cancer Risk Prediction using EHR data.](http://arxiv.org/abs/2309.15039) | 该论文介绍了一种利用 EHR 数据进行大规模肿瘤风险预测的新方法，其创新之处在于只需利用历史的医疗服务代码和诊断信息来实现最小化的数据需求，通过将存活分析和机器学习相结合，可以在大规模应用中实现对患者癌症风险的个性化评估。 |

# 详细

[^1]: 语义向量的脑接地改善了神经解码视觉刺激

    Brain-grounding of semantic vectors improves neural decoding of visual stimuli

    [https://arxiv.org/abs/2403.15176](https://arxiv.org/abs/2403.15176)

    提出了一种表示学习框架，称为语义向量的脑接地，通过微调预训练的特征向量，使其更好地与人类大脑中视觉刺激的神经表示对齐。

    

    发展准确全面的算法来解码大脑内容是神经科学和脑机接口领域的一个长期目标。之前的研究已经证明了通过训练机器学习模型将大脑活动模式映射到一个语义向量表示的神经解码的可行性。为了解决这个问题，我们提出了一个表示学习框架，称为语义向量的脑接地，它对预训练的特征向量进行微调，以更好地与人类大脑中视觉刺激的神经表示对齐。

    arXiv:2403.15176v1 Announce Type: cross  Abstract: Developing algorithms for accurate and comprehensive neural decoding of mental contents is one of the long-cherished goals in the field of neuroscience and brain-machine interfaces. Previous studies have demonstrated the feasibility of neural decoding by training machine learning models to map brain activity patterns into a semantic vector representation of stimuli. These vectors, hereafter referred as pretrained feature vectors, are usually derived from semantic spaces based solely on image and/or text features and therefore they might have a totally different characteristics than how visual stimuli is represented in the human brain, resulting in limiting the capability of brain decoders to learn this mapping. To address this issue, we propose a representation learning framework, termed brain-grounding of semantic vectors, which fine-tunes pretrained feature vectors to better align with the neural representation of visual stimuli in t
    
[^2]: PERP: 在LLMs时代重新思考修剪-重新训练范式

    PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs

    [https://arxiv.org/abs/2312.15230](https://arxiv.org/abs/2312.15230)

    本研究中，通过仅更新少部分高度表达力的参数，我们挑战了全参数重新训练的做法，在修剪后恢复或甚至提升了性能。PERP方法显著减少了计算量和存储需求。

    

    神经网络可以通过修剪实现高效压缩，显著减少存储和计算需求同时保持预测性能。像迭代幅值修剪（IMP，Han等，2015）这样的简单而有效的方法可以去除不重要的参数，并需要昂贵的重新训练过程以在修剪后恢复性能。然而，随着大型语言模型（LLMs）的兴起，由于内存和计算限制，完全重新训练变得不可行。在本研究中，我们挑战了重新训练所有参数的做法，通过证明只更新少部分高度表达力的参数通常足以恢复甚至提高性能。令人惊讶的是，仅重新训练GPT-结构的0.27%-0.35%的参数即可在不同稀疏水平上实现与一次性IMP相当的性能。我们的方法，即修剪后参数高效重新训练（PERP），大大减少了计算量。

    Neural Networks can be efficiently compressed through pruning, significantly reducing storage and computational demands while maintaining predictive performance. Simple yet effective methods like Iterative Magnitude Pruning (IMP, Han et al., 2015) remove less important parameters and require a costly retraining procedure to recover performance after pruning. However, with the rise of Large Language Models (LLMs), full retraining has become infeasible due to memory and compute constraints. In this study, we challenge the practice of retraining all parameters by demonstrating that updating only a small subset of highly expressive parameters is often sufficient to recover or even improve performance compared to full retraining. Surprisingly, retraining as little as 0.27%-0.35% of the parameters of GPT-architectures achieves comparable performance to One Shot IMP across various sparsity levels. Our approach, Parameter-Efficient Retraining after Pruning (PERP), drastically reduces compute a
    
[^3]: 计算版权: 面向音乐生成AI的版税模型

    Computational Copyright: Towards A Royalty Model for Music Generative AI

    [https://arxiv.org/abs/2312.06646](https://arxiv.org/abs/2312.06646)

    本文旨在解决音乐生成AI领域中的版权问题，提出了一种用于AI音乐生成平台的版税模型，并探讨了对AI生成音乐进行版权归因的算法解决方案。

    

    生成AI的进步引发了版权挑战，在音乐行业尤为突出。本文关注这些挑战的经济方面，强调经济影响在版权领域中构成一个核心问题。黑盒生成AI技术的复杂性不仅表明，而且需要算法解决方案。然而，这样的解决方案在很大程度上缺失，导致监管挑战。我们旨在通过为AI音乐生成平台提出潜在的版税模型来弥补当前方法的差距。我们的方法涉及对Spotify和YouTube等平台现有版税模型的详细分析，并将其调整到AI生成音乐的独特背景中。我们面临的一个重要挑战是将AI生成的音乐归因于训练数据中有影响力的版权内容。为此，我们提出了利用数据归因的算法解决方案。

    The advancement of generative AI has given rise to pressing copyright challenges, particularly in music industry. This paper focuses on the economic aspects of these challenges, emphasizing that the economic impact constitutes a central issue in the copyright arena. The complexity of the black-box generative AI technologies not only suggests but necessitates algorithmic solutions. However, such solutions have been largely missing, leading to regulatory challenges in this landscape. We aim to bridge the gap in current approaches by proposing potential royalty models for revenue sharing on AI music generation platforms. Our methodology involves a detailed analysis of existing royalty models in platforms like Spotify and YouTube, and adapting these to the unique context of AI-generated music. A significant challenge we address is the attribution of AI-generated music to influential copyrighted content in the training data. To this end, we present algorithmic solutions employing data attri
    
[^4]: 结合存活分析和机器学习利用电子健康记录数据进行肿瘤风险预测

    Combining Survival Analysis and Machine Learning for Mass Cancer Risk Prediction using EHR data. (arXiv:2309.15039v1 [cs.LG])

    [http://arxiv.org/abs/2309.15039](http://arxiv.org/abs/2309.15039)

    该论文介绍了一种利用 EHR 数据进行大规模肿瘤风险预测的新方法，其创新之处在于只需利用历史的医疗服务代码和诊断信息来实现最小化的数据需求，通过将存活分析和机器学习相结合，可以在大规模应用中实现对患者癌症风险的个性化评估。

    

    纯粹的医学肿瘤筛查方法通常费用高昂、耗时长，并且仅适用于大规模应用。先进的人工智能（AI）方法在癌症检测方面发挥了巨大作用，但需要特定或深入的医学数据。这些方面影响了癌症筛查方法的大规模实施。因此，基于已有的电子健康记录（EHR）数据对患者进行大规模个性化癌症风险评估应用AI方法是一种颠覆性的改变。本文提出了一种利用EHR数据进行大规模肿瘤风险预测的新方法。与其他方法相比，我们的方法通过最小的数据贪婪策略脱颖而出，仅需要来自EHR的医疗服务代码和诊断历史。我们将问题形式化为二分类问题。该数据集包含了175441名不记名的患者（其中2861名被诊断为癌症）。作为基准，我们实现了一个基于循环神经网络（RNN）的解决方案。我们提出了一种方法，将存活分析和机器学习相结合，

    Purely medical cancer screening methods are often costly, time-consuming, and weakly applicable on a large scale. Advanced Artificial Intelligence (AI) methods greatly help cancer detection but require specific or deep medical data. These aspects affect the mass implementation of cancer screening methods. For these reasons, it is a disruptive change for healthcare to apply AI methods for mass personalized assessment of the cancer risk among patients based on the existing Electronic Health Records (EHR) volume.  This paper presents a novel method for mass cancer risk prediction using EHR data. Among other methods, our one stands out by the minimum data greedy policy, requiring only a history of medical service codes and diagnoses from EHR. We formulate the problem as a binary classification. This dataset contains 175 441 de-identified patients (2 861 diagnosed with cancer). As a baseline, we implement a solution based on a recurrent neural network (RNN). We propose a method that combine
    


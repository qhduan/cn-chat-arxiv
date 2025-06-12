# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Structured Information Matters: Incorporating Abstract Meaning Representation into LLMs for Improved Open-Domain Dialogue Evaluation](https://arxiv.org/abs/2404.01129) | 将抽象意义表示结合到LLMs中，提出了一个简单有效的框架用于改善开放领域对话评估 |
| [^2] | [Knowing Your Nonlinearities: Shapley Interactions Reveal the Underlying Structure of Data](https://arxiv.org/abs/2403.13106) | 该论文使用Shapley Taylor互动指数（STII）分析了底层数据结构对各种模态、任务和架构中模型表征的影响，发现了语言模型和语音模型中的新颖现象，并展示了特征交互如何直观反映对象边界。 |
| [^3] | [DREsS: Dataset for Rubric-based Essay Scoring on EFL Writing](https://arxiv.org/abs/2402.16733) | 本文发布了一个大型标准数据集DREsS，用于基于评分标准的自动作文评分，在提出了一种基于破坏的作文增强策略CASE后，这个数据集的基线结果提高了45.44％。 |
| [^4] | [Listen, Chat, and Edit: Text-Guided Soundscape Modification for Enhanced Auditory Experience](https://arxiv.org/abs/2402.03710) | 本研究引入了一种新颖的多模态声音混合编辑器，通过用户提供的文本指令实现对声音源的修改，实现了同时编辑多个声音源的能力，无需将它们分离。实验证明了该编辑器的实用性和效果。 |
| [^5] | [AMELI: Enhancing Multimodal Entity Linking with Fine-Grained Attributes.](http://arxiv.org/abs/2305.14725) | 提出了一种属性感知的多模态实体链接方法，并构建了一个大型数据集AMELI，实验证明了将属性信息纳入实体链接过程的重要性。 |

# 详细

[^1]: 结构化信息很重要：将抽象意义表示引入LLMs以改善开放领域对话评估

    Structured Information Matters: Incorporating Abstract Meaning Representation into LLMs for Improved Open-Domain Dialogue Evaluation

    [https://arxiv.org/abs/2404.01129](https://arxiv.org/abs/2404.01129)

    将抽象意义表示结合到LLMs中，提出了一个简单有效的框架用于改善开放领域对话评估

    

    arXiv:2404.01129v1 公告类型：新的 摘要：自动的开放领域对话评估已经引起越来越多的关注。可训练的评估指标通常是通过训练具有真正正例和随机选择的负例回复来训练的，导致它们倾向于将更高内容相似性的回复分配更高的得分给定一个上下文。然而，对抗性的负面回复具有与上下文高内容相似性，同时在语义上不同。因此，现有的评估指标不足以评估这类回复，导致与人类判断之间的相关性较低。虽然最近的研究已经显示出在利用大型语言模型（LLMs）进行开放领域对话评估方面有一定效果，但它们仍然在有效处理对抗性负面示例方面遇到挑战。在本文中，我们提出了一个简单而有效的框架用于开放领域对话评估，它结合了领域特定的语言模型（SLMs）。

    arXiv:2404.01129v1 Announce Type: new  Abstract: Automatic open-domain dialogue evaluation has attracted increasing attention. Trainable evaluation metrics are commonly trained with true positive and randomly selected negative responses, resulting in a tendency for them to assign a higher score to the responses that share higher content similarity with a given context. However, adversarial negative responses possess high content similarity with the contexts whilst being semantically different. Therefore, existing evaluation metrics are not robust enough to evaluate such responses, resulting in low correlations with human judgments. While recent studies have shown some efficacy in utilizing Large Language Models (LLMs) for open-domain dialogue evaluation, they still encounter challenges in effectively handling adversarial negative examples. In this paper, we propose a simple yet effective framework for open-domain dialogue evaluation, which combines domain-specific language models (SLMs
    
[^2]: 认识你的非线性：Shapley互动揭示数据的潜在结构

    Knowing Your Nonlinearities: Shapley Interactions Reveal the Underlying Structure of Data

    [https://arxiv.org/abs/2403.13106](https://arxiv.org/abs/2403.13106)

    该论文使用Shapley Taylor互动指数（STII）分析了底层数据结构对各种模态、任务和架构中模型表征的影响，发现了语言模型和语音模型中的新颖现象，并展示了特征交互如何直观反映对象边界。

    

    测量非线性特征交互是理解许多模型中复杂归因模式的一种已建立的方法。本文使用Shapley Taylor互动指数（STII）来分析底层数据结构对多种模态、任务和架构中模型表征的影响。在考虑掩码和自回归语言模型（MLMs和ALMs）中的语言结构时，我们发现STII在惯用表达中增加，MLMs随句法距离扩展STII，更多地依赖语法在其非线性结构中相比ALMs。我们的语音模型研究反映了口腔张开程度决定音素根据上下文变化的数量的原则。最后，我们研究图像分类器并说明特征交互直观反映对象边界。我们广泛的结果展示了跨学科工作和领域之间的益处。

    arXiv:2403.13106v1 Announce Type: cross  Abstract: Measuring nonlinear feature interaction is an established approach to understanding complex patterns of attribution in many models. In this paper, we use Shapley Taylor interaction indices (STII) to analyze the impact of underlying data structure on model representations in a variety of modalities, tasks, and architectures. Considering linguistic structure in masked and auto-regressive language models (MLMs and ALMs), we find that STII increases within idiomatic expressions and that MLMs scale STII with syntactic distance, relying more on syntax in their nonlinear structure than ALMs do. Our speech model findings reflect the phonetic principal that the openness of the oral cavity determines how much a phoneme varies based on its context. Finally, we study image classifiers and illustrate that feature interactions intuitively reflect object boundaries. Our wide range of results illustrates the benefits of interdisciplinary work and doma
    
[^3]: DREsS: 英语作为外语写作基于评分标准的数据集

    DREsS: Dataset for Rubric-based Essay Scoring on EFL Writing

    [https://arxiv.org/abs/2402.16733](https://arxiv.org/abs/2402.16733)

    本文发布了一个大型标准数据集DREsS，用于基于评分标准的自动作文评分，在提出了一种基于破坏的作文增强策略CASE后，这个数据集的基线结果提高了45.44％。

    

    自动化作文评分（AES）是英语作为外语写作教育中一种有用的工具，为学生和教师提供实时作文评分。然而，先前的AES模型是在与EFL写作教育实际场景不相关的作文和分数上进行训练的，并且通常由于缺乏适当的数据集而提供单一的整体评分。在本文中，我们发布了DREsS，这是一个用于基于评分标准的自动作文评分的大型标准数据集。DREsS包括三个子数据集：DREsS_New，DREsS_Std.和DREsS_CASE。我们收集了DREsS_New，这是一个由EFL本科生撰写并由英语教育专家评分的真实课堂数据集。我们还将现有的基于评分标准的作文评分数据集标准化为DREsS_Std。我们提出了一个名为CASE的基于破坏的作文增强策略，用于生成20K个DREsS_CASE的合成样本，并将基线结果提高了45.44％。

    arXiv:2402.16733v1 Announce Type: new  Abstract: Automated essay scoring (AES) is a useful tool in English as a Foreign Language (EFL) writing education, offering real-time essay scores for students and instructors. However, previous AES models were trained on essays and scores irrelevant to the practical scenarios of EFL writing education and usually provided a single holistic score due to the lack of appropriate datasets. In this paper, we release DREsS, a large-scale, standard dataset for rubric-based automated essay scoring. DREsS comprises three sub-datasets: DREsS_New, DREsS_Std., and DREsS_CASE. We collect DREsS_New, a real-classroom dataset with 1.7K essays authored by EFL undergraduate students and scored by English education experts. We also standardize existing rubric-based essay scoring datasets as DREsS_Std. We suggest CASE, a corruption-based augmentation strategy for essays, which generates 20K synthetic samples of DREsS_CASE and improves the baseline results by 45.44%. 
    
[^4]: 听、聊、编辑：基于文本指导的声景修改以增强听觉体验

    Listen, Chat, and Edit: Text-Guided Soundscape Modification for Enhanced Auditory Experience

    [https://arxiv.org/abs/2402.03710](https://arxiv.org/abs/2402.03710)

    本研究引入了一种新颖的多模态声音混合编辑器，通过用户提供的文本指令实现对声音源的修改，实现了同时编辑多个声音源的能力，无需将它们分离。实验证明了该编辑器的实用性和效果。

    

    在日常生活中，我们遇到各种各样的声音，有些是我们期望的，有些是我们不希望的，对它们的存在和音量的控制有限。我们的工作引入了一种新颖的多模态声音混合编辑器"听、聊、编辑"(LCE)，该编辑器根据用户提供的文本指令修改混合中的每个声音源。LCE通过用户友好的聊天界面以及其在不需要将声音源分离的情况下同时对多个声音源进行编辑的能力而与众不同。用户输入开放性的文本提示，这些提示由大型语言模型解释，用于创建编辑声音混合的语义滤波器。系统然后将混合解析成其组成部分，应用语义滤波器，并将其重新组装成期望的输出。我们开发了一个包括语音和各种音频源以及用于不同编辑任务的文本提示的160小时数据集，包括提取、删除和音量控制。我们的实验证明。

    In daily life, we encounter a variety of sounds, both desirable and undesirable, with limited control over their presence and volume. Our work introduces "Listen, Chat, and Edit" (LCE), a novel multimodal sound mixture editor that modifies each sound source in a mixture based on user-provided text instructions. LCE distinguishes itself with a user-friendly chat interface and its unique ability to edit multiple sound sources simultaneously within a mixture, without needing to separate them. Users input open-vocabulary text prompts, which are interpreted by a large language model to create a semantic filter for editing the sound mixture. The system then decomposes the mixture into its components, applies the semantic filter, and reassembles it into the desired output. We developed a 160-hour dataset with over 100k mixtures, including speech and various audio sources, along with text prompts for diverse editing tasks like extraction, removal, and volume control. Our experiments demonstrat
    
[^5]: AMELI:细粒度属性增强多模态实体链接

    AMELI: Enhancing Multimodal Entity Linking with Fine-Grained Attributes. (arXiv:2305.14725v1 [cs.CL])

    [http://arxiv.org/abs/2305.14725](http://arxiv.org/abs/2305.14725)

    提出了一种属性感知的多模态实体链接方法，并构建了一个大型数据集AMELI，实验证明了将属性信息纳入实体链接过程的重要性。

    

    我们提出了一种属性感知的多模态实体链接方法，其中输入是一个由文本和图像描述的提及，目标是从一个多模态知识库中预测相应的目标实体，其中每个实体都是用文本描述、视觉图像和一组属性值描述的。为了支持这项研究，我们构建了一个大型数据集AMELI，其中包含18,472个评论和35,598个产品。我们在AMELI上进行了实验，使用当前最先进的多模态实体链接方法和我们增强的属性感知模型来建立基准性能，并展示了将属性信息纳入实体链接过程中的重要性。据我们所知，我们是第一个为属性感知多模态实体链接任务建立基准数据集和解决方案的团队。数据集和代码将公开提供。

    We propose attribute-aware multimodal entity linking, where the input is a mention described with a text and image, and the goal is to predict the corresponding target entity from a multimodal knowledge base (KB) where each entity is also described with a text description, a visual image and a set of attributes and values. To support this research, we construct AMELI, a large-scale dataset consisting of 18,472 reviews and 35,598 products. To establish baseline performance on AMELI, we experiment with the current state-of-the-art multimodal entity linking approaches and our enhanced attribute-aware model and demonstrate the importance of incorporating the attribute information into the entity linking process. To be best of our knowledge, we are the first to build benchmark dataset and solutions for the attribute-aware multimodal entity linking task. Datasets and codes will be made publicly available.
    


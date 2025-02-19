# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AutoMathText: Autonomous Data Selection with Language Models for Mathematical Texts](https://arxiv.org/abs/2402.07625) | 本论文介绍了一种自主数据选择策略，利用语言模型进行数学文本的自动评估和选择，并通过连续预训练显著提高了数学推理能力。主要创新包括利用元提示语言模型作为验证器，发布了高质量的AutoMathText数据集，并实现了预训练令牌效率的提升。 |
| [^2] | [Multimodal Sentiment Analysis with Missing Modality: A Knowledge-Transfer Approach.](http://arxiv.org/abs/2401.10747) | 本文提出了一种知识迁移方法，用于在缺失模态下进行多模态情感分析。通过翻译不同模态之间的内容以重构缺失的音频模态，并利用跨模态注意机制进行情感预测，实验证明了该方法在多个数据集上表现出显著的改进和与完整多模态监督方法相媲美的效果。 |
| [^3] | [A Framework for Responsible Development of Automated Student Feedback with Generative AI.](http://arxiv.org/abs/2308.15334) | 一种基于生成AI的自动学生反馈框架可以提供丰富的反馈，但引入了伦理问题，并需要解决“多数人的暴政”和忽视长尾中少数群体需求的挑战。 |
| [^4] | [Seed-Guided Topic Discovery with Out-of-Vocabulary Seeds.](http://arxiv.org/abs/2205.01845) | 本文提出了一种带有未登录词种子的主题发现方法，将预训练语言模型和来自输入语料库的局部语义相结合，实验证明了该方法在主题连贯性、准确性和多样性方面的有效性。 |
| [^5] | [Media Slant is Contagious.](http://arxiv.org/abs/2202.07269) | 本文研究了国家有线电视新闻对美国本土报纸的影响，发现当地报纸的内容会因为当地 FNC 观众数量的增加而趋向于 FNC 的倾向，并且有线电视倾向会极化地方新闻内容。 |

# 详细

[^1]: AutoMathText：使用语言模型进行数学文本的自主数据选择

    AutoMathText: Autonomous Data Selection with Language Models for Mathematical Texts

    [https://arxiv.org/abs/2402.07625](https://arxiv.org/abs/2402.07625)

    本论文介绍了一种自主数据选择策略，利用语言模型进行数学文本的自动评估和选择，并通过连续预训练显著提高了数学推理能力。主要创新包括利用元提示语言模型作为验证器，发布了高质量的AutoMathText数据集，并实现了预训练令牌效率的提升。

    

    为了通过持续的预训练改善语言模型在数学推理方面的能力，我们引入了一种新颖的策略，利用基础语言模型进行自主数据选择。与传统的有人工标注数据的监督微调或训练过的分类器不同，我们的方法利用元提示语言模型作为零样本验证器，自主评估和选择高质量的数学内容，并发布了经过策划的开源AutoMathText数据集，其中包含超过200GB的数据。为了证明我们方法的有效性，我们对AutoMathText数据集进行了连续预训练，使得7B参数的Mistral语言模型在MATH数据集上的下游性能大幅提升，而令牌数量比之前的连续预训练工作减少了几个数量级。我们的方法展示了基准的预训练令牌效率提高了2倍，突显了我们方法在增强中的潜力。

    To improve language models' proficiency in mathematical reasoning via continual pretraining, we introduce a novel strategy that leverages base language models for autonomous data selection. Departing from conventional supervised fine-tuning or trained classifiers with human-annotated data, our approach utilizes meta-prompted language models as zero-shot verifiers to autonomously evaluate and select high-quality mathematical content, and we release the curated open-source AutoMathText dataset encompassing over 200GB of data. To demonstrate the efficacy of our method, we continuously pretrained a 7B-parameter Mistral language model on the AutoMathText dataset, achieving substantial improvements in downstream performance on the MATH dataset with a token amount reduced by orders of magnitude compared to previous continuous pretraining works. Our method showcases a 2 times increase in pretraining token efficiency compared to baselines, underscoring the potential of our approach in enhancing
    
[^2]: 缺失模态下的多模态情感分析:一种知识迁移方法

    Multimodal Sentiment Analysis with Missing Modality: A Knowledge-Transfer Approach. (arXiv:2401.10747v1 [cs.SD])

    [http://arxiv.org/abs/2401.10747](http://arxiv.org/abs/2401.10747)

    本文提出了一种知识迁移方法，用于在缺失模态下进行多模态情感分析。通过翻译不同模态之间的内容以重构缺失的音频模态，并利用跨模态注意机制进行情感预测，实验证明了该方法在多个数据集上表现出显著的改进和与完整多模态监督方法相媲美的效果。

    

    多模态情感分析旨在通过视觉、语言和声音线索来识别个体表达的情绪。然而，现有研究大多假设在训练和测试过程中所有模态都是可用的，这使得它们的算法容易受到缺失模态的影响。在本文中，我们提出了一种新颖的知识迁移网络，用于在不同模态之间进行翻译，以重构缺失的音频模态。此外，我们还开发了一种跨模态注意机制，以保留重构和观察到的模态的最大信息，用于情感预测。在三个公开数据集上进行的大量实验证明了相对于基线算法的显著改进，并实现了与具有完整多模态监督的先前方法相媲美的结果。

    Multimodal sentiment analysis aims to identify the emotions expressed by individuals through visual, language, and acoustic cues. However, most of the existing research efforts assume that all modalities are available during both training and testing, making their algorithms susceptible to the missing modality scenario. In this paper, we propose a novel knowledge-transfer network to translate between different modalities to reconstruct the missing audio modalities. Moreover, we develop a cross-modality attention mechanism to retain the maximal information of the reconstructed and observed modalities for sentiment prediction. Extensive experiments on three publicly available datasets demonstrate significant improvements over baselines and achieve comparable results to the previous methods with complete multi-modality supervision.
    
[^3]: 一种负责任开发基于生成AI的自动学生反馈框架

    A Framework for Responsible Development of Automated Student Feedback with Generative AI. (arXiv:2308.15334v1 [cs.CY])

    [http://arxiv.org/abs/2308.15334](http://arxiv.org/abs/2308.15334)

    一种基于生成AI的自动学生反馈框架可以提供丰富的反馈，但引入了伦理问题，并需要解决“多数人的暴政”和忽视长尾中少数群体需求的挑战。

    

    提供丰富的反馈对于支持学生学习至关重要。最近生成AI尤其是大规模语言模型的进展，为向学生提供可重复、可扩展和即时生成的自动反馈提供了机会，使得之前稀缺且昂贵的学习资源变得丰富起来。从技术角度而言，这种方法是可行的，得益于最近人工智能和自然语言处理的进步；然而，采用这些技术也引入了一系列潜在的伦理问题，需要认真考虑。人工智能系统的吸引力在于它们可以有效地自动化最乏味的任务；但是这也可能导致“多数人的暴政”，即忽视了长尾中少数群体的需求，因为这些需求很难自动化。因此，开发能够产生有价值和真实的机器学习模型变得至关重要。

    Providing rich feedback to students is essential for supporting student learning. Recent advances in generative AI, particularly within large language modelling (LLM), provide the opportunity to deliver repeatable, scalable and instant automatically generated feedback to students, making abundant a previously scarce and expensive learning resource. Such an approach is feasible from a technical perspective due to these recent advances in Artificial Intelligence (AI) and Natural Language Processing (NLP); while the potential upside is a strong motivator, doing so introduces a range of potential ethical issues that must be considered as we apply these technologies. The attractiveness of AI systems is that they can effectively automate the most mundane tasks; but this risks introducing a "tyranny of the majority", where the needs of minorities in the long tail are overlooked because they are difficult to automate.  Developing machine learning models that can generate valuable and authentic
    
[^4]: 带有未登录词种子的主题发现

    Seed-Guided Topic Discovery with Out-of-Vocabulary Seeds. (arXiv:2205.01845v1 [cs.CL] CROSS LISTED)

    [http://arxiv.org/abs/2205.01845](http://arxiv.org/abs/2205.01845)

    本文提出了一种带有未登录词种子的主题发现方法，将预训练语言模型和来自输入语料库的局部语义相结合，实验证明了该方法在主题连贯性、准确性和多样性方面的有效性。

    

    多年来，从文本语料库中发现潜在主题一直是研究的课题。许多现有的主题模型采用完全无监督的设置，由于它们无法利用用户指导，所以它们发现的主题可能不符合用户的特定兴趣。虽然存在利用用户提供的种子词来发现主题代表词的种子引导主题发现方法，但它们较少关注两个因素：(1)未登录词种子的存在和(2)预训练语言模型的能力。在本文中，我们将种子引导主题发现的任务推广到允许未登录词种子。我们提出了一个新的框架，名为SeeTopic，在其中PLM的通用知识和从输入语料库中学习的局部语义可以相互受益。在来自不同领域的三个真实数据集上的实验证明了SeeTopic在主题连贯性、准确性和多样性方面的有效性。

    Discovering latent topics from text corpora has been studied for decades. Many existing topic models adopt a fully unsupervised setting, and their discovered topics may not cater to users' particular interests due to their inability of leveraging user guidance. Although there exist seed-guided topic discovery approaches that leverage user-provided seeds to discover topic-representative terms, they are less concerned with two factors: (1) the existence of out-of-vocabulary seeds and (2) the power of pre-trained language models (PLMs). In this paper, we generalize the task of seed-guided topic discovery to allow out-of-vocabulary seeds. We propose a novel framework, named SeeTopic, wherein the general knowledge of PLMs and the local semantics learned from the input corpus can mutually benefit each other. Experiments on three real datasets from different domains demonstrate the effectiveness of SeeTopic in terms of topic coherence, accuracy, and diversity.
    
[^5]: 媒体倾向是具有传染性的。

    Media Slant is Contagious. (arXiv:2202.07269v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2202.07269](http://arxiv.org/abs/2202.07269)

    本文研究了国家有线电视新闻对美国本土报纸的影响，发现当地报纸的内容会因为当地 FNC 观众数量的增加而趋向于 FNC 的倾向，并且有线电视倾向会极化地方新闻内容。

    

    本研究考察了媒体倾向的传播，具体来说是国家有线电视新闻对美国本土报纸（2005-2008）的影响。我们使用一种基于 Fox News Channel（FNC）、CNN 和 MSNBC 内容的有线电视倾向文本度量方法，分析地方报纸如何采用 FNC 的倾向而不是 CNN/MSNBC 的倾向。研究结果显示，地方新闻随着当地 FNC 观众人数的外部增长而变得更加类似于 FNC 的内容。这种转变不仅限于从有线电视借鉴，而是地方报纸自身内容的改变。此外，有线电视倾向极化了地方新闻内容。

    We examine the diffusion of media slant, specifically how partisan content from national cable news affects local newspapers in the U.S., 2005-2008. We use a text-based measure of cable news slant trained on content from Fox News Channel (FNC), CNN, and MSNBC to analyze how local newspapers adopt FNC's slant over CNN/MSNBC's. Our findings show that local news becomes more similar to FNC content in response to an exogenous increase in local FNC viewership. This shift is not limited to borrowing from cable news, but rather, local newspapers' own content changes. Further, cable TV slant polarizes local news content.
    


# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CODE-ACCORD: A Corpus of Building Regulatory Data for Rule Generation towards Automatic Compliance Checking](https://arxiv.org/abs/2403.02231) | 介绍了一个独特的数据集CODE-ACCORD，旨在解决自动合规检查中解释建筑法规的挑战，成为机器可读规则生成的基础。 |
| [^2] | [Causal Learning for Trustworthy Recommender Systems: A Survey](https://arxiv.org/abs/2402.08241) | 本文概述了从因果学习的角度对可信赖的推荐系统进行调查。因果学习提供了一种解决TRS中潜在偏见和噪声的方法，同时提供了深入的解释。然而，在这个充满活力的领域中，缺乏一个及时的调查。 |
| [^3] | [Seed-Guided Topic Discovery with Out-of-Vocabulary Seeds.](http://arxiv.org/abs/2205.01845) | 本文提出了一种带有未登录词种子的主题发现方法，将预训练语言模型和来自输入语料库的局部语义相结合，实验证明了该方法在主题连贯性、准确性和多样性方面的有效性。 |

# 详细

[^1]: CODE-ACCORD：用于规则生成的建筑法规数据语料库

    CODE-ACCORD: A Corpus of Building Regulatory Data for Rule Generation towards Automatic Compliance Checking

    [https://arxiv.org/abs/2403.02231](https://arxiv.org/abs/2403.02231)

    介绍了一个独特的数据集CODE-ACCORD，旨在解决自动合规检查中解释建筑法规的挑战，成为机器可读规则生成的基础。

    

    自动合规检查（ACC）在建筑、工程和施工（AEC）领域内的自动合规检查需要自动解释建筑法规，以发挥其全部潜力。然而，从文本规则中提取信息以将其转换为机器可读格式由于自然语言的复杂性以及仅能支持先进的机器学习技术的有限资源而成为一个挑战。为了解决这一挑战，我们介绍了一个独特的数据集CODE-ACCORD，这是在欧盟Horizon ACCORD项目下编制的。CODE-ACCORD包含862个来自英格兰和芬兰建筑法规的自包含句子。与我们的核心目标一致，即促进从文本中提取信息以生成机器可读规则，每个句子都注释了实体和关系。实体代表特定组件，如“窗户”和“烟雾探测器”，而re

    arXiv:2403.02231v1 Announce Type: new  Abstract: Automatic Compliance Checking (ACC) within the Architecture, Engineering, and Construction (AEC) sector necessitates automating the interpretation of building regulations to achieve its full potential. However, extracting information from textual rules to convert them to a machine-readable format has been a challenge due to the complexities associated with natural language and the limited resources that can support advanced machine-learning techniques. To address this challenge, we introduce CODE-ACCORD, a unique dataset compiled under the EU Horizon ACCORD project. CODE-ACCORD comprises 862 self-contained sentences extracted from the building regulations of England and Finland. Aligned with our core objective of facilitating information extraction from text for machine-readable rule generation, each sentence was annotated with entities and relations. Entities represent specific components such as "window" and "smoke detectors", while re
    
[^2]: 可信赖的推荐系统的因果推理技术：一项调查的综述

    Causal Learning for Trustworthy Recommender Systems: A Survey

    [https://arxiv.org/abs/2402.08241](https://arxiv.org/abs/2402.08241)

    本文概述了从因果学习的角度对可信赖的推荐系统进行调查。因果学习提供了一种解决TRS中潜在偏见和噪声的方法，同时提供了深入的解释。然而，在这个充满活力的领域中，缺乏一个及时的调查。

    

    推荐系统（RS）在在线内容发现和个性化决策方面取得了显著进展。然而，RS中出现的漏洞促使了向可信赖的推荐系统（TRS）的范式转变。尽管TRS取得了许多进展，但大部分都集中在数据相关性上，而忽视了推荐中的基本因果关系。这个缺点阻碍了TRS在解决可信赖性问题时识别原因，导致公平性、鲁棒性和可解释性受到限制。为了弥补这一差距，因果学习作为一类有前途的方法出现，以增强TRS。这些方法以可靠的因果关系为基础，在减轻各种偏见和噪声的同时，为TRS提供了深入的解释。然而，在这个充满活力的领域中，缺乏一个及时的调查。本文从因果学习的角度对TRS进行了概述。我们首先介绍了因果导向TRS（CTRS）的优势和常见程序。接下来，我们确定了潜在的因果学习方法在TRS中的应用领域。

    Recommender Systems (RS) have significantly advanced online content discovery and personalized decision-making. However, emerging vulnerabilities in RS have catalyzed a paradigm shift towards Trustworthy RS (TRS). Despite numerous progress on TRS, most of them focus on data correlations while overlooking the fundamental causal nature in recommendation. This drawback hinders TRS from identifying the cause in addressing trustworthiness issues, leading to limited fairness, robustness, and explainability. To bridge this gap, causal learning emerges as a class of promising methods to augment TRS. These methods, grounded in reliable causality, excel in mitigating various biases and noises while offering insightful explanations for TRS. However, there lacks a timely survey in this vibrant area. This paper creates an overview of TRS from the perspective of causal learning. We begin by presenting the advantages and common procedures of Causality-oriented TRS (CTRS). Then, we identify potential 
    
[^3]: 带有未登录词种子的主题发现

    Seed-Guided Topic Discovery with Out-of-Vocabulary Seeds. (arXiv:2205.01845v1 [cs.CL] CROSS LISTED)

    [http://arxiv.org/abs/2205.01845](http://arxiv.org/abs/2205.01845)

    本文提出了一种带有未登录词种子的主题发现方法，将预训练语言模型和来自输入语料库的局部语义相结合，实验证明了该方法在主题连贯性、准确性和多样性方面的有效性。

    

    多年来，从文本语料库中发现潜在主题一直是研究的课题。许多现有的主题模型采用完全无监督的设置，由于它们无法利用用户指导，所以它们发现的主题可能不符合用户的特定兴趣。虽然存在利用用户提供的种子词来发现主题代表词的种子引导主题发现方法，但它们较少关注两个因素：(1)未登录词种子的存在和(2)预训练语言模型的能力。在本文中，我们将种子引导主题发现的任务推广到允许未登录词种子。我们提出了一个新的框架，名为SeeTopic，在其中PLM的通用知识和从输入语料库中学习的局部语义可以相互受益。在来自不同领域的三个真实数据集上的实验证明了SeeTopic在主题连贯性、准确性和多样性方面的有效性。

    Discovering latent topics from text corpora has been studied for decades. Many existing topic models adopt a fully unsupervised setting, and their discovered topics may not cater to users' particular interests due to their inability of leveraging user guidance. Although there exist seed-guided topic discovery approaches that leverage user-provided seeds to discover topic-representative terms, they are less concerned with two factors: (1) the existence of out-of-vocabulary seeds and (2) the power of pre-trained language models (PLMs). In this paper, we generalize the task of seed-guided topic discovery to allow out-of-vocabulary seeds. We propose a novel framework, named SeeTopic, wherein the general knowledge of PLMs and the local semantics learned from the input corpus can mutually benefit each other. Experiments on three real datasets from different domains demonstrate the effectiveness of SeeTopic in terms of topic coherence, accuracy, and diversity.
    


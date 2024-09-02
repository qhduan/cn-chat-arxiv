# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Frankenstein: Generating Semantic-Compositional 3D Scenes in One Tri-Plane](https://arxiv.org/abs/2403.16210) | Frankenstein是一个框架，可以在单个通道中同时生成多个语义相关的3D形状，为生成房间内部和人类化身等场景提供了有希望的结果。 |
| [^2] | [Evaluating Named Entity Recognition: Comparative Analysis of Mono- and Multilingual Transformer Models on Brazilian Corporate Earnings Call Transcriptions](https://arxiv.org/abs/2403.12212) | 本研究通过引入新方法，将标记分类任务重新构建为文本生成问题，评估了在巴西银行财报电话转录中使用的单语和多语言Transformer模型的性能。 |
| [^3] | [Emergence of Social Norms in Large Language Model-based Agent Societies](https://arxiv.org/abs/2403.08251) | 提出了第一个赋予大型语言模型Agent群体内社会规范出现的生成式Agent架构CRSEC，实验证明其能力。 |
| [^4] | [Large Language Multimodal Models for 5-Year Chronic Disease Cohort Prediction Using EHR Data](https://arxiv.org/abs/2403.04785) | 本研究提出了一种大型语言多模型（LLMMs）框架，结合临床笔记和实验室检验结果的多模态数据，用于预测慢性疾病风险。 |
| [^5] | [Advancing Biomedical Text Mining with Community Challenges](https://arxiv.org/abs/2403.04261) | 社区挑战评估竞赛在促进生物医学文本挖掘研究中的技术创新和跨学科合作方面起着重要作用。 |
| [^6] | [SimuCourt: Building Judicial Decision-Making Agents with Real-world Judgement Documents](https://arxiv.org/abs/2403.02959) | 提出了SimuCourt司法基准，包括真实世界的司法文件，并引入了司法决策任务和多代理框架，评估了代理的司法分析和决策能力 |
| [^7] | [Language models align with human judgments on key grammatical constructions](https://arxiv.org/abs/2402.01676) | 本研究通过对比评估发现，大型语言模型（LLMs）在俘获人类行为方面的表现非常出色，不仅整体准确率高，而且能够捕捉到人类语言判断中的细微差异。 |
| [^8] | [Generative Design of Crystal Structures by Point Cloud Representations and Diffusion Model.](http://arxiv.org/abs/2401.13192) | 本研究提出了一种基于点云和扩散模型的晶体结构生成设计框架，并通过重建输入结构和生成全新材料的实验证明了其有效性和潜力。 |
| [^9] | [Tailoring Adversarial Attacks on Deep Neural Networks for Targeted Class Manipulation Using DeepFool Algorithm.](http://arxiv.org/abs/2310.13019) | 本文提出了一种增强版DeepFool算法，名为Targeted DeepFool，可以针对特定类别进行错误分类，并引入了最小置信度分数要求超参数来提高灵活性。 |
| [^10] | [DiffLoad: Uncertainty Quantification in Load Forecasting with Diffusion Model.](http://arxiv.org/abs/2306.01001) | 本文提出了一种扩散模型中的负荷预测不确定性量化方法，采用Seq2Seq网络结构来分离两种类型的不确定性并处理异常情况，不仅着眼于预测条件期望值。 |
| [^11] | [Scale-Adaptive Balancing of Exploration and Exploitation in Classical Planning.](http://arxiv.org/abs/2305.09840) | 本文提出了一种MCTS/THTS算法GreedyUCT-Normal，该算法能够通过采用奖励变化的尺度处理不同尺度的分布，以在经典计划中平衡探索和开发。 |
| [^12] | [Incorporating Unlabelled Data into Bayesian Neural Networks.](http://arxiv.org/abs/2304.01762) | 该论文提出了一种利用未标记数据学习贝叶斯神经网络（BNNs）的对比框架，通过该框架提出了一种同时具备自监督学习的标签效率和贝叶斯方法中的不确定性估计的实用BNN算法。最后，该方法在半监督和低预算主动学习问题中展现出了数据高效学习的优势。 |
| [^13] | [Optimizing Agent Collaboration through Heuristic Multi-Agent Planning.](http://arxiv.org/abs/2301.01246) | 提出了一种启发式多智能体规划算法，解决了涉及不同类型智能体的问题，比现有算法表现更好。 |
| [^14] | [Scalable Multi-Agent Reinforcement Learning for Warehouse Logistics with Robotic and Human Co-Workers.](http://arxiv.org/abs/2212.11498) | 该论文提出了一种可扩展的多智能体强化学习方法，用于仓库物流中的机器人和人类同事合作。他们通过分层的MARL算法，让经理和工人代理根据全局目标进行协同训练，以最大化拣货速率。 |
| [^15] | [Does CLIP Bind Concepts? Probing Compositionality in Large Image Models.](http://arxiv.org/abs/2212.10537) | 本文分析了大型神经网络模型CLIP的组合性能力以及以结构敏感的方式捆绑变量的能力，发现其能够在单一对象的情况下组合概念，但在需要概念捆绑的情况下性能显著下降。 |

# 详细

[^1]: Frankenstein: 在一个三面位平面中生成语义-组合式3D场景

    Frankenstein: Generating Semantic-Compositional 3D Scenes in One Tri-Plane

    [https://arxiv.org/abs/2403.16210](https://arxiv.org/abs/2403.16210)

    Frankenstein是一个框架，可以在单个通道中同时生成多个语义相关的3D形状，为生成房间内部和人类化身等场景提供了有希望的结果。

    

    我们提出了Frankenstein，这是一个基于扩散的框架，可以在单个通道中生成语义-组合式3D场景。与现有方法输出单个统一的3D形状不同，Frankenstein同时生成多个独立的形状，每个对应一个语义上有意义的部分。3D场景信息编码在一个三面位平面张量中，从中可以解码多个符号距离函数（SDF）场以表示组合形状。在训练期间，一个自编码器将三面位平面压缩到潜在空间，然后使用去噪扩散过程来逼近组合场景的分布。Frankenstein在生成房间内部和具有自动分离部分的人类化身方面表现出有希望的结果。生成的场景有助于许多下游应用，例如部分重贴图、房间或化身衣服的对象重新排列。

    arXiv:2403.16210v1 Announce Type: cross  Abstract: We present Frankenstein, a diffusion-based framework that can generate semantic-compositional 3D scenes in a single pass. Unlike existing methods that output a single, unified 3D shape, Frankenstein simultaneously generates multiple separated shapes, each corresponding to a semantically meaningful part. The 3D scene information is encoded in one single tri-plane tensor, from which multiple Singed Distance Function (SDF) fields can be decoded to represent the compositional shapes. During training, an auto-encoder compresses tri-planes into a latent space, and then the denoising diffusion process is employed to approximate the distribution of the compositional scenes. Frankenstein demonstrates promising results in generating room interiors as well as human avatars with automatically separated parts. The generated scenes facilitate many downstream applications, such as part-wise re-texturing, object rearrangement in the room or avatar clo
    
[^2]: 评估命名实体识别：比较分析巴西公司财报电话转录上的单语和多语言Transformer模型

    Evaluating Named Entity Recognition: Comparative Analysis of Mono- and Multilingual Transformer Models on Brazilian Corporate Earnings Call Transcriptions

    [https://arxiv.org/abs/2403.12212](https://arxiv.org/abs/2403.12212)

    本研究通过引入新方法，将标记分类任务重新构建为文本生成问题，评估了在巴西银行财报电话转录中使用的单语和多语言Transformer模型的性能。

    

    命名实体识别（NER）是一种从文本文档中提取信息的自然语言处理技术。然而，现有关于NER的大部分研究都集中在英语文档上，导致缺乏专门针对葡萄牙语财务领域的数据集。本研究解决了金融领域内NER需求，并侧重于从巴西银行财报电话转录中提取的葡萄牙语文本。通过整理包括384个转录的综合数据集，并利用弱监督技术进行注释，我们评估了在葡萄牙语（BERTimbau和PTT5）训练的单语模型以及多语言模型（mBERT和mT5）的性能。值得注意的是，我们引入了一种新方法，将标记分类任务重新构建为文本生成问题，从而实现T5模型的微调和评估。在模型微调之后，

    arXiv:2403.12212v1 Announce Type: cross  Abstract: Named Entity Recognition (NER) is a Natural Language Processing technique for extracting information from textual documents. However, much of the existing research on NER has been centered around English-language documents, leaving a gap in the availability of datasets tailored to the financial domain in Portuguese. This study addresses the need for NER within the financial domain, focusing on Portuguese-language texts extracted from earnings call transcriptions of Brazilian banks. By curating a comprehensive dataset comprising 384 transcriptions and leveraging weak supervision techniques for annotation, we evaluate the performance of monolingual models trained on Portuguese (BERTimbau and PTT5) and multilingual models (mBERT and mT5). Notably, we introduce a novel approach that reframes the token classification task as a text generation problem, enabling fine-tuning and evaluation of T5 models. Following the fine-tuning of the models,
    
[^3]: 基于大型语言模型的Agent社会中社会规范的出现

    Emergence of Social Norms in Large Language Model-based Agent Societies

    [https://arxiv.org/abs/2403.08251](https://arxiv.org/abs/2403.08251)

    提出了第一个赋予大型语言模型Agent群体内社会规范出现的生成式Agent架构CRSEC，实验证明其能力。

    

    社会规范的出现吸引了社会科学、认知科学以及人工智能等各个领域的广泛关注。本文提出了第一个赋予大型语言模型Agent群体内社会规范出现的生成式Agent架构CRSEC。我们的架构包括四个模块：Creation & Representation、Spreading、Evaluation和Compliance。我们的架构处理了几个关键方面的紧急过程：(i)社会规范的来源，(ii)它们如何被正式表示，(iii)它们如何通过Agent的交流和观察传播，(iv)如何通过合理检查进行检查并在长期内进行综合，(v)如何被纳入Agent的计划和行动中。我们在Smallville沙盒游戏环境中进行的实验展示了我们的架构的能力。

    arXiv:2403.08251v1 Announce Type: cross  Abstract: The emergence of social norms has attracted much interest in a wide array of disciplines, ranging from social science and cognitive science to artificial intelligence. In this paper, we propose the first generative agent architecture that empowers the emergence of social norms within a population of large language model-based agents. Our architecture, named CRSEC, consists of four modules: Creation & Representation, Spreading, Evaluation, and Compliance. Our architecture addresses several important aspects of the emergent processes all in one: (i) where social norms come from, (ii) how they are formally represented, (iii) how they spread through agents' communications and observations, (iv) how they are examined with a sanity check and synthesized in the long term, and (v) how they are incorporated into agents' planning and actions. Our experiments deployed in the Smallville sandbox game environment demonstrate the capability of our ar
    
[^4]: 使用电子健康记录数据预测5年慢性疾病队列的大型语言多模型

    Large Language Multimodal Models for 5-Year Chronic Disease Cohort Prediction Using EHR Data

    [https://arxiv.org/abs/2403.04785](https://arxiv.org/abs/2403.04785)

    本研究提出了一种大型语言多模型（LLMMs）框架，结合临床笔记和实验室检验结果的多模态数据，用于预测慢性疾病风险。

    

    慢性疾病如糖尿病是全球发病率和死亡率的主要原因。本研究从台湾医院数据库收集了五年的电子健康记录数据，包括1,420,596份临床笔记、387,392份实验室检验结果以及超过1,505种实验室检验项目，重点研究了用于研究预训练大型语言模型的方法。我们提出了一种新颖的大型语言多模型（LLMMs）框架，将临床笔记和实验室检验结果的多模态数据相结合，用于预测慢性疾病风险。我们的方法结合了文本嵌入编码器和多头注意力层来学习实验室检验数值，利用深度神经网络（DNN）模块进行预测。

    arXiv:2403.04785v1 Announce Type: cross  Abstract: Chronic diseases such as diabetes are the leading causes of morbidity and mortality worldwide. Numerous research studies have been attempted with various deep learning models in diagnosis. However, most previous studies had certain limitations, including using publicly available datasets (e.g. MIMIC), and imbalanced data. In this study, we collected five-year electronic health records (EHRs) from the Taiwan hospital database, including 1,420,596 clinical notes, 387,392 laboratory test results, and more than 1,505 laboratory test items, focusing on research pre-training large language models. We proposed a novel Large Language Multimodal Models (LLMMs) framework incorporating multimodal data from clinical notes and laboratory test results for the prediction of chronic disease risk. Our method combined a text embedding encoder and multi-head attention layer to learn laboratory test values, utilizing a deep neural network (DNN) module to 
    
[^5]: 通过社区挑战推动生物医学文本挖掘的发展

    Advancing Biomedical Text Mining with Community Challenges

    [https://arxiv.org/abs/2403.04261](https://arxiv.org/abs/2403.04261)

    社区挑战评估竞赛在促进生物医学文本挖掘研究中的技术创新和跨学科合作方面起着重要作用。

    

    生物医学研究领域积累了大量来自科学文献、电子病历、临床试验报告和社交媒体等各方面的文本数据，然而手动处理和分析这些庞大且复杂的资源是耗时且低效的。为了解决这一挑战，生物医学文本挖掘，也称为生物医学自然语言处理，备受关注。社区挑战评估竞赛在促进生物医学文本挖掘研究中的技术创新和跨学科合作方面发挥了重要作用。这些挑战为研究人员提供了开发生物医学研究中数据挖掘和信息处理的最新解决方案的平台。在本文中，我们回顾了与中文生物医学文本挖掘有关的最新社区挑战的进展。

    arXiv:2403.04261v1 Announce Type: new  Abstract: The field of biomedical research has witnessed a significant increase in the accumulation of vast amounts of textual data from various sources such as scientific literatures, electronic health records, clinical trial reports, and social media. However, manually processing and analyzing these extensive and complex resources is time-consuming and inefficient. To address this challenge, biomedical text mining, also known as biomedical natural language processing, has garnered great attention. Community challenge evaluation competitions have played an important role in promoting technology innovation and interdisciplinary collaboration in biomedical text mining research. These challenges provide platforms for researchers to develop state-of-the-art solutions for data mining and information processing in biomedical research. In this article, we review the recent advances in community challenges specific to Chinese biomedical text mining. Firs
    
[^6]: SimuCourt: 利用真实司法判决文件构建司法决策代理

    SimuCourt: Building Judicial Decision-Making Agents with Real-world Judgement Documents

    [https://arxiv.org/abs/2403.02959](https://arxiv.org/abs/2403.02959)

    提出了SimuCourt司法基准，包括真实世界的司法文件，并引入了司法决策任务和多代理框架，评估了代理的司法分析和决策能力

    

    随着深度学习、自然语言处理技术的发展，有效提高了传统司法行业各个方面的效率。然而，目前大多数工作主要集中在个别司法阶段，忽视了跨阶段的协作。随着由大型语言模型提供支持的自主代理在现实环境中变得越来越智能，并能做出复杂决策，为司法智能提供了新的见解。本文介绍了SimuCourt，一个司法基准，包括来自真实世界的420份判决文件，涵盖了三种最常见类型的司法案例，以及一个新颖任务司法决策，用于评估代理的司法分析和决策能力。为了支持这一任务，我们构建了一个大规模司法知识库，JudicialKB，其中包含多种法律知识。我们提出了一种新颖的多代理框架，AgentsCourt

    arXiv:2403.02959v1 Announce Type: cross  Abstract: With the development of deep learning, natural language processing technology has effectively improved the efficiency of various aspects of the traditional judicial industry. However, most current efforts focus solely on individual judicial stage, overlooking cross-stage collaboration. As the autonomous agents powered by large language models are becoming increasingly smart and able to make complex decisions in real-world settings, offering new insights for judicial intelligence. In this paper, (1) we introduce SimuCourt, a judicial benchmark that encompasses 420 judgment documents from real-world, spanning the three most common types of judicial cases, and a novel task Judicial Decision-Making to evaluate the judicial analysis and decision-making power of agents. To support this task, we construct a large-scale judicial knowledge base, JudicialKB, with multiple legal knowledge. (2) we propose a novel multi-agent framework, AgentsCourt
    
[^7]: 语言模型与人类在关键语法结构上的判断一致性

    Language models align with human judgments on key grammatical constructions

    [https://arxiv.org/abs/2402.01676](https://arxiv.org/abs/2402.01676)

    本研究通过对比评估发现，大型语言模型（LLMs）在俘获人类行为方面的表现非常出色，不仅整体准确率高，而且能够捕捉到人类语言判断中的细微差异。

    

    大型语言模型（LLMs）是否具有类似人类的语言普遍性？Dentella等人（2023年；“DGL”）使用多个LLMs提示语法正确性问题，以获取80个英语句子的语法句子判断，得出LLMs存在“是”偏向和“不能区分语法和非语法句子”的结论。我们采用了既定的实践方法重新评估LLM的性能，并发现DGL的数据实际上证明了LLM如何准确捕捉人类行为。模型不仅整体上实现了高准确率，还捕捉到了人类语言判断的细微变化。

    Do Large Language Models (LLMs) make human-like linguistic generalizations? Dentella et al. (2023; "DGL") prompt several LLMs ("Is the following sentence grammatically correct in English?") to elicit grammaticality judgments of 80 English sentences, concluding that LLMs demonstrate a "yes-response bias" and a "failure to distinguish grammatical from ungrammatical sentences". We re-evaluate LLM performance using well-established practices and find that DGL's data in fact provide evidence for just how well LLMs capture human behaviors. Models not only achieve high accuracy overall, but also capture fine-grained variation in human linguistic judgments.
    
[^8]: 基于点云表示和扩散模型的晶体结构生成设计

    Generative Design of Crystal Structures by Point Cloud Representations and Diffusion Model. (arXiv:2401.13192v1 [cs.AI])

    [http://arxiv.org/abs/2401.13192](http://arxiv.org/abs/2401.13192)

    本研究提出了一种基于点云和扩散模型的晶体结构生成设计框架，并通过重建输入结构和生成全新材料的实验证明了其有效性和潜力。

    

    在材料设计中，高效地生成能量稳定的晶体结构一直是个挑战，主要是因为晶格中原子的巨大排列。为了促进稳定材料的发现，我们提出了一个用于生成可合成材料的框架，利用点云表示来编码复杂的结构信息。在这个框架的核心是引入扩散模型作为基础支柱。为了评估我们方法的有效性，我们使用它来重建训练数据集中的输入结构，并严格验证其高重建性能。此外，我们通过生成全新的材料，重点强调了基于点云的晶体扩散(PCCD)的巨大潜力，并展示了其可合成性。我们的研究在材料设计和合成的推进中，通过先进的生成设计方法，做出了显著贡献。

    Efficiently generating energetically stable crystal structures has long been a challenge in material design, primarily due to the immense arrangement of atoms in a crystal lattice. To facilitate the discovery of stable material, we present a framework for the generation of synthesizable materials, leveraging a point cloud representation to encode intricate structural information. At the heart of this framework lies the introduction of a diffusion model as its foundational pillar. To gauge the efficacy of our approach, we employ it to reconstruct input structures from our training datasets, rigorously validating its high reconstruction performance. Furthermore, we demonstrate the profound potential of Point Cloud-Based Crystal Diffusion (PCCD) by generating entirely new materials, emphasizing their synthesizability. Our research stands as a noteworthy contribution to the advancement of materials design and synthesis through the cutting-edge avenue of generative design instead of the con
    
[^9]: 通过DeepFool算法对深度神经网络进行有针对性的类别操纵的对抗攻击定制

    Tailoring Adversarial Attacks on Deep Neural Networks for Targeted Class Manipulation Using DeepFool Algorithm. (arXiv:2310.13019v1 [cs.CV])

    [http://arxiv.org/abs/2310.13019](http://arxiv.org/abs/2310.13019)

    本文提出了一种增强版DeepFool算法，名为Targeted DeepFool，可以针对特定类别进行错误分类，并引入了最小置信度分数要求超参数来提高灵活性。

    

    深度神经网络（DNNs）在各个领域都取得了显著的进展，但对抗攻击的易受攻击性引起了严重关注。了解这些易受攻击性并开发有效的防御机制至关重要。DeepFool是Moosavi-Dezfooli等人（2016年）提出的一种算法，用于找到将输入图像错误分类的最小扰动。然而，DeepFool缺乏有针对性的方法，使其在特定攻击场景中的有效性较低。此外，在先前的相关工作中，研究人员主要关注的是成功率，而没有考虑图像被扭曲的程度、图像质量的完整性以及错误分类的置信度水平。因此，在本文中，我们提出了Targeted DeepFool，这是DeepFool的增强版，可以针对特定类别进行错误分类。我们还引入了一个最小置信度分数要求超参数来增强灵活性。我们的实验证明了所提方法在不同情况下的有效性和效率。

    Deep neural networks (DNNs) have significantly advanced various domains, but their vulnerability to adversarial attacks poses serious concerns. Understanding these vulnerabilities and developing effective defense mechanisms is crucial. DeepFool, an algorithm proposed by Moosavi-Dezfooli et al. (2016), finds minimal perturbations to misclassify input images. However, DeepFool lacks a targeted approach, making it less effective in specific attack scenarios. Also, in previous related works, researchers primarily focus on success, not considering how much an image is getting distorted; the integrity of the image quality, and the confidence level to misclassifying. So, in this paper, we propose Targeted DeepFool, an augmented version of DeepFool that allows targeting specific classes for misclassification. We also introduce a minimum confidence score requirement hyperparameter to enhance flexibility. Our experiments demonstrate the effectiveness and efficiency of the proposed method across 
    
[^10]: DiffLoad:扩散模型中的负荷预测不确定性量化

    DiffLoad: Uncertainty Quantification in Load Forecasting with Diffusion Model. (arXiv:2306.01001v1 [cs.LG])

    [http://arxiv.org/abs/2306.01001](http://arxiv.org/abs/2306.01001)

    本文提出了一种扩散模型中的负荷预测不确定性量化方法，采用Seq2Seq网络结构来分离两种类型的不确定性并处理异常情况，不仅着眼于预测条件期望值。

    

    电力负荷预测对电力系统的决策制定，如机组投入和能源管理等具有重要意义。近年来，各种基于自监督神经网络的方法已经被应用于电力负荷预测，以提高预测准确性和捕捉不确定性。然而，大多数现有的方法是基于高斯似然方法的，它旨在在给定的协变量下准确估计分布期望值。这种方法很难适应存在分布偏移和异常值的时间数据。在本文中，我们提出了一种基于扩散的Seq2seq结构来估计本体不确定性，并使用鲁棒的加性柯西分布来估计物象不确定性。我们展示了我们的方法能够分离两种类型的不确定性并处理突变情况，而不是准确预测条件期望。

    Electrical load forecasting is of great significance for the decision makings in power systems, such as unit commitment and energy management. In recent years, various self-supervised neural network-based methods have been applied to electrical load forecasting to improve forecasting accuracy and capture uncertainties. However, most current methods are based on Gaussian likelihood methods, which aim to accurately estimate the distribution expectation under a given covariate. This kind of approach is difficult to adapt to situations where temporal data has a distribution shift and outliers. In this paper, we propose a diffusion-based Seq2seq structure to estimate epistemic uncertainty and use the robust additive Cauchy distribution to estimate aleatoric uncertainty. Rather than accurately forecasting conditional expectations, we demonstrate our method's ability in separating two types of uncertainties and dealing with the mutant scenarios.
    
[^11]: 经典规划中探索和开发的自适应平衡

    Scale-Adaptive Balancing of Exploration and Exploitation in Classical Planning. (arXiv:2305.09840v1 [cs.AI])

    [http://arxiv.org/abs/2305.09840](http://arxiv.org/abs/2305.09840)

    本文提出了一种MCTS/THTS算法GreedyUCT-Normal，该算法能够通过采用奖励变化的尺度处理不同尺度的分布，以在经典计划中平衡探索和开发。

    

    在游戏树搜索和自动化规划中，平衡探索和开发一直是一个重要的问题。然而，虽然这个问题在多臂赌博机（MAB）文献中已经被广泛分析，但规划社区在试图应用这些结果时取得的成功有限。我们展示了MAB文献更详细的理论理解有助于改进基于蒙特卡罗树搜索（MCTS）/基于试验的启发式树搜索（THTS）的现有规划算法。具体而言，THTS在一种临时方法中使用UCB1 MAB算法，因为在启发式搜索中UCB1理论上需要有界支持奖励分布的要求在经典规划中不被满足。核心问题在于UCB1缺乏对不同奖励尺度的自适应。我们提出了GreedyUCT-Normal，这是一种具有UCB1-Normal赌博机的MCTS/THTS算法，用于敏捷经典计划，它通过采用奖励变化的尺度处理不同尺度的分布。

    Balancing exploration and exploitation has been an important problem in both game tree search and automated planning. However, while the problem has been extensively analyzed within the Multi-Armed Bandit (MAB) literature, the planning community has had limited success when attempting to apply those results. We show that a more detailed theoretical understanding of MAB literature helps improve existing planning algorithms that are based on Monte Carlo Tree Search (MCTS) / Trial Based Heuristic Tree Search (THTS). In particular, THTS uses UCB1 MAB algorithms in an ad hoc manner, as UCB1's theoretical requirement of fixed bounded support reward distributions is not satisfied within heuristic search for classical planning. The core issue lies in UCB1's lack of adaptations to the different scales of the rewards. We propose GreedyUCT-Normal, a MCTS/THTS algorithm with UCB1-Normal bandit for agile classical planning, which handles distributions with different scales by taking the reward vari
    
[^12]: 将未标记数据纳入贝叶斯神经网络中

    Incorporating Unlabelled Data into Bayesian Neural Networks. (arXiv:2304.01762v1 [cs.LG])

    [http://arxiv.org/abs/2304.01762](http://arxiv.org/abs/2304.01762)

    该论文提出了一种利用未标记数据学习贝叶斯神经网络（BNNs）的对比框架，通过该框架提出了一种同时具备自监督学习的标签效率和贝叶斯方法中的不确定性估计的实用BNN算法。最后，该方法在半监督和低预算主动学习问题中展现出了数据高效学习的优势。

    

    我们提出了一个对贝叶斯神经网络（BNNs）中先验分布进行学习的对比框架，利用未标记数据来优化。基于该框架，我们提出了一种实用的BNN算法，同时具备自监督学习的标签效率和贝叶斯方法中的根据原则的不确定性估计。最后，我们展示了我们的方法在半监督和低预算主动学习问题中的数据高效学习优势。

    We develop a contrastive framework for learning better prior distributions for Bayesian Neural Networks (BNNs) using unlabelled data. With this framework, we propose a practical BNN algorithm that offers the label-efficiency of self-supervised learning and the principled uncertainty estimates of Bayesian methods. Finally, we demonstrate the advantages of our approach for data-efficient learning in semi-supervised and low-budget active learning problems.
    
[^13]: 启发式多智能体规划优化智能体协作

    Optimizing Agent Collaboration through Heuristic Multi-Agent Planning. (arXiv:2301.01246v3 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2301.01246](http://arxiv.org/abs/2301.01246)

    提出了一种启发式多智能体规划算法，解决了涉及不同类型智能体的问题，比现有算法表现更好。

    

    针对涉及到不同类型感知智能体的问题，目前解决QDec-POMDP的SOTA算法QDec-FP和QDec-FPS无法有效解决。本文提出了一种新算法，通过要求智能体采取相同的计划，以解决这个问题。在这些情况下，我们的算法比QDec-FP和QDec-FPS都表现更好。

    The SOTA algorithms for addressing QDec-POMDP issues, QDec-FP and QDec-FPS, are unable to effectively tackle problems that involve different types of sensing agents. We propose a new algorithm that addresses this issue by requiring agents to adopt the same plan if one agent is unable to take a sensing action but the other can. Our algorithm performs significantly better than both QDec-FP and QDec-FPS in these types of situations.
    
[^14]: 可扩展的多智能体强化学习在仓库物流中与机器人和人类同事合作

    Scalable Multi-Agent Reinforcement Learning for Warehouse Logistics with Robotic and Human Co-Workers. (arXiv:2212.11498v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.11498](http://arxiv.org/abs/2212.11498)

    该论文提出了一种可扩展的多智能体强化学习方法，用于仓库物流中的机器人和人类同事合作。他们通过分层的MARL算法，让经理和工人代理根据全局目标进行协同训练，以最大化拣货速率。

    

    我们设想一个仓库里有数十个移动机器人和人类分拣员一起工作，收集和交付仓库内的物品。我们要解决的基本问题是称为拣货问题，即这些工作代理人如何在仓库中协调他们的移动和行为以最大化性能（例如订单吞吐量）。传统的行业方法使用启发式方法需要大量的工程努力来为固有可变的仓库配置进行优化。相比之下，多智能体强化学习（MARL）可以灵活地应用于不同的仓库配置（例如大小，布局，工人数量/类型，物品补充频率），因为代理人通过经验学习如何最优地相互合作。我们开发了分层MARL算法，其中一个管理者为工人代理分配目标，并且管理者和工人的策略被共同训练以最大化全局目标（例如拣货速率）。

    We envision a warehouse in which dozens of mobile robots and human pickers work together to collect and deliver items within the warehouse. The fundamental problem we tackle, called the order-picking problem, is how these worker agents must coordinate their movement and actions in the warehouse to maximise performance (e.g. order throughput). Established industry methods using heuristic approaches require large engineering efforts to optimise for innately variable warehouse configurations. In contrast, multi-agent reinforcement learning (MARL) can be flexibly applied to diverse warehouse configurations (e.g. size, layout, number/types of workers, item replenishment frequency), as the agents learn through experience how to optimally cooperate with one another. We develop hierarchical MARL algorithms in which a manager assigns goals to worker agents, and the policies of the manager and workers are co-trained toward maximising a global objective (e.g. pick rate). Our hierarchical algorith
    
[^15]: CLIP是否捆绑概念？探索大型图像模型的组合性。

    Does CLIP Bind Concepts? Probing Compositionality in Large Image Models. (arXiv:2212.10537v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2212.10537](http://arxiv.org/abs/2212.10537)

    本文分析了大型神经网络模型CLIP的组合性能力以及以结构敏感的方式捆绑变量的能力，发现其能够在单一对象的情况下组合概念，但在需要概念捆绑的情况下性能显著下降。

    

    近年来，结合文本和图像的大型神经网络模型取得了令人瞩目的进展。然而，这些模型在多大程度上编码了它们操作的概念的组成性表示，如通过对“红色立方体”进行推理以正确识别“红色”和“立方体”这些成分，这仍然是一个开放性问题。本文关注一个大型预训练的视觉和语言模型（CLIP）编码组合概念的能力以及以结构敏感的方式捆绑变量的能力（例如区分“立方体在球体后面”和“球体在立方体后面”）。为了检查CLIP的性能，我们比较了许多来自组合分布语义模型（CDSMs）的架构，这是一种试图在嵌入空间中实现传统组合语言结构的研究方向。我们发现CLIP能够在单一对象的情况下组合概念，但在需要概念捆绑的情况下性能显著下降。我们的分析凸显了评估大型模型组合性的重要性，并为未来研究提出了方向。

    Large-scale neural network models combining text and images have made incredible progress in recent years. However, it remains an open question to what extent such models encode compositional representations of the concepts over which they operate, such as correctly identifying ''red cube'' by reasoning over the constituents ''red'' and ''cube''. In this work, we focus on the ability of a large pretrained vision and language model (CLIP) to encode compositional concepts and to bind variables in a structure-sensitive way (e.g., differentiating ''cube behind sphere'' from ''sphere behind cube''). In order to inspect the performance of CLIP, we compare several architectures from research on compositional distributional semantics models (CDSMs), a line of research that attempts to implement traditional compositional linguistic structures within embedding spaces. We find that CLIP can compose concepts in a single-object setting, but in situations where concept binding is needed, performance
    


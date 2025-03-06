# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient Domain Adaptation for Endoscopic Visual Odometry](https://arxiv.org/abs/2403.10860) | 这项工作提出了一个高效的内窥镜视觉里程计神经风格迁移框架，将从术前规划到测试阶段的时间缩短至不到五分钟，通过利用有限数量的真实图像和术前先验信息进行训练，以及引入测试时间自适应方法来减小训练和测试之间的光照条件差距。 |
| [^2] | [A privacy-preserving, distributed and cooperative FCM-based learning approach for Cancer Research](https://arxiv.org/abs/2402.10102) | 本论文提出了一种隐私保护的、分布式的、合作的基于FCM的学习方法，用于癌症研究，通过联邦学习过程改进了模型的性能。 |
| [^3] | [Multimodal Action Quality Assessment](https://arxiv.org/abs/2402.09444) | 该论文提出了一个名为PAMFN的渐进自适应多模态融合网络，用于多模态动作质量评估。该模型利用RGB、光流和音频信息，分别建模模态特定信息和混合模态信息，并通过充分利用音频信息，提高了评分回归的准确性。 |
| [^4] | [Improved Performances and Motivation in Intelligent Tutoring Systems: Combining Machine Learning and Learner Choice](https://arxiv.org/abs/2402.01669) | 本研究通过结合机器学习和学生选择，改进了智能辅导系统的性能和动机。使用ZPDES算法，该系统能够最大化学习进展，并在实地研究中提高了不同学生群体的学习成绩。研究还探讨了学生选择对学习效率和动机的影响。 |
| [^5] | [ExpertPrompting: Instructing Large Language Models to be Distinguished Experts.](http://arxiv.org/abs/2305.14688) | 本论文提出了“专家提示”技术，用于训练大型语言模型成为杰出的专家。该方法使用上下文学习自动生成每个指令的详细和定制的专家身份描述，并要求模型根据这些提示提供答案。基于这种技术，本文提出了一个新的开源聊天助手ExpertLLaMA，该助手在评估中表现出高质量的数据和96％的ChatGPT能力。 |

# 详细

[^1]: 内窥镜视觉里程计的高效领域自适应

    Efficient Domain Adaptation for Endoscopic Visual Odometry

    [https://arxiv.org/abs/2403.10860](https://arxiv.org/abs/2403.10860)

    这项工作提出了一个高效的内窥镜视觉里程计神经风格迁移框架，将从术前规划到测试阶段的时间缩短至不到五分钟，通过利用有限数量的真实图像和术前先验信息进行训练，以及引入测试时间自适应方法来减小训练和测试之间的光照条件差距。

    

    arXiv:2403.10860v1 公告类型: 跨领域 摘要: 视觉里程计在内窥镜成像中起着至关重要的作用，然而缺乏具有地面真实性的图像对于学习里程计信息提出了重大挑战。因此，领域自适应为连接术前规划领域和术中实际领域学习里程计信息提供了一种有前途的方法。然而，现有方法在训练时间上存在低效性。本文提出了一种针对内窥镜视觉里程计的高效神经风格迁移框架，将从术前规划到测试阶段的时间缩短至不到五分钟。为了进行高效训练，本研究专注于用有限数量的真实图像训练模块，并利用术前先验信息大大减少训练时间。此外，在测试阶段，我们提出了一种新颖的测试时间自适应（TTA）方法来消除训练和测试之间的光照条件差距。

    arXiv:2403.10860v1 Announce Type: cross  Abstract: Visual odometry plays a crucial role in endoscopic imaging, yet the scarcity of realistic images with ground truth poses poses a significant challenge. Therefore, domain adaptation offers a promising approach to bridge the pre-operative planning domain with the intra-operative real domain for learning odometry information. However, existing methodologies suffer from inefficiencies in the training time. In this work, an efficient neural style transfer framework for endoscopic visual odometry is proposed, which compresses the time from pre-operative planning to testing phase to less than five minutes. For efficient traing, this work focuses on training modules with only a limited number of real images and we exploit pre-operative prior information to dramatically reduce training duration. Moreover, during the testing phase, we propose a novel Test Time Adaptation (TTA) method to mitigate the gap in lighting conditions between training an
    
[^2]: 一个隐私保护的、分布式的、合作的基于FCM的癌症研究学习方法

    A privacy-preserving, distributed and cooperative FCM-based learning approach for Cancer Research

    [https://arxiv.org/abs/2402.10102](https://arxiv.org/abs/2402.10102)

    本论文提出了一种隐私保护的、分布式的、合作的基于FCM的学习方法，用于癌症研究，通过联邦学习过程改进了模型的性能。

    

    分布式人工智能引起了越来越多的关注。本文介绍了一种创新的隐私保护分布式学习方法，基于粒子群优化的模糊认知图。作者设计了一种协作FCM学习的训练方案，提供了符合当前规定的数据隐私保护。该方法应用于癌症检测问题，证明了联邦学习过程改进了模型的性能，并获得了类似于文献中的结果。

    arXiv:2402.10102v1 Announce Type: new  Abstract: Distributed Artificial Intelligence is attracting interest day by day. In this paper, the authors introduce an innovative methodology for distributed learning of Particle Swarm Optimization-based Fuzzy Cognitive Maps in a privacy-preserving way. The authors design a training scheme for collaborative FCM learning that offers data privacy compliant with the current regulation. This method is applied to a cancer detection problem, proving that the performance of the model is improved by the Federated Learning process, and obtaining similar results to the ones that can be found in the literature.
    
[^3]: 多模态动作质量评估

    Multimodal Action Quality Assessment

    [https://arxiv.org/abs/2402.09444](https://arxiv.org/abs/2402.09444)

    该论文提出了一个名为PAMFN的渐进自适应多模态融合网络，用于多模态动作质量评估。该模型利用RGB、光流和音频信息，分别建模模态特定信息和混合模态信息，并通过充分利用音频信息，提高了评分回归的准确性。

    

    行动质量评估（AQA）是评估动作执行情况的方法。以往的研究仅利用视觉信息进行建模，忽视了音频信息。我们认为，虽然AQA高度依赖视觉信息，但音频也是提高评分回归准确性的有用补充信息，特别是在具有背景音乐的运动项目中，如花样滑冰和韵律体操。为了利用多模态信息进行AQA，即RGB、光流和音频信息，我们提出了一个渐进自适应多模态融合网络（PAMFN），它分别对模态特定信息和混合模态信息进行建模。我们的模型由三个模态特定分支和一个混合模态分支组成，独立地探索模态特定信息，并渐进地聚合来自模态特定分支的模态特定信息。

    arXiv:2402.09444v1 Announce Type: cross  Abstract: Action quality assessment (AQA) is to assess how well an action is performed. Previous works perform modelling by only the use of visual information, ignoring audio information. We argue that although AQA is highly dependent on visual information, the audio is useful complementary information for improving the score regression accuracy, especially for sports with background music, such as figure skating and rhythmic gymnastics. To leverage multimodal information for AQA, i.e., RGB, optical flow and audio information, we propose a Progressive Adaptive Multimodal Fusion Network (PAMFN) that separately models modality-specific information and mixed-modality information. Our model consists of with three modality-specific branches that independently explore modality-specific information and a mixed-modality branch that progressively aggregates the modality-specific information from the modality-specific branches. To build the bridge between
    
[^4]: 智能辅导系统中的性能和动机的改进：结合机器学习和学习者选择

    Improved Performances and Motivation in Intelligent Tutoring Systems: Combining Machine Learning and Learner Choice

    [https://arxiv.org/abs/2402.01669](https://arxiv.org/abs/2402.01669)

    本研究通过结合机器学习和学生选择，改进了智能辅导系统的性能和动机。使用ZPDES算法，该系统能够最大化学习进展，并在实地研究中提高了不同学生群体的学习成绩。研究还探讨了学生选择对学习效率和动机的影响。

    

    在学校中，大规模的课堂规模给个性化学习带来了挑战，教育技术，尤其是智能辅导系统（ITS）试图解决这个问题。在这个背景下，基于学习进展假设（LPH）和多臂赌博机器学习技术的ZPDES算法对最大化学习进展（LP）的练习进行排序。该算法在之前的实地研究中已经显示出将学习表现提升到更广泛的学生群体中，与手工设计的课程相比。然而，其动机影响尚未评估。此外，ZPDES不允许学生发表选择意见。这种缺乏机构的限制与关注建模好奇驱动学习的LPH理论不一致。我们在这里研究了这种选择可能性的引入如何影响学习效率和动机。给定的选择与练习难度正交的维度有关，作为一种有趣的特性。

    Large class sizes pose challenges to personalized learning in schools, which educational technologies, especially intelligent tutoring systems (ITS), aim to address. In this context, the ZPDES algorithm, based on the Learning Progress Hypothesis (LPH) and multi-armed bandit machine learning techniques, sequences exercises that maximize learning progress (LP). This algorithm was previously shown in field studies to boost learning performances for a wider diversity of students compared to a hand-designed curriculum. However, its motivational impact was not assessed. Also, ZPDES did not allow students to express choices. This limitation in agency is at odds with the LPH theory concerned with modeling curiosity-driven learning. We here study how the introduction of such choice possibilities impact both learning efficiency and motivation. The given choice concerns dimensions that are orthogonal to exercise difficulty, acting as a playful feature.   In an extensive field study (265 7-8 years
    
[^5]: 专家提示：指导大型语言模型成为杰出的专家

    ExpertPrompting: Instructing Large Language Models to be Distinguished Experts. (arXiv:2305.14688v1 [cs.CL])

    [http://arxiv.org/abs/2305.14688](http://arxiv.org/abs/2305.14688)

    本论文提出了“专家提示”技术，用于训练大型语言模型成为杰出的专家。该方法使用上下文学习自动生成每个指令的详细和定制的专家身份描述，并要求模型根据这些提示提供答案。基于这种技术，本文提出了一个新的开源聊天助手ExpertLLaMA，该助手在评估中表现出高质量的数据和96％的ChatGPT能力。

    

    如果以适当的提示方式进行处理，对齐的大型语言模型（LLM）的回答质量可以大大提高。在本文中，我们提出了专家提示，以引发LLMs作为杰出专家回答的潜力。我们首先利用上下文学习自动生成每个特定指令的详细和定制的专家身份描述，然后要求LLMs根据这种代理人背景提供答案。基于这种增强的提示策略，我们使用GPT-3.5生成了一组新的指令遵循数据，并训练了一个竞争性的开源聊天助手ExpertLLaMA。我们采用基于GPT4的评估显示：1）专家数据的质量显著高于普通答案，2）ExpertLLaMA胜过现有的开源对手，实现了ChatGPT能力的96％。所有数据和ExpertLLaMA模型将在\url{https://github.com/OFA-Sys/Exp}上公开。

    The answering quality of an aligned large language model (LLM) can be drastically improved if treated with proper crafting of prompts. In this paper, we propose ExpertPrompting to elicit the potential of LLMs to answer as distinguished experts. We first utilize In-Context Learning to automatically synthesize detailed and customized descriptions of the expert identity for each specific instruction, and then ask LLMs to provide answer conditioned on such agent background. Based on this augmented prompting strategy, we produce a new set of instruction-following data using GPT-3.5, and train a competitive open-source chat assistant called ExpertLLaMA. We employ GPT4-based evaluation to show that 1) the expert data is of significantly higher quality than vanilla answers, and 2) ExpertLLaMA outperforms existing open-source opponents and achieves 96\% of the original ChatGPT's capability. All data and the ExpertLLaMA model will be made publicly available at \url{https://github.com/OFA-Sys/Exp
    


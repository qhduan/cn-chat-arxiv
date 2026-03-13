# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Partially Recentralization Softmax Loss for Vision-Language Models Robustness](https://arxiv.org/abs/2402.03627) | 本文研究了通过修改预训练多模态模型的损失函数来提高对抗鲁棒性，通过限制前K个softmax输出。实验结果表明，经过微调后，模型的对抗鲁棒性显著提高，能够有效抵御常见的攻击。 |
| [^2] | [A Multi-Perspective Machine Learning Approach to Evaluate Police-Driver Interaction in Los Angeles](https://arxiv.org/abs/2402.01703) | 该研究提出了一种多角度的机器学习方法，用于分析洛杉矶警察与司机的互动。该方法利用多模态的数据包括音频、视频和文字信息，旨在提供对复杂和有争议的警民互动的分析工具。 |
| [^3] | [Domain-Independent Dynamic Programming.](http://arxiv.org/abs/2401.13883) | 本文提出了一种领域无关的动态规划方法，并介绍了基于状态转移系统的动态规划描述语言。实验证明，该方法在许多组合优化问题上优于传统的混合整数规划和约束规划方法。 |

# 详细

[^1]: 近似的中心化softmax损失用于视觉-语言模型的鲁棒性

    Partially Recentralization Softmax Loss for Vision-Language Models Robustness

    [https://arxiv.org/abs/2402.03627](https://arxiv.org/abs/2402.03627)

    本文研究了通过修改预训练多模态模型的损失函数来提高对抗鲁棒性，通过限制前K个softmax输出。实验结果表明，经过微调后，模型的对抗鲁棒性显著提高，能够有效抵御常见的攻击。

    

    随着大型语言模型在自然语言处理任务中的突破，多模态技术变得非常流行。然而，已经证明多模态自然语言处理模型容易受到对抗攻击，即模型的输出可以通过对输入进行微小扰动而发生巨大变化。虽然计算机视觉和自然语言处理模型中已经提出了几种防御技术，但对多模态模型的鲁棒性还没有进行充分探索。在本文中，我们研究了通过修改预训练多模态模型的损失函数，通过限制前K个softmax输出来提供的对抗鲁棒性。基于评估和评分，我们的实验结果显示，在经过微调后，预训练模型的对抗鲁棒性可以显着提高，对抗常见的攻击有效。进一步的研究应该探索这类损失函数的输出多样性、泛化能力以及鲁棒性和性能之间的平衡。我们的代码将在之后提供。

    As Large Language Models make a breakthrough in natural language processing tasks (NLP), multimodal technique becomes extremely popular. However, it has been shown that multimodal NLP are vulnerable to adversarial attacks, where the outputs of a model can be dramatically changed by a perturbation to the input. While several defense techniques have been proposed both in computer vision and NLP models, the multimodal robustness of models have not been fully explored. In this paper, we study the adversarial robustness provided by modifying loss function of pre-trained multimodal models, by restricting top K softmax outputs. Based on the evaluation and scoring, our experiments show that after a fine-tuning, adversarial robustness of pre-trained models can be significantly improved, against popular attacks. Further research should be studying, such as output diversity, generalization and the robustness-performance trade-off of this kind of loss functions. Our code will be available after th
    
[^2]: 一种多角度的机器学习方法用于评估洛杉矶警察与司机的互动

    A Multi-Perspective Machine Learning Approach to Evaluate Police-Driver Interaction in Los Angeles

    [https://arxiv.org/abs/2402.01703](https://arxiv.org/abs/2402.01703)

    该研究提出了一种多角度的机器学习方法，用于分析洛杉矶警察与司机的互动。该方法利用多模态的数据包括音频、视频和文字信息，旨在提供对复杂和有争议的警民互动的分析工具。

    

    政府官员与市民之间的互动影响公共福祉和民主社会的正当性。警察是国家最显而易见、最接触市民的代理人，在交通站停期间，他们每年与公众互动超过2000万次。如今，这些互动经常被戴在身上的摄像机记录下来，这被视为提高警察问责制和改善警民互动的手段。然而，由于缺乏可靠的自动化工具来分析这些复杂而有争议的警民互动，这些记录的及时分析受到了阻碍。本文提出了一种新的多角度、多模态机器学习（ML）工具的方法，用于分析来自这些身上摄像机记录的音频、视频和文字信息。我们的方法首先确定与不同利益相关者最相关的沟通方面，包括共同感知互动的标志标记以及具有这些标记的符号。

    Interactions between the government officials and civilians affect public wellbeing and the state legitimacy that is necessary for the functioning of democratic society. Police officers, the most visible and contacted agents of the state, interact with the public more than 20 million times a year during traffic stops. Today, these interactions are regularly recorded by body-worn cameras (BWCs), which are lauded as a means to enhance police accountability and improve police-public interactions. However, the timely analysis of these recordings is hampered by a lack of reliable automated tools that can enable the analysis of these complex and contested police-public interactions. This article proposes an approach to developing new multi-perspective, multimodal machine learning (ML) tools to analyze the audio, video, and transcript information from this BWC footage. Our approach begins by identifying the aspects of communication most salient to different stakeholders, including both commun
    
[^3]: 领域无关的动态规划方法

    Domain-Independent Dynamic Programming. (arXiv:2401.13883v1 [cs.AI])

    [http://arxiv.org/abs/2401.13883](http://arxiv.org/abs/2401.13883)

    本文提出了一种领域无关的动态规划方法，并介绍了基于状态转移系统的动态规划描述语言。实验证明，该方法在许多组合优化问题上优于传统的混合整数规划和约束规划方法。

    

    对于组合优化问题，基于模型的范例如混合整数规划 (MIP) 和约束规划 (CP) 旨在解耦问题的建模和求解过程，这是声明性问题求解的“圣杯”。我们提出了领域无关的动态规划（DIDP），这是一种基于动态规划 (DP) 的新的基于模型的方法。虽然DP并不新鲜，但通常它被作为一种特定问题的方法来实现。我们引入了动态规划描述语言 (DyPDL)，一种基于状态转移系统的形式化语言，灵感来自于AI规划。我们展示了启发式搜索算法可以用来求解DyPDL模型，并提出了七种DIDP求解器。我们在常见的11个组合优化问题类别的基准实例上，将我们的DIDP求解器与商业MIP和CP求解器进行了实验比较（分别求解MIP和CP模型）。结果显示DIDP在九个问题类别中优于MIP，也优于CP在九个问题类别中。

    For combinatorial optimization problems, model-based paradigms such as mixed-integer programming (MIP) and constraint programming (CP) aim to decouple modeling and solving a problem: the `holy grail' of declarative problem solving. We propose domain-independent dynamic programming (DIDP), a new model-based paradigm based on dynamic programming (DP). While DP is not new, it has typically been implemented as a problem-specific method. We introduce Dynamic Programming Description Language (DyPDL), a formalism to define DP models based on a state transition system, inspired by AI planning. We show that heuristic search algorithms can be used to solve DyPDL models and propose seven DIDP solvers. We experimentally compare our DIDP solvers with commercial MIP and CP solvers (solving MIP and CP models, respectively) on common benchmark instances of eleven combinatorial optimization problem classes. We show that DIDP outperforms MIP in nine problem classes, CP also in nine problem classes, and 
    


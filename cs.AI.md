# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MacroSwarm: A Field-based Compositional Framework for Swarm Programming.](http://arxiv.org/abs/2401.10969) | MacroSwarm是一种基于场的群体编程框架，通过可组合的功能模块实现复杂的群体行为，通过将感知场映射为执行目标场，提供了一种系统化的设计和实现群体行为的方法。 |
| [^2] | [Food Classification using Joint Representation of Visual and Textual Data.](http://arxiv.org/abs/2308.02562) | 本研究提出了一种使用联合表示的多模态分类框架，通过修改版的EfficientNet和Mish激活函数实现图像分类，使用基于BERT的网络实现文本分类。实验结果表明，所提出的网络在图像和文本分类上表现优于其他方法，准确率提高了11.57%和6.34%。比较分析还证明了所提出方法的效率和鲁棒性。 |
| [^3] | [You Can Generate It Again: Data-to-text Generation with Verification and Correction Prompting.](http://arxiv.org/abs/2306.15933) | 本文提出了一种多步骤生成、验证和纠正的数据生成文本方法，通过专门的错误指示提示来改善输出质量。 |
| [^4] | [Impartial Games: A Challenge for Reinforcement Learning.](http://arxiv.org/abs/2205.12787) | AlphaZero-style reinforcement learning algorithms excel in various board games but face challenges with impartial games. The researchers present a concrete example of the game nim, and show that AlphaZero-style algorithms have difficulty learning these impartial games on larger board sizes. The difference between impartial games and partisan games can be explained by the vulnerability to adversarial attacks and perturbations. |

# 详细

[^1]: MacroSwarm: 一种基于场的组合框架用于群体编程

    MacroSwarm: A Field-based Compositional Framework for Swarm Programming. (arXiv:2401.10969v1 [cs.AI])

    [http://arxiv.org/abs/2401.10969](http://arxiv.org/abs/2401.10969)

    MacroSwarm是一种基于场的群体编程框架，通过可组合的功能模块实现复杂的群体行为，通过将感知场映射为执行目标场，提供了一种系统化的设计和实现群体行为的方法。

    

    群体行为工程是一项旨在研究协调简单智能体团体内计算和行动的方法和技术，以实现复杂的全局目标，如图案形成、集体移动、聚类和分布式感知。尽管在群体（无人机、机器人、车辆）分析和工程方面取得了一些进展，但仍然需要通用的设计和实现方法和工具，以系统化的方式定义复杂的群体行为。为了对此做出贡献，本文提出了一种新的基于场的协调方法，称为MacroSwarm，以可重用且完全可组合的功能模块为基础，嵌入集体计算和协调。基于集成计算的宏编程范式，MacroSwarm提出了将每个群体行为块表示为将感知场映射为执行目标场的纯函数的思路。

    Swarm behaviour engineering is an area of research that seeks to investigate methods and techniques for coordinating computation and action within groups of simple agents to achieve complex global goals like pattern formation, collective movement, clustering, and distributed sensing. Despite recent progress in the analysis and engineering of swarms (of drones, robots, vehicles), there is still a need for general design and implementation methods and tools that can be used to define complex swarm behaviour in a principled way. To contribute to this quest, this article proposes a new field-based coordination approach, called MacroSwarm, to design and program swarm behaviour in terms of reusable and fully composable functional blocks embedding collective computation and coordination. Based on the macroprogramming paradigm of aggregate computing, MacroSwarm builds on the idea of expressing each swarm behaviour block as a pure function mapping sensing fields into actuation goal fields, e.g.
    
[^2]: 使用视觉和文本数据的联合表示进行食物分类

    Food Classification using Joint Representation of Visual and Textual Data. (arXiv:2308.02562v1 [cs.CV])

    [http://arxiv.org/abs/2308.02562](http://arxiv.org/abs/2308.02562)

    本研究提出了一种使用联合表示的多模态分类框架，通过修改版的EfficientNet和Mish激活函数实现图像分类，使用基于BERT的网络实现文本分类。实验结果表明，所提出的网络在图像和文本分类上表现优于其他方法，准确率提高了11.57%和6.34%。比较分析还证明了所提出方法的效率和鲁棒性。

    

    食物分类是健康保健中的重要任务。在这项工作中，我们提出了一个多模态分类框架，该框架使用了修改版的EfficientNet和Mish激活函数用于图像分类，同时使用传统的基于BERT的网络进行文本分类。我们在一个大型开源数据集UPMC Food-101上评估了所提出的网络和其他最先进的方法。实验结果显示，所提出的网络在图像和文本分类上的准确率分别比第二最好的方法提高了11.57%和6.34%。我们还比较了使用机器学习和深度学习模型进行文本分类的准确率、精确率和召回率。通过对图像和文本的预测结果进行比较分析，证明了所提出方法的效率和鲁棒性。

    Food classification is an important task in health care. In this work, we propose a multimodal classification framework that uses the modified version of EfficientNet with the Mish activation function for image classification, and the traditional BERT transformer-based network is used for text classification. The proposed network and the other state-of-the-art methods are evaluated on a large open-source dataset, UPMC Food-101. The experimental results show that the proposed network outperforms the other methods, a significant difference of 11.57% and 6.34% in accuracy is observed for image and text classification, respectively, when compared with the second-best performing method. We also compared the performance in terms of accuracy, precision, and recall for text classification using both machine learning and deep learning-based models. The comparative analysis from the prediction results of both images and text demonstrated the efficiency and robustness of the proposed approach.
    
[^3]: 通过验证和纠正提示进行数据生成文本生成

    You Can Generate It Again: Data-to-text Generation with Verification and Correction Prompting. (arXiv:2306.15933v1 [cs.CL])

    [http://arxiv.org/abs/2306.15933](http://arxiv.org/abs/2306.15933)

    本文提出了一种多步骤生成、验证和纠正的数据生成文本方法，通过专门的错误指示提示来改善输出质量。

    

    尽管现有模型取得了显著进展，从结构化数据输入生成文本描述（称为数据生成文本）仍然是一个具有挑战性的任务。在本文中，我们提出了一种新的方法，通过引入包括生成、验证和纠正阶段的多步骤过程，超越了传统的一次性生成方法。我们的方法，VCP（验证和纠正提示），从模型生成初始输出开始。然后，我们继续验证所生成文本的不同方面的正确性。验证步骤的观察结果被转化为专门的错误指示提示，该提示指示模型在重新生成输出时考虑已识别的错误。为了增强模型的纠正能力，我们开发了一个经过精心设计的培训过程。该过程使模型能够融入错误指示提示的反馈，从而改善输出生成。

    Despite significant advancements in existing models, generating text descriptions from structured data input, known as data-to-text generation, remains a challenging task. In this paper, we propose a novel approach that goes beyond traditional one-shot generation methods by introducing a multi-step process consisting of generation, verification, and correction stages. Our approach, VCP(Verification and Correction Prompting), begins with the model generating an initial output. We then proceed to verify the correctness of different aspects of the generated text. The observations from the verification step are converted into a specialized error-indication prompt, which instructs the model to regenerate the output while considering the identified errors. To enhance the model's correction ability, we have developed a carefully designed training procedure. This procedure enables the model to incorporate feedback from the error-indication prompt, resulting in improved output generation. Throu
    
[^4]: 公正游戏：对强化学习的挑战

    Impartial Games: A Challenge for Reinforcement Learning. (arXiv:2205.12787v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2205.12787](http://arxiv.org/abs/2205.12787)

    AlphaZero-style reinforcement learning algorithms excel in various board games but face challenges with impartial games. The researchers present a concrete example of the game nim, and show that AlphaZero-style algorithms have difficulty learning these impartial games on larger board sizes. The difference between impartial games and partisan games can be explained by the vulnerability to adversarial attacks and perturbations.

    

    类似AlphaZero的强化学习算法在各种棋盘游戏中表现出色，但在公正游戏中却面临挑战，这些游戏中玩家共享棋子。我们提供了一个具体的游戏例子，即小孩们玩的尼姆游戏，以及其他一些公正游戏，这些游戏似乎成为AlphaZero和类似的强化学习算法的绊脚石。我们的发现与最近的研究一致，表明AlphaZero-style算法容易受到敌对攻击和敌对扰动的影响，显示了在所有合法状态下学习掌握这些游戏的困难。我们发现尼姆游戏在小型棋盘上可以学习，但当棋盘尺寸增大时，AlphaZero-style算法的学习速度显著减慢。直观上，尼姆等公正游戏与象棋和围棋等党派游戏之间的区别在于，如果系统中添加了微小的噪音（例如，棋盘的一小部分被覆盖），对于公正游戏来说，这是一种典型的情况。

    AlphaZero-style reinforcement learning (RL) algorithms excel in various board games but face challenges with impartial games, where players share pieces. We present a concrete example of a game - namely the children's game of nim - and other impartial games that seem to be a stumbling block for AlphaZero-style and similar reinforcement learning algorithms.  Our findings are consistent with recent studies showing that AlphaZero-style algorithms are vulnerable to adversarial attacks and adversarial perturbations, showing the difficulty of learning to master the games in all legal states.  We show that nim can be learned on small boards, but AlphaZero-style algorithms learning dramatically slows down when the board size increases. Intuitively, the difference between impartial games like nim and partisan games like Chess and Go can be explained by the fact that if a tiny amount of noise is added to the system (e.g. if a small part of the board is covered), for impartial games, it is typica
    


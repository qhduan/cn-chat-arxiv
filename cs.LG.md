# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enhancing Sequential Model Performance with Squared Sigmoid TanH (SST) Activation Under Data Constraints](https://arxiv.org/abs/2402.09034) | 该论文提出了一种平方Sigmoid TanH（SST）激活函数，用于增强在数据限制下的顺序模型学习能力。通过数学平方放大强激活和弱激活之间的差异，改善梯度流和信息过滤。在多个应用中评估了SST驱动的LSTM和GRU模型的性能。 |
| [^2] | [APALU: A Trainable, Adaptive Activation Function for Deep Learning Networks](https://arxiv.org/abs/2402.08244) | APALU 是一种可训练的激活函数，通过增强深度学习的学习性能，在适应复杂数据表示的同时保持稳定和高效。 在图像分类任务中，APALU相对于传统激活函数能够显著提高准确性。 |
| [^3] | [Knolling Bot: Learning Robotic Object Arrangement from Tidy Demonstrations](https://arxiv.org/abs/2310.04566) | 本论文介绍了一种自监督学习框架，利用Transformer神经网络使机器人能够从整齐排列的示范中理解和复制整洁的概念，从而实现整理物品的功能。 |
| [^4] | [Hybrid-Task Meta-Learning: A Graph Neural Network Approach for Scalable and Transferable Bandwidth Allocation.](http://arxiv.org/abs/2401.10253) | 本文提出了一种基于图神经网络的混合任务元学习算法，用于可扩展和可转移的带宽分配。通过引入GNN和HML算法，该方法在不同的通信场景下具有较好的性能和采样效率。 |
| [^5] | [LPAC: Learnable Perception-Action-Communication Loops with Applications to Coverage Control.](http://arxiv.org/abs/2401.04855) | 提出了一种可学习的感知-行动-通信(LPAC)架构，使用卷积神经网络处理环境感知，图神经网络实现机器人之间的信息交流，浅层多层感知机计算机器人的动作。使用集中式显微算法训练模型，实现机器人群体的协作。 |
| [^6] | [On the Variance, Admissibility, and Stability of Empirical Risk Minimization.](http://arxiv.org/abs/2305.18508) | 本文指出，对于使用平方损失函数的经验风险最小化(ERM)，其次优性必须归因于大的偏差而非方差，并且在ERM的平方误差的偏差-方差分解中，方差项必然具有极小的失误率。作者还提供了Chatterjee的不可允许性定理的简单证明，并表示他们的估计表明ERM的稳定性。 |
| [^7] | [Complex QA and language models hybrid architectures, Survey.](http://arxiv.org/abs/2302.09051) | 本文综述了语言模型架构和策略的最新进展，并重点关注混合技术在复杂问题回答中的应用，讨论了该领域的挑战和未来研究方向。 |
| [^8] | [Algorithmic Assistance with Recommendation-Dependent Preferences.](http://arxiv.org/abs/2208.07626) | 本研究提出了一个联合人机决策的委托代理模型，探讨了算法推荐对选择的影响和设计，特别关注算法对偏好的改变，以解决算法辅助可能带来的意外后果。 |

# 详细

[^1]: 使用平方Sigmoid TanH (SST)激活在数据限制下提高顺序模型性能

    Enhancing Sequential Model Performance with Squared Sigmoid TanH (SST) Activation Under Data Constraints

    [https://arxiv.org/abs/2402.09034](https://arxiv.org/abs/2402.09034)

    该论文提出了一种平方Sigmoid TanH（SST）激活函数，用于增强在数据限制下的顺序模型学习能力。通过数学平方放大强激活和弱激活之间的差异，改善梯度流和信息过滤。在多个应用中评估了SST驱动的LSTM和GRU模型的性能。

    

    激活函数通过引入非线性来使神经网络能够学习复杂的表示。虽然前馈模型通常使用修正线性单元，但是顺序模型如递归神经网络、长短时记忆（LSTM）和门控循环单元（GRU）仍然依赖于Sigmoid和TanH激活函数。然而，这些传统的激活函数常常在训练在小顺序数据集上时难以建模稀疏模式以有效捕获时间依赖性。为了解决这个限制，我们提出了特别针对在数据限制下增强顺序模型学习能力的平方Sigmoid TanH（SST）激活。SST通过数学平方来放大强激活和弱激活之间的差异，随着信号随时间传播，有助于改善梯度流和信息过滤。我们评估了使用SST的LSTM和GRU模型在不同应用中的性能。

    arXiv:2402.09034v1 Announce Type: cross Abstract: Activation functions enable neural networks to learn complex representations by introducing non-linearities. While feedforward models commonly use rectified linear units, sequential models like recurrent neural networks, long short-term memory (LSTMs) and gated recurrent units (GRUs) still rely on Sigmoid and TanH activation functions. However, these classical activation functions often struggle to model sparse patterns when trained on small sequential datasets to effectively capture temporal dependencies. To address this limitation, we propose squared Sigmoid TanH (SST) activation specifically tailored to enhance the learning capability of sequential models under data constraints. SST applies mathematical squaring to amplify differences between strong and weak activations as signals propagate over time, facilitating improved gradient flow and information filtering. We evaluate SST-powered LSTMs and GRUs for diverse applications, such a
    
[^2]: APALU: 一种可训练、适应性激活函数用于深度学习网络

    APALU: A Trainable, Adaptive Activation Function for Deep Learning Networks

    [https://arxiv.org/abs/2402.08244](https://arxiv.org/abs/2402.08244)

    APALU 是一种可训练的激活函数，通过增强深度学习的学习性能，在适应复杂数据表示的同时保持稳定和高效。 在图像分类任务中，APALU相对于传统激活函数能够显著提高准确性。

    

    激活函数是深度学习的关键组成部分，有助于提取复杂的数据模式。虽然类似ReLU及其变种的经典激活函数被广泛应用，但它们的静态特性和简洁性，尽管有利，但通常限制了它们在特定任务中的有效性。可训练的激活函数有时也难以适应数据的独特特征。针对这些限制，我们引入了一种新颖的可训练激活函数，即自适应分段逼近激活线性单元（APALU），以增强深度学习在各种任务中的学习性能。它具有一套独特的特性，使其能够在学习过程中保持稳定和高效，并适应复杂的数据表示。实验证实，在不同任务中，与广泛使用的激活函数相比，APALU取得了显著的改进。在图像分类中，APALU提升了MobileNet和GoogleNet的准确性。

    Activation function is a pivotal component of deep learning, facilitating the extraction of intricate data patterns. While classical activation functions like ReLU and its variants are extensively utilized, their static nature and simplicity, despite being advantageous, often limit their effectiveness in specialized tasks. The trainable activation functions also struggle sometimes to adapt to the unique characteristics of the data. Addressing these limitations, we introduce a novel trainable activation function, adaptive piecewise approximated activation linear unit (APALU), to enhance the learning performance of deep learning across a broad range of tasks. It presents a unique set of features that enable it to maintain stability and efficiency in the learning process while adapting to complex data representations. Experiments reveal significant improvements over widely used activation functions for different tasks. In image classification, APALU increases MobileNet and GoogleNet accur
    
[^3]: Knolling Bot: 从整洁的示范中学习机器人对象排列

    Knolling Bot: Learning Robotic Object Arrangement from Tidy Demonstrations

    [https://arxiv.org/abs/2310.04566](https://arxiv.org/abs/2310.04566)

    本论文介绍了一种自监督学习框架，利用Transformer神经网络使机器人能够从整齐排列的示范中理解和复制整洁的概念，从而实现整理物品的功能。

    

    地址：arXiv:2310.04566v2  公告类型：replace-cross  摘要：解决家庭空间中散乱物品的整理挑战受到整洁性的多样性和主观性的复杂性影响。正如人类语言的复杂性允许同一理念的多种表达一样，家庭整洁偏好和组织模式变化广泛，因此预设物体位置将限制对新物体和环境的适应性。受自然语言处理（NLP）的进展启发，本文引入一种自监督学习框架，使机器人能够从整洁布局的示范中理解和复制整洁的概念，类似于使用会话数据集训练大语言模型（LLM）。我们利用一个Transformer神经网络来预测后续物体的摆放位置。我们展示了一个“整理”系统，利用机械臂和RGB相机在桌子上组织不同大小和数量的物品。

    arXiv:2310.04566v2 Announce Type: replace-cross  Abstract: Addressing the challenge of organizing scattered items in domestic spaces is complicated by the diversity and subjective nature of tidiness. Just as the complexity of human language allows for multiple expressions of the same idea, household tidiness preferences and organizational patterns vary widely, so presetting object locations would limit the adaptability to new objects and environments. Inspired by advancements in natural language processing (NLP), this paper introduces a self-supervised learning framework that allows robots to understand and replicate the concept of tidiness from demonstrations of well-organized layouts, akin to using conversational datasets to train Large Language Models(LLM). We leverage a transformer neural network to predict the placement of subsequent objects. We demonstrate a ``knolling'' system with a robotic arm and an RGB camera to organize items of varying sizes and quantities on a table. Our 
    
[^4]: 混合任务元学习：一种用于可扩展和可转移带宽分配的图神经网络方法

    Hybrid-Task Meta-Learning: A Graph Neural Network Approach for Scalable and Transferable Bandwidth Allocation. (arXiv:2401.10253v1 [cs.NI])

    [http://arxiv.org/abs/2401.10253](http://arxiv.org/abs/2401.10253)

    本文提出了一种基于图神经网络的混合任务元学习算法，用于可扩展和可转移的带宽分配。通过引入GNN和HML算法，该方法在不同的通信场景下具有较好的性能和采样效率。

    

    本文提出了一种基于深度学习的带宽分配策略，该策略具有以下特点：1）随着用户数量的增加具有可扩展性；2）能够在不同的通信场景下进行转移，例如非平稳的无线信道、不同的服务质量要求和动态可用资源。为了支持可扩展性，带宽分配策略采用了图神经网络（GNN）进行表示，训练参数的数量随用户数量的增加而不变。为了实现GNN的泛化能力，我们开发了一种混合任务元学习（HML）算法，在元训练过程中使用不同的通信场景来训练GNN的初始参数。然后，在元测试过程中，使用少量样本对GNN进行微调以适应未见过的通信场景。仿真结果表明，与现有基准相比，我们的HML方法可以将初始性能提高8.79％，并提高采样效率73％。在微调后，

    In this paper, we develop a deep learning-based bandwidth allocation policy that is: 1) scalable with the number of users and 2) transferable to different communication scenarios, such as non-stationary wireless channels, different quality-of-service (QoS) requirements, and dynamically available resources. To support scalability, the bandwidth allocation policy is represented by a graph neural network (GNN), with which the number of training parameters does not change with the number of users. To enable the generalization of the GNN, we develop a hybrid-task meta-learning (HML) algorithm that trains the initial parameters of the GNN with different communication scenarios during meta-training. Next, during meta-testing, a few samples are used to fine-tune the GNN with unseen communication scenarios. Simulation results demonstrate that our HML approach can improve the initial performance by $8.79\%$, and sampling efficiency by $73\%$, compared with existing benchmarks. After fine-tuning,
    
[^5]: LPAC: 可学习的感知-行动-通信循环及其在覆盖控制中的应用

    LPAC: Learnable Perception-Action-Communication Loops with Applications to Coverage Control. (arXiv:2401.04855v1 [cs.RO])

    [http://arxiv.org/abs/2401.04855](http://arxiv.org/abs/2401.04855)

    提出了一种可学习的感知-行动-通信(LPAC)架构，使用卷积神经网络处理环境感知，图神经网络实现机器人之间的信息交流，浅层多层感知机计算机器人的动作。使用集中式显微算法训练模型，实现机器人群体的协作。

    

    覆盖控制是指导机器人群体协同监测未知的感兴趣特征或现象的问题。在有限的通信和感知能力的分散设置中，这个问题具有挑战性。本文提出了一种可学习的感知-行动-通信(LPAC)架构来解决覆盖控制问题。在该解决方案中，卷积神经网络(CNN)处理了环境的局部感知；图神经网络(GNN)实现了邻近机器人之间的相关信息通信；最后，浅层多层感知机(MLP)计算机器人的动作。通信模块中的GNN通过计算应该与邻居通信哪些信息以及如何利用接收到的信息采取适当的行动来实现机器人群体的协作。我们使用一个知晓整个环境的集中式显微算法来进行模型的训练。

    Coverage control is the problem of navigating a robot swarm to collaboratively monitor features or a phenomenon of interest not known a priori. The problem is challenging in decentralized settings with robots that have limited communication and sensing capabilities. This paper proposes a learnable Perception-Action-Communication (LPAC) architecture for the coverage control problem. In the proposed solution, a convolution neural network (CNN) processes localized perception of the environment; a graph neural network (GNN) enables communication of relevant information between neighboring robots; finally, a shallow multi-layer perceptron (MLP) computes robot actions. The GNN in the communication module enables collaboration in the robot swarm by computing what information to communicate with neighbors and how to use received information to take appropriate actions. We train models using imitation learning with a centralized clairvoyant algorithm that is aware of the entire environment. Eva
    
[^6]: 论经验风险最小化的方差、可允许性和稳定性

    On the Variance, Admissibility, and Stability of Empirical Risk Minimization. (arXiv:2305.18508v1 [math.ST])

    [http://arxiv.org/abs/2305.18508](http://arxiv.org/abs/2305.18508)

    本文指出，对于使用平方损失函数的经验风险最小化(ERM)，其次优性必须归因于大的偏差而非方差，并且在ERM的平方误差的偏差-方差分解中，方差项必然具有极小的失误率。作者还提供了Chatterjee的不可允许性定理的简单证明，并表示他们的估计表明ERM的稳定性。

    

    众所周知，使用平方损失的经验风险最小化可能会达到极小的最大失误率。本文的关键信息是，在温和的假设下，ERM的次优性必须归因于大的偏差而非方差。在ERM的平方误差的偏差-方差分解中，方差项必然具有极小的失误率。我们为固定设计提供了一个简单的、使用概率方法证明这一事实的证明。然后，我们在随机设计设置下为各种模型证明了这一结果。此外，我们提供了 Chatterjee 不可允许性定理 (Chatterjee, 2014, Theorem 1.4) 的简单证明，该定理指出，在固定设计设置中，ERM不能被排除为一种最优方法，并将该结果扩展到随机设计设置。我们还表明，我们的估计表明ERM的稳定性，为Caponnetto和Rakhlin(2006)的非Donsker类的主要结果提供了补充。

    It is well known that Empirical Risk Minimization (ERM) with squared loss may attain minimax suboptimal error rates (Birg\'e and Massart, 1993). The key message of this paper is that, under mild assumptions, the suboptimality of ERM must be due to large bias rather than variance. More precisely, in the bias-variance decomposition of the squared error of the ERM, the variance term necessarily enjoys the minimax rate. In the case of fixed design, we provide an elementary proof of this fact using the probabilistic method. Then, we prove this result for various models in the random design setting. In addition, we provide a simple proof of Chatterjee's admissibility theorem (Chatterjee, 2014, Theorem 1.4), which states that ERM cannot be ruled out as an optimal method, in the fixed design setting, and extend this result to the random design setting. We also show that our estimates imply stability of ERM, complementing the main result of Caponnetto and Rakhlin (2006) for non-Donsker classes.
    
[^7]: 复杂问答和语言模型混合架构综述

    Complex QA and language models hybrid architectures, Survey. (arXiv:2302.09051v4 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.09051](http://arxiv.org/abs/2302.09051)

    本文综述了语言模型架构和策略的最新进展，并重点关注混合技术在复杂问题回答中的应用，讨论了该领域的挑战和未来研究方向。

    

    本文回顾了语言模型架构和策略的最新进展，重点关注混合技术在复杂问题回答中的应用。大型语言模型能够在标准问题上利用公共数据，但在解决更具体的复杂问题时（如在不同文化中个人自由概念的变化如何？什么是为减少气候变化而实现的最佳发电方法组合？），需要特定的架构、知识、技能、方法、敏感数据保护、可解释性、人类审批和多功能反馈。最近的项目如ChatGPT和GALACTICA允许非专业人员了解LLM在复杂QA中的巨大潜力以及同等强大的局限性。在本文中，我们首先审查所需的技能和评估技术。然后，我们综述了现有的混合架构，将LLM与基于规则的方法、信息检索、知识图谱和其他AI/ML技术相结合。最后，我们指出这些CQA系统的挑战，并提出未来研究的可能方向。

    This paper reviews the state-of-the-art of language models architectures and strategies for "complex" question-answering (QA, CQA, CPS) with a focus on hybridization. Large Language Models (LLM) are good at leveraging public data on standard problems but once you want to tackle more specific complex questions or problems (e.g. How does the concept of personal freedom vary between different cultures ? What is the best mix of power generation methods to reduce climate change ?) you may need specific architecture, knowledge, skills, methods, sensitive data protection, explainability, human approval and versatile feedback... Recent projects like ChatGPT and GALACTICA have allowed non-specialists to grasp the great potential as well as the equally strong limitations of LLM in complex QA. In this paper, we start by reviewing required skills and evaluation techniques. We integrate findings from the robust community edited research papers BIG, BLOOM and HELM which open source, benchmark and an
    
[^8]: 算法辅助下的推荐相关偏好

    Algorithmic Assistance with Recommendation-Dependent Preferences. (arXiv:2208.07626v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.07626](http://arxiv.org/abs/2208.07626)

    本研究提出了一个联合人机决策的委托代理模型，探讨了算法推荐对选择的影响和设计，特别关注算法对偏好的改变，以解决算法辅助可能带来的意外后果。

    

    当算法提供风险评估时，我们通常将其视为对人类决策的有益输入，例如将风险评分呈现给法官或医生。然而，决策者可能不仅仅只针对算法提供的信息做出反应。决策者还可能将算法推荐视为默认操作，使其难以偏离，例如法官在对被告进行高风险评估的时候不愿意推翻，或医生担心偏离推荐的程序会带来后果。为了解决算法辅助的这种意外后果，我们提出了一个联合人机决策的委托代理模型。在该模型中，我们考虑了算法推荐对选择的影响和设计，这种影响不仅仅是通过改变信念，还通过改变偏好。我们从制度因素和行为经济学中的已有模型等方面进行了这个假设的动机论证。

    When an algorithm provides risk assessments, we typically think of them as helpful inputs to human decisions, such as when risk scores are presented to judges or doctors. However, a decision-maker may not only react to the information provided by the algorithm. The decision-maker may also view the algorithmic recommendation as a default action, making it costly for them to deviate, such as when a judge is reluctant to overrule a high-risk assessment for a defendant or a doctor fears the consequences of deviating from recommended procedures. To address such unintended consequences of algorithmic assistance, we propose a principal-agent model of joint human-machine decision-making. Within this model, we consider the effect and design of algorithmic recommendations when they affect choices not just by shifting beliefs, but also by altering preferences. We motivate this assumption from institutional factors, such as a desire to avoid audits, as well as from well-established models in behav
    


# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DIRECT: Deep Active Learning under Imbalance and Label Noise](https://rss.arxiv.org/abs/2312.09196) | 这篇论文提出了一种名为DIRECT的算法，用于处理不平衡和标签噪声下的深度主动学习问题。通过确定类别分割阈值并标记最不确定且离其最近的示例，该算法能够有效解决罕见类和少数类的性能问题，并具有批次标记和对标签噪声的容忍能力。 |
| [^2] | [Securing Large Language Models: Threats, Vulnerabilities and Responsible Practices](https://arxiv.org/abs/2403.12503) | 该论文全面调查了与大型语言模型相关的安全和隐私问题，并提出了未来研究的有前途方向。 |
| [^3] | [Mitigating Object Hallucination in Large Vision-Language Models via Classifier-Free Guidance](https://arxiv.org/abs/2402.08680) | 本文介绍了一种名为MARINE的框架，用于通过无分类器引导来减少大型视觉语言模型的物体幻觉。该框架无需训练或API访问，并通过集成视觉模型和引入额外的物体基础特征来提高模型的生成精确性和效率。 |
| [^4] | [Efficient Constrained $k$-Center Clustering with Background Knowledge.](http://arxiv.org/abs/2401.12533) | 本论文提出了一种在k中心聚类上利用背景知识的约束聚类算法，通过采用一系列技术，得到了效率高且具有最佳近似比例2的算法。 |
| [^5] | [IoTGeM: Generalizable Models for Behaviour-Based IoT Attack Detection.](http://arxiv.org/abs/2401.01343) | 本研究提出了一种通用模型，用于基于行为的物联网攻击检测。该模型通过改进滚动窗口特征提取、引入多步骤特征选择、使用隔离的训练和测试数据集以及使用可解释的人工智能技术来提高检测和性能，并且经过严格评估。 |
| [^6] | [Law of Balance and Stationary Distribution of Stochastic Gradient Descent.](http://arxiv.org/abs/2308.06671) | 本文证明了随机梯度下降算法中的小批量噪音会使解决方案向平衡解靠近，只要损失函数包含重新缩放对称性。利用这个结果，我们导出了对角线性网络的随机梯度流稳态分布，该分布展示了复杂的非线性现象。这些发现揭示了动态梯度下降法在训练神经网络中的工作原理。 |

# 详细

[^1]: DIRECT: 处理不平衡和标签噪声下的深度主动学习

    DIRECT: Deep Active Learning under Imbalance and Label Noise

    [https://rss.arxiv.org/abs/2312.09196](https://rss.arxiv.org/abs/2312.09196)

    这篇论文提出了一种名为DIRECT的算法，用于处理不平衡和标签噪声下的深度主动学习问题。通过确定类别分割阈值并标记最不确定且离其最近的示例，该算法能够有效解决罕见类和少数类的性能问题，并具有批次标记和对标签噪声的容忍能力。

    

    在现实世界的机器学习应用中，类别不平衡是一个普遍存在的问题，通常会导致罕见和少数类的性能较差。在大量未标记数据的情况下，主动学习可能是解决该问题的最有效技术，它从根本上采集更平衡和具有信息量的标记示例进行注释。标签噪声是数据注释任务中另一个常见问题，对于主动学习方法来说尤其具有挑战性。本文首次研究了在类别不平衡和标签噪声下的主动学习。我们提出了一种新颖的算法，能够稳健地确定类别分割阈值并标记最不确定且离其最近的示例。通过将问题简化为一维主动学习，我们的算法DIRECT能够利用经典的主动学习文献来解决批次标记和对标签噪声的容忍等问题。我们在大量实验中展示了算法的性能。

    Class imbalance is a prevalent issue in real world machine learning applications, often leading to poor performance in rare and minority classes. With an abundance of wild unlabeled data, active learning is perhaps the most effective technique in solving the problem at its root -- collecting a more balanced and informative set of labeled examples during annotation. Label noise is another common issue in data annotation jobs, which is especially challenging for active learning methods. In this work, we conduct the first study of active learning under both class imbalance and label noise. We propose a novel algorithm that robustly identifies the class separation threshold and annotates the most uncertain examples that are closest from it. Through a novel reduction to one-dimensional active learning, our algorithm DIRECT is able to leverage the classic active learning literature to address issues such as batch labeling and tolerance towards label noise. We present extensive experiments on
    
[^2]: 保护大型语言模型：威胁、漏洞和负责任的实践

    Securing Large Language Models: Threats, Vulnerabilities and Responsible Practices

    [https://arxiv.org/abs/2403.12503](https://arxiv.org/abs/2403.12503)

    该论文全面调查了与大型语言模型相关的安全和隐私问题，并提出了未来研究的有前途方向。

    

    大型语言模型(LLMs)显著改变了自然语言处理(NLP)的格局。它们对各种任务产生了影响，从而彻底改变了我们处理语言理解和生成的方式。然而，除了它们引人注目的实用性外，LLMs还带来了重要的安全和风险考虑。这些挑战需要仔细研究，以确保负责任的部署，并防范潜在的漏洞。本研究全面调查了与LLMs相关的安全和隐私问题，从五个主题角度进行：安全和隐私问题、对抗性攻击的漏洞、LLMs误用可能造成的潜在危害、缓解策略以解决这些挑战，同时识别当前策略的局限性。最后，本文建议未来研究的有前途的方向，以增强LLMs的安全和风险管理。

    arXiv:2403.12503v1 Announce Type: cross  Abstract: Large language models (LLMs) have significantly transformed the landscape of Natural Language Processing (NLP). Their impact extends across a diverse spectrum of tasks, revolutionizing how we approach language understanding and generations. Nevertheless, alongside their remarkable utility, LLMs introduce critical security and risk considerations. These challenges warrant careful examination to ensure responsible deployment and safeguard against potential vulnerabilities. This research paper thoroughly investigates security and privacy concerns related to LLMs from five thematic perspectives: security and privacy concerns, vulnerabilities against adversarial attacks, potential harms caused by misuses of LLMs, mitigation strategies to address these challenges while identifying limitations of current strategies. Lastly, the paper recommends promising avenues for future research to enhance the security and risk management of LLMs.
    
[^3]: 通过无分类器引导来减轻大型视觉语言模型中的物体幻觉

    Mitigating Object Hallucination in Large Vision-Language Models via Classifier-Free Guidance

    [https://arxiv.org/abs/2402.08680](https://arxiv.org/abs/2402.08680)

    本文介绍了一种名为MARINE的框架，用于通过无分类器引导来减少大型视觉语言模型的物体幻觉。该框架无需训练或API访问，并通过集成视觉模型和引入额外的物体基础特征来提高模型的生成精确性和效率。

    

    大型视觉语言模型（LVLM）的进展越来越突出了它们在图像中产生虚假物体的严重问题。为了解决这个问题，先前的研究着重于使用特殊策划的数据集或强大的LLM（例如GPT-3.5）来纠正LVLM的输出。然而，这些方法要求昂贵的训练/微调或API访问先进的LLM来在生成后纠正模型的输出。在本文中，我们通过引入一个名为通过无分类器引导缓解幻觉的框架（MARINE）来解决这个挑战，该框架既无需训练也无需API访问，可以在生成过程中有效地减少物体幻觉。具体而言，MARINE通过集成现有的开源视觉模型丰富LVLM的视觉语境，并使用无分类器引导来整合额外的物体基础特征，以提高LVLM生成的精确性。

    The advancement of Large Vision-Language Models (LVLMs) has increasingly highlighted the critical issue of their tendency to hallucinate non-existing objects in the images. To address this issue, previous works focused on using specially curated datasets or powerful LLMs (e.g., GPT-3.5) to rectify the outputs of LVLMs. However, these approaches require either expensive training/fine-tuning or API access to advanced LLMs to correct the model's output post-generation. In this paper, we tackle this challenge by introducing a framework called Mitigating hallucinAtion via classifieR-Free guIdaNcE (MARINE), which is both training-free and API-free, and can effectively and efficiently reduce object hallucinations during the generation process. Specifically, MARINE enriches the visual context of LVLMs by integrating existing open-source vision models, and employs classifier-free guidance to incorporate the additional object grounding features to improve the precision of LVLMs' generations. Thr
    
[^4]: 有效利用背景知识的约束k中心聚类

    Efficient Constrained $k$-Center Clustering with Background Knowledge. (arXiv:2401.12533v1 [cs.LG])

    [http://arxiv.org/abs/2401.12533](http://arxiv.org/abs/2401.12533)

    本论文提出了一种在k中心聚类上利用背景知识的约束聚类算法，通过采用一系列技术，得到了效率高且具有最佳近似比例2的算法。

    

    中心为基础的聚类在理论和实践中都引起了重要的研究兴趣。在许多实际应用中，输入数据通常包含可以用于改进聚类结果的背景知识。在这项工作中，我们基于广泛采用的k中心聚类，并将其输入的背景知识建模为必连（ML）和不连（CL）约束集。然而，大多数包括k中心在内的聚类问题本质上都是NP困难的，而更复杂的受约束变体被认为受到更严重的近似和计算障碍的限制，极大地限制了它们的适用性。通过采用一系列技术，包括反支配集，线性规划（LP）整数平面和LP对偶性，我们得到了第一个具有最佳近似比例2的约束k中心的高效近似算法。我们还构建了竞争基准算法，并对我们的近似算法进行了实证评估。

    Center-based clustering has attracted significant research interest from both theory and practice. In many practical applications, input data often contain background knowledge that can be used to improve clustering results. In this work, we build on widely adopted $k$-center clustering and model its input background knowledge as must-link (ML) and cannot-link (CL) constraint sets. However, most clustering problems including $k$-center are inherently $\mathcal{NP}$-hard, while the more complex constrained variants are known to suffer severer approximation and computation barriers that significantly limit their applicability. By employing a suite of techniques including reverse dominating sets, linear programming (LP) integral polyhedron, and LP duality, we arrive at the first efficient approximation algorithm for constrained $k$-center with the best possible ratio of 2. We also construct competitive baseline algorithms and empirically evaluate our approximation algorithm against them o
    
[^5]: IoTGeM: 用于基于行为的物联网攻击检测的通用模型

    IoTGeM: Generalizable Models for Behaviour-Based IoT Attack Detection. (arXiv:2401.01343v1 [cs.CR])

    [http://arxiv.org/abs/2401.01343](http://arxiv.org/abs/2401.01343)

    本研究提出了一种通用模型，用于基于行为的物联网攻击检测。该模型通过改进滚动窗口特征提取、引入多步骤特征选择、使用隔离的训练和测试数据集以及使用可解释的人工智能技术来提高检测和性能，并且经过严格评估。

    

    过去关于物联网设备网络的基于行为的攻击检测研究，所得到的机器学习模型的适应能力有限，并且往往没有得到证明。本文提出了一种针对建模物联网网络攻击的方法，着重于通用性，同时也能提高检测和性能。首先，我们提出了一种改进的滚动窗口特征提取方法，并引入了一个多步骤特征选择过程来减少过拟合。其次，我们使用隔离的训练和测试数据集来构建和测试模型，从而避免了先前模型在通用性方面的常见数据泄漏问题。第三，我们使用多样化的机器学习模型、评估指标和数据集对我们的方法进行了严格评估。最后，我们使用可解释的人工智能技术来增加模型的可信度，从而能够识别出支撑攻击准确检测的特征。

    Previous research on behaviour-based attack detection on networks of IoT devices has resulted in machine learning models whose ability to adapt to unseen data is limited, and often not demonstrated. In this paper we present an approach for modelling IoT network attacks that focuses on generalizability, yet also leads to better detection and performance. First, we present an improved rolling window approach for feature extraction, and introduce a multi-step feature selection process that reduces overfitting. Second, we build and test models using isolated train and test datasets, thereby avoiding common data leaks that have limited the generalizability of previous models. Third, we rigorously evaluate our methodology using a diverse portfolio of machine learning models, evaluation metrics and datasets. Finally, we build confidence in the models by using explainable AI techniques, allowing us to identify the features that underlie accurate detection of attacks.
    
[^6]: 动态梯度下降法的平衡法则与稳态分布

    Law of Balance and Stationary Distribution of Stochastic Gradient Descent. (arXiv:2308.06671v1 [cs.LG])

    [http://arxiv.org/abs/2308.06671](http://arxiv.org/abs/2308.06671)

    本文证明了随机梯度下降算法中的小批量噪音会使解决方案向平衡解靠近，只要损失函数包含重新缩放对称性。利用这个结果，我们导出了对角线性网络的随机梯度流稳态分布，该分布展示了复杂的非线性现象。这些发现揭示了动态梯度下降法在训练神经网络中的工作原理。

    

    随机梯度下降（SGD）算法是我们用于训练神经网络的算法。然而，我们很难理解SGD如何在神经网络的非线性和退化的损失曲面中进行导航。在这项工作中，我们证明了SGD的小批量噪音可以使解决方案向平衡解靠近，只要损失函数包含一个重新缩放对称性。由于简单扩散过程和SGD动力学的差异在对称性存在时最重要，我们的理论表明，损失函数的对称性是了解SGD工作方式的重要线索。然后，我们将这个结果应用于导出具有任意深度和宽度的对角线性网络的随机梯度流的稳态分布。稳态分布展现了复杂的非线性现象，如相变、破坏的遍历性和波动反转。这些现象仅在深层网络中存在，表明了一种基本的新的加深训练理论。

    The stochastic gradient descent (SGD) algorithm is the algorithm we use to train neural networks. However, it remains poorly understood how the SGD navigates the highly nonlinear and degenerate loss landscape of a neural network. In this work, we prove that the minibatch noise of SGD regularizes the solution towards a balanced solution whenever the loss function contains a rescaling symmetry. Because the difference between a simple diffusion process and SGD dynamics is the most significant when symmetries are present, our theory implies that the loss function symmetries constitute an essential probe of how SGD works. We then apply this result to derive the stationary distribution of stochastic gradient flow for a diagonal linear network with arbitrary depth and width. The stationary distribution exhibits complicated nonlinear phenomena such as phase transitions, broken ergodicity, and fluctuation inversion. These phenomena are shown to exist uniquely in deep networks, implying a fundam
    


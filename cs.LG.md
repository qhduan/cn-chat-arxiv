# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DIRECT: Deep Active Learning under Imbalance and Label Noise](https://rss.arxiv.org/abs/2312.09196) | 这篇论文提出了一种名为DIRECT的算法，用于处理不平衡和标签噪声下的深度主动学习问题。通过确定类别分割阈值并标记最不确定且离其最近的示例，该算法能够有效解决罕见类和少数类的性能问题，并具有批次标记和对标签噪声的容忍能力。 |
| [^2] | [Securing Large Language Models: Threats, Vulnerabilities and Responsible Practices](https://arxiv.org/abs/2403.12503) | 该论文全面调查了与大型语言模型相关的安全和隐私问题，并提出了未来研究的有前途方向。 |
| [^3] | [Mitigating Object Hallucination in Large Vision-Language Models via Classifier-Free Guidance](https://arxiv.org/abs/2402.08680) | 本文介绍了一种名为MARINE的框架，用于通过无分类器引导来减少大型视觉语言模型的物体幻觉。该框架无需训练或API访问，并通过集成视觉模型和引入额外的物体基础特征来提高模型的生成精确性和效率。 |
| [^4] | [Bandit Convex Optimisation](https://arxiv.org/abs/2402.06535) | 这篇论文介绍了强盗凸优化的基本框架和用于解决这一问题的多种工具。虽然没有太多创新，但通过以新颖的方式应用现有工具，获得了新的算法和改进了一些界限。 |
| [^5] | [Learning a Gaussian Mixture for Sparsity Regularization in Inverse Problems.](http://arxiv.org/abs/2401.16612) | 本研究提出了一种基于高斯混合模型的稀疏正则化方法，通过神经网络进行贝叶斯估计，有效地解决了逆问题中的稀疏建模和参数估计问题。 |
| [^6] | [Efficient Constrained $k$-Center Clustering with Background Knowledge.](http://arxiv.org/abs/2401.12533) | 本论文提出了一种在k中心聚类上利用背景知识的约束聚类算法，通过采用一系列技术，得到了效率高且具有最佳近似比例2的算法。 |
| [^7] | [IoTGeM: Generalizable Models for Behaviour-Based IoT Attack Detection.](http://arxiv.org/abs/2401.01343) | 本研究提出了一种通用模型，用于基于行为的物联网攻击检测。该模型通过改进滚动窗口特征提取、引入多步骤特征选择、使用隔离的训练和测试数据集以及使用可解释的人工智能技术来提高检测和性能，并且经过严格评估。 |
| [^8] | [A Unified Framework to Enforce, Discover, and Promote Symmetry in Machine Learning.](http://arxiv.org/abs/2311.00212) | 本文提供了一个统一的框架，通过三种方式将对称性融入机器学习模型：1. 强制已知的对称性；2. 发现未知的对称性；3. 在训练过程中促进对称性。 |
| [^9] | [Law of Balance and Stationary Distribution of Stochastic Gradient Descent.](http://arxiv.org/abs/2308.06671) | 本文证明了随机梯度下降算法中的小批量噪音会使解决方案向平衡解靠近，只要损失函数包含重新缩放对称性。利用这个结果，我们导出了对角线性网络的随机梯度流稳态分布，该分布展示了复杂的非线性现象。这些发现揭示了动态梯度下降法在训练神经网络中的工作原理。 |
| [^10] | [For Kernel Range Spaces a Constant Number of Queries Are Sufficient.](http://arxiv.org/abs/2306.16516) | 对于核范围空间，引入了ε-覆盖的概念，用于处理不确定或不精确的数据分析任务。 |
| [^11] | [Three iterations of $(1-d)$-WL test distinguish non isometric clouds of $d$-dimensional points.](http://arxiv.org/abs/2303.12853) | 本文研究了WL测试在点云中的应用，结果发现三次迭代的$(d-1)$-WL测试可以区分$d$维欧几里得空间中的点云，且只需要一次迭代的$d$-WL测试就可以达到完整性。 |
| [^12] | [Second-order Conditional Gradient Sliding.](http://arxiv.org/abs/2002.08907) | 提出了一种二阶条件梯度滑动（SOCGS）算法，可以高效解决约束二次凸优化问题，并在有限次线性收敛迭代后二次收敛于原始间隙。 |

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
    
[^4]: Bandit Convex Optimisation（强盗凸优化）

    Bandit Convex Optimisation

    [https://arxiv.org/abs/2402.06535](https://arxiv.org/abs/2402.06535)

    这篇论文介绍了强盗凸优化的基本框架和用于解决这一问题的多种工具。虽然没有太多创新，但通过以新颖的方式应用现有工具，获得了新的算法和改进了一些界限。

    

    强盗凸优化是研究零阶凸优化的基本框架。本文介绍了用于解决该问题的许多工具，包括切平面方法、内点方法、连续指数权重、梯度下降和在线牛顿步骤。解释了许多假设和设置之间的细微差别。尽管在这里没有太多真正新的东西，但一些现有工具以新颖的方式应用于获得新算法。一些界限稍微改进了一些。

    Bandit convex optimisation is a fundamental framework for studying zeroth-order convex optimisation. These notes cover the many tools used for this problem, including cutting plane methods, interior point methods, continuous exponential weights, gradient descent and online Newton step. The nuances between the many assumptions and setups are explained. Although there is not much truly new here, some existing tools are applied in novel ways to obtain new algorithms. A few bounds are improved in minor ways.
    
[^5]: 在逆问题中学习高斯混合物进行稀疏正则化

    Learning a Gaussian Mixture for Sparsity Regularization in Inverse Problems. (arXiv:2401.16612v1 [stat.ML])

    [http://arxiv.org/abs/2401.16612](http://arxiv.org/abs/2401.16612)

    本研究提出了一种基于高斯混合模型的稀疏正则化方法，通过神经网络进行贝叶斯估计，有效地解决了逆问题中的稀疏建模和参数估计问题。

    

    在逆问题中，广泛认为引入稀疏先验对解决方案具有正则化效果。这种方法是基于一个先验假设，即未知量可以在一个有限数量的显著成分的基础上适当表示，而大多数系数接近于零。这种情况在现实世界中经常出现，比如分段平滑信号。在本研究中，我们提出了一种以高斯退化混合物形式表述的概率稀疏先验，能够对于任意基进行稀疏建模。在这个前提下，我们设计了一个可以解释为线性逆问题的贝叶斯估计器的神经网络。此外，我们提出了一种有监督和无监督的训练策略来估计这个网络的参数。为了评估我们方法的有效性，我们进行了与常用的稀疏正则化方法的数值比较。

    In inverse problems, it is widely recognized that the incorporation of a sparsity prior yields a regularization effect on the solution. This approach is grounded on the a priori assumption that the unknown can be appropriately represented in a basis with a limited number of significant components, while most coefficients are close to zero. This occurrence is frequently observed in real-world scenarios, such as with piecewise smooth signals. In this study, we propose a probabilistic sparsity prior formulated as a mixture of degenerate Gaussians, capable of modeling sparsity with respect to a generic basis. Under this premise, we design a neural network that can be interpreted as the Bayes estimator for linear inverse problems. Additionally, we put forth both a supervised and an unsupervised training strategy to estimate the parameters of this network. To evaluate the effectiveness of our approach, we conduct a numerical comparison with commonly employed sparsity-promoting regularization
    
[^6]: 有效利用背景知识的约束k中心聚类

    Efficient Constrained $k$-Center Clustering with Background Knowledge. (arXiv:2401.12533v1 [cs.LG])

    [http://arxiv.org/abs/2401.12533](http://arxiv.org/abs/2401.12533)

    本论文提出了一种在k中心聚类上利用背景知识的约束聚类算法，通过采用一系列技术，得到了效率高且具有最佳近似比例2的算法。

    

    中心为基础的聚类在理论和实践中都引起了重要的研究兴趣。在许多实际应用中，输入数据通常包含可以用于改进聚类结果的背景知识。在这项工作中，我们基于广泛采用的k中心聚类，并将其输入的背景知识建模为必连（ML）和不连（CL）约束集。然而，大多数包括k中心在内的聚类问题本质上都是NP困难的，而更复杂的受约束变体被认为受到更严重的近似和计算障碍的限制，极大地限制了它们的适用性。通过采用一系列技术，包括反支配集，线性规划（LP）整数平面和LP对偶性，我们得到了第一个具有最佳近似比例2的约束k中心的高效近似算法。我们还构建了竞争基准算法，并对我们的近似算法进行了实证评估。

    Center-based clustering has attracted significant research interest from both theory and practice. In many practical applications, input data often contain background knowledge that can be used to improve clustering results. In this work, we build on widely adopted $k$-center clustering and model its input background knowledge as must-link (ML) and cannot-link (CL) constraint sets. However, most clustering problems including $k$-center are inherently $\mathcal{NP}$-hard, while the more complex constrained variants are known to suffer severer approximation and computation barriers that significantly limit their applicability. By employing a suite of techniques including reverse dominating sets, linear programming (LP) integral polyhedron, and LP duality, we arrive at the first efficient approximation algorithm for constrained $k$-center with the best possible ratio of 2. We also construct competitive baseline algorithms and empirically evaluate our approximation algorithm against them o
    
[^7]: IoTGeM: 用于基于行为的物联网攻击检测的通用模型

    IoTGeM: Generalizable Models for Behaviour-Based IoT Attack Detection. (arXiv:2401.01343v1 [cs.CR])

    [http://arxiv.org/abs/2401.01343](http://arxiv.org/abs/2401.01343)

    本研究提出了一种通用模型，用于基于行为的物联网攻击检测。该模型通过改进滚动窗口特征提取、引入多步骤特征选择、使用隔离的训练和测试数据集以及使用可解释的人工智能技术来提高检测和性能，并且经过严格评估。

    

    过去关于物联网设备网络的基于行为的攻击检测研究，所得到的机器学习模型的适应能力有限，并且往往没有得到证明。本文提出了一种针对建模物联网网络攻击的方法，着重于通用性，同时也能提高检测和性能。首先，我们提出了一种改进的滚动窗口特征提取方法，并引入了一个多步骤特征选择过程来减少过拟合。其次，我们使用隔离的训练和测试数据集来构建和测试模型，从而避免了先前模型在通用性方面的常见数据泄漏问题。第三，我们使用多样化的机器学习模型、评估指标和数据集对我们的方法进行了严格评估。最后，我们使用可解释的人工智能技术来增加模型的可信度，从而能够识别出支撑攻击准确检测的特征。

    Previous research on behaviour-based attack detection on networks of IoT devices has resulted in machine learning models whose ability to adapt to unseen data is limited, and often not demonstrated. In this paper we present an approach for modelling IoT network attacks that focuses on generalizability, yet also leads to better detection and performance. First, we present an improved rolling window approach for feature extraction, and introduce a multi-step feature selection process that reduces overfitting. Second, we build and test models using isolated train and test datasets, thereby avoiding common data leaks that have limited the generalizability of previous models. Third, we rigorously evaluate our methodology using a diverse portfolio of machine learning models, evaluation metrics and datasets. Finally, we build confidence in the models by using explainable AI techniques, allowing us to identify the features that underlie accurate detection of attacks.
    
[^8]: 在机器学习中强制、发现和推动对称性的统一框架

    A Unified Framework to Enforce, Discover, and Promote Symmetry in Machine Learning. (arXiv:2311.00212v1 [cs.LG])

    [http://arxiv.org/abs/2311.00212](http://arxiv.org/abs/2311.00212)

    本文提供了一个统一的框架，通过三种方式将对称性融入机器学习模型：1. 强制已知的对称性；2. 发现未知的对称性；3. 在训练过程中促进对称性。

    

    对称性存在于自然界中，并在物理学和机器学习中扮演着越来越核心的角色。基本对称性，如庞加莱不变性，使在地球上实验室中发现的物理定律能够推广到宇宙的最远处。对称性对于在机器学习应用中实现这种推广能力至关重要。例如，在图像分类中的平移不变性允许使用参数更少的模型（如卷积神经网络）在较小的数据集上进行训练，并达到最先进的性能。本文提供了一个统一的理论和方法框架，用于在机器学习模型中以三种方式融入对称性：1. 在训练模型时强制已知的对称性；2. 发现给定模型或数据集的未知对称性；3. 在训练过程中通过学习打破用户指定的候选群体内的对称性来促进对称性。

    Symmetry is present throughout nature and continues to play an increasingly central role in physics and machine learning. Fundamental symmetries, such as Poincar\'{e} invariance, allow physical laws discovered in laboratories on Earth to be extrapolated to the farthest reaches of the universe. Symmetry is essential to achieving this extrapolatory power in machine learning applications. For example, translation invariance in image classification allows models with fewer parameters, such as convolutional neural networks, to be trained on smaller data sets and achieve state-of-the-art performance. In this paper, we provide a unifying theoretical and methodological framework for incorporating symmetry into machine learning models in three ways: 1. enforcing known symmetry when training a model; 2. discovering unknown symmetries of a given model or data set; and 3. promoting symmetry during training by learning a model that breaks symmetries within a user-specified group of candidates when 
    
[^9]: 动态梯度下降法的平衡法则与稳态分布

    Law of Balance and Stationary Distribution of Stochastic Gradient Descent. (arXiv:2308.06671v1 [cs.LG])

    [http://arxiv.org/abs/2308.06671](http://arxiv.org/abs/2308.06671)

    本文证明了随机梯度下降算法中的小批量噪音会使解决方案向平衡解靠近，只要损失函数包含重新缩放对称性。利用这个结果，我们导出了对角线性网络的随机梯度流稳态分布，该分布展示了复杂的非线性现象。这些发现揭示了动态梯度下降法在训练神经网络中的工作原理。

    

    随机梯度下降（SGD）算法是我们用于训练神经网络的算法。然而，我们很难理解SGD如何在神经网络的非线性和退化的损失曲面中进行导航。在这项工作中，我们证明了SGD的小批量噪音可以使解决方案向平衡解靠近，只要损失函数包含一个重新缩放对称性。由于简单扩散过程和SGD动力学的差异在对称性存在时最重要，我们的理论表明，损失函数的对称性是了解SGD工作方式的重要线索。然后，我们将这个结果应用于导出具有任意深度和宽度的对角线性网络的随机梯度流的稳态分布。稳态分布展现了复杂的非线性现象，如相变、破坏的遍历性和波动反转。这些现象仅在深层网络中存在，表明了一种基本的新的加深训练理论。

    The stochastic gradient descent (SGD) algorithm is the algorithm we use to train neural networks. However, it remains poorly understood how the SGD navigates the highly nonlinear and degenerate loss landscape of a neural network. In this work, we prove that the minibatch noise of SGD regularizes the solution towards a balanced solution whenever the loss function contains a rescaling symmetry. Because the difference between a simple diffusion process and SGD dynamics is the most significant when symmetries are present, our theory implies that the loss function symmetries constitute an essential probe of how SGD works. We then apply this result to derive the stationary distribution of stochastic gradient flow for a diagonal linear network with arbitrary depth and width. The stationary distribution exhibits complicated nonlinear phenomena such as phase transitions, broken ergodicity, and fluctuation inversion. These phenomena are shown to exist uniquely in deep networks, implying a fundam
    
[^10]: 对于核范围空间，只需要固定数量的查询就足够了

    For Kernel Range Spaces a Constant Number of Queries Are Sufficient. (arXiv:2306.16516v1 [cs.CG])

    [http://arxiv.org/abs/2306.16516](http://arxiv.org/abs/2306.16516)

    对于核范围空间，引入了ε-覆盖的概念，用于处理不确定或不精确的数据分析任务。

    

    我们引入了核范围空间的ε-覆盖概念。核范围空间涉及一个点集X⊂R^d和由固定核函数（例如高斯核函数K(p,·)=exp(-||p-·||^2)）定义的查询空间。对于大小为n的点集X，查询返回一个值向量Rp∈R^n，其中第i个坐标(Rp)_i=K(p,x_i)，其中x_i∈X。ε-覆盖是点集Q⊂R^d的子集，对于任意p∈R^d，存在q∈Q使得||(Rp-Rq)/n||_1≤ε。这是Haussler在组合范围空间（例如由球查询定义的点集子集）中ε-覆盖概念的平滑模拟，其中得到的向量Rp是{0,1}^n而不是[0,1]^n。这些范围空间的核版本出现在数据分析任务中，其中坐标可能是不确定或不精确的，因此希望在范围查询中添加一些灵活性。

    We introduce the notion of an $\varepsilon$-cover for a kernel range space. A kernel range space concerns a set of points $X \subset \mathbb{R}^d$ and the space of all queries by a fixed kernel (e.g., a Gaussian kernel $K(p,\cdot) = \exp(-\|p-\cdot\|^2)$). For a point set $X$ of size $n$, a query returns a vector of values $R_p \in \mathbb{R}^n$, where the $i$th coordinate $(R_p)_i = K(p,x_i)$ for $x_i \in X$. An $\varepsilon$-cover is a subset of points $Q \subset \mathbb{R}^d$ so for any $p \in \mathbb{R}^d$ that $\frac{1}{n} \|R_p R_q\|_1\leq \varepsilon$ for some $q \in Q$. This is a smooth analog of Haussler's notion of $\varepsilon$-covers for combinatorial range spaces (e.g., defined by subsets of points within a ball query) where the resulting vectors $R_p$ are in $\{0,1\}^n$ instead of $[0,1]^n$. The kernel versions of these range spaces show up in data analysis tasks where the coordinates may be uncertain or imprecise, and hence one wishes to add some flexibility in the not
    
[^11]: 三次迭代的$(1-d)$-WL测试可以区分$d$维点云的非等距变换. (arXiv:2303.12853v1 [cs.LG])

    Three iterations of $(1-d)$-WL test distinguish non isometric clouds of $d$-dimensional points. (arXiv:2303.12853v1 [cs.LG])

    [http://arxiv.org/abs/2303.12853](http://arxiv.org/abs/2303.12853)

    本文研究了WL测试在点云中的应用，结果发现三次迭代的$(d-1)$-WL测试可以区分$d$维欧几里得空间中的点云，且只需要一次迭代的$d$-WL测试就可以达到完整性。

    

    Weisfeiler-Lehman (WL)测试是一个检查图同构的基本迭代算法。它被观察到是几种图神经网络体系结构设计的基础，这些网络的能力和性能可以用这个测试的表示能力来理解。受最近机器学习应用于涉及三维物体的数据集的发展启发，我们研究了当WL测试对完整的距离图表示的欧几里得点云是“完整的”时，它何时能够识别出任意一个任意点云.我们的主要结果是，$(d-1)$-维WL测试可以区分$d$维欧几里得空间中的点云，任何$d\ge 2$都可以，而且只需要进行三次测试。我们的结果对于$d=2,3$是紧的。我们还观察到$d$维WL测试只需要进行一次迭代就可以达到完整性。

    The Weisfeiler--Lehman (WL) test is a fundamental iterative algorithm for checking isomorphism of graphs. It has also been observed that it underlies the design of several graph neural network architectures, whose capabilities and performance can be understood in terms of the expressive power of this test. Motivated by recent developments in machine learning applications to datasets involving three-dimensional objects, we study when the WL test is {\em complete} for clouds of euclidean points represented by complete distance graphs, i.e., when it can distinguish, up to isometry, any arbitrary such cloud.  Our main result states that the $(d-1)$-dimensional WL test is complete for point clouds in $d$-dimensional Euclidean space, for any $d\ge 2$, and that only three iterations of the test suffice. Our result is tight for $d = 2, 3$. We also observe that the $d$-dimensional WL test only requires one iteration to achieve completeness.
    
[^12]: 二阶条件梯度滑动

    Second-order Conditional Gradient Sliding. (arXiv:2002.08907v3 [math.OC] UPDATED)

    [http://arxiv.org/abs/2002.08907](http://arxiv.org/abs/2002.08907)

    提出了一种二阶条件梯度滑动（SOCGS）算法，可以高效解决约束二次凸优化问题，并在有限次线性收敛迭代后二次收敛于原始间隙。

    

    当需要高精度解决问题时，约束二阶凸优化算法是首选，因为它们具有局部二次收敛性。这些算法在每次迭代时需要解决一个约束二次子问题。我们提出了\emph{二阶条件梯度滑动}（SOCGS）算法，它使用一种无投影算法来近似解决约束二次子问题。当可行域是一个多面体时，该算法在有限次线性收敛迭代后二次收敛于原始间隙。进入二次收敛阶段后，SOCGS算法需通过$\mathcal{O}(\log(\log 1/\varepsilon))$次一阶和Hessian正交调用以及$\mathcal{O}(\log (1/\varepsilon) \log(\log1/\varepsilon))$次线性最小化正交调用来实现$\varepsilon$-最优解。当可行域只能通过线性优化正交调用高效访问时，此算法非常有用。

    Constrained second-order convex optimization algorithms are the method of choice when a high accuracy solution to a problem is needed, due to their local quadratic convergence. These algorithms require the solution of a constrained quadratic subproblem at every iteration. We present the \emph{Second-Order Conditional Gradient Sliding} (SOCGS) algorithm, which uses a projection-free algorithm to solve the constrained quadratic subproblems inexactly. When the feasible region is a polytope the algorithm converges quadratically in primal gap after a finite number of linearly convergent iterations. Once in the quadratic regime the SOCGS algorithm requires $\mathcal{O}(\log(\log 1/\varepsilon))$ first-order and Hessian oracle calls and $\mathcal{O}(\log (1/\varepsilon) \log(\log1/\varepsilon))$ linear minimization oracle calls to achieve an $\varepsilon$-optimal solution. This algorithm is useful when the feasible region can only be accessed efficiently through a linear optimization oracle, 
    


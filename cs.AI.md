# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Algorithmic Collusion by Large Language Models](https://arxiv.org/abs/2404.00806) | 大型语言模型的算法定价代理在寡头市场环境中自主勾结，对消费者利益有害，其说明书中的短语变化可能增加勾结。 |
| [^2] | [How Secure Are Large Language Models (LLMs) for Navigation in Urban Environments?](https://arxiv.org/abs/2402.09546) | 本文首次研究了基于大型语言模型的导航系统在城市环境中的安全漏洞，并提出了一种新颖的NPS Attack方法，该方法通过添加后缀来操纵导航模型，导致不正确的行为。该研究对自动驾驶、物流和紧急服务等领域具有重要意义。 |
| [^3] | [Function Aligned Regression: A Method Explicitly Learns Functional Derivatives from Data](https://arxiv.org/abs/2402.06104) | 该论文提出了一种名为FAR的方法，通过捕捉函数导数来更好、更高效地拟合底层真实函数。在合成数据集和八个真实世界任务中证明了该方法的有效性。 |
| [^4] | [Learning Formal Specifications from Membership and Preference Queries.](http://arxiv.org/abs/2307.10434) | 该论文提出了一种新的框架，通过请求成员标签和成对偏好来扩展主动规范学习，提高学习形式规范的灵活性。在两个不同领域的实验中，结果表明通过学习成员和偏好的组合可以稳定和方便地识别规范。 |
| [^5] | [MDI+: A Flexible Random Forest-Based Feature Importance Framework.](http://arxiv.org/abs/2307.01932) | MDI+是一种灵活的基于随机森林的特征重要性框架，通过替换线性回归模型和度量，利用正则化的广义线性模型和更适合数据结构的度量来推广MDI。此外，MDI+还引入了其他特征来减轻决策树对加法或平滑模型的已知偏差。 |
| [^6] | [Improving Multi-task Learning via Seeking Task-based Flat Regions.](http://arxiv.org/abs/2211.13723) | 通过寻找基于任务的平坦区域，可以改进多任务学习并提高模型性能，但需要正确使用正则化技术以避免次优解。 |

# 详细

[^1]: 大型语言模型的算法勾结

    Algorithmic Collusion by Large Language Models

    [https://arxiv.org/abs/2404.00806](https://arxiv.org/abs/2404.00806)

    大型语言模型的算法定价代理在寡头市场环境中自主勾结，对消费者利益有害，其说明书中的短语变化可能增加勾结。

    

    arXiv:2404.00806v1 公告类型:交叉摘要:算法定价的兴起引起了对算法勾结的担忧。我们对基于大型语言模型（LLMs）特别是GPT-4的算法定价代理进行实验。我们发现：（1）基于LLM的代理在定价任务上表现出色，（2）基于LLM的定价代理在寡头市场环境中自主勾结，损害消费者利益，（3）LLM说明书中看似无害短语("提示")的变化可能会增加勾结。这些结果也适用于拍卖设置。我们的发现强调了有关算法定价的反垄断监管的必要性，并发现了基于LLM的定价代理所面临的监管挑战。

    arXiv:2404.00806v1 Announce Type: cross  Abstract: The rise of algorithmic pricing raises concerns of algorithmic collusion. We conduct experiments with algorithmic pricing agents based on Large Language Models (LLMs), and specifically GPT-4. We find that (1) LLM-based agents are adept at pricing tasks, (2) LLM-based pricing agents autonomously collude in oligopoly settings to the detriment of consumers, and (3) variation in seemingly innocuous phrases in LLM instructions ("prompts") may increase collusion. These results extend to auction settings. Our findings underscore the need for antitrust regulation regarding algorithmic pricing, and uncover regulatory challenges unique to LLM-based pricing agents.
    
[^2]: 大型语言模型（LLMs）在城市环境中导航时有多安全？

    How Secure Are Large Language Models (LLMs) for Navigation in Urban Environments?

    [https://arxiv.org/abs/2402.09546](https://arxiv.org/abs/2402.09546)

    本文首次研究了基于大型语言模型的导航系统在城市环境中的安全漏洞，并提出了一种新颖的NPS Attack方法，该方法通过添加后缀来操纵导航模型，导致不正确的行为。该研究对自动驾驶、物流和紧急服务等领域具有重要意义。

    

    在机器人和自动化领域，基于大型语言模型（LLMs）的导航系统最近展示了令人印象深刻的性能。然而，这些系统的安全性方面受到的关注相对较少。本文在城市户外环境中首次探索了LLM-based导航模型的漏洞，这是一个关键领域，因为该技术广泛应用于自动驾驶、物流和紧急服务。具体地，我们引入了一种新颖的Navigational Prompt Suffix (NPS) Attack，通过将梯度导出的后缀添加到原始导航提示，操纵LLM-based导航模型，从而导致不正确的行为。我们对基于LLMs的导航模型进行了全面的实验，该模型采用各种LLMs进行推理。我们的结果来自Touchdown和Map2Seq街景数据集，在few-shot学习和fine-tuning配置下进行实验，结果证明了NPS Attack的有效性。

    arXiv:2402.09546v1 Announce Type: cross  Abstract: In the field of robotics and automation, navigation systems based on Large Language Models (LLMs) have recently shown impressive performance. However, the security aspects of these systems have received relatively less attention. This paper pioneers the exploration of vulnerabilities in LLM-based navigation models in urban outdoor environments, a critical area given the technology's widespread application in autonomous driving, logistics, and emergency services. Specifically, we introduce a novel Navigational Prompt Suffix (NPS) Attack that manipulates LLM-based navigation models by appending gradient-derived suffixes to the original navigational prompt, leading to incorrect actions. We conducted comprehensive experiments on an LLMs-based navigation model that employs various LLMs for reasoning. Our results, derived from the Touchdown and Map2Seq street-view datasets under both few-shot learning and fine-tuning configurations, demonstr
    
[^3]: 功能对齐回归：一种从数据中明确学习函数导数的方法

    Function Aligned Regression: A Method Explicitly Learns Functional Derivatives from Data

    [https://arxiv.org/abs/2402.06104](https://arxiv.org/abs/2402.06104)

    该论文提出了一种名为FAR的方法，通过捕捉函数导数来更好、更高效地拟合底层真实函数。在合成数据集和八个真实世界任务中证明了该方法的有效性。

    

    回归是机器学习中的一个基本任务，在过去几十年中引起了广泛关注。传统的回归方法主要通过使用损失函数来将模型预测与每个个体数据样本的真实值对齐，然而，我们发现这种方法可能导致在不同样本之间关系的预测不够优化。近期的研究工作引入了标签相似性信息来改进回归方法，但在完全捕捉底层真实函数的复杂性方面仍存在明显的差距。在本文中，我们提出了FAR（功能对齐回归）作为一种更好、更高效的解决方案，通过捕捉函数导数来拟合底层真实函数。我们在两个合成数据集和六个领域的八个大规模真实世界任务中验证了该方法的有效性。

    Regression is a fundamental task in machine learning that has garnered extensive attention over the past decades. The conventional approach for regression involves employing loss functions that primarily concentrate on aligning model prediction with the ground truth for each individual data sample, which, as we show, can result in sub-optimal prediction of the relationships between the different samples. Recent research endeavors have introduced novel perspectives by incorporating label similarity information to regression. However, a notable gap persists in these approaches when it comes to fully capturing the intricacies of the underlying ground truth function. In this work, we propose FAR (Function Aligned Regression) as a arguably better and more efficient solution to fit the underlying function of ground truth by capturing functional derivatives. We demonstrate the effectiveness of the proposed method practically on 2 synthetic datasets and on 8 extensive real-world tasks from 6 b
    
[^4]: 从成员和偏好查询中学习形式规范

    Learning Formal Specifications from Membership and Preference Queries. (arXiv:2307.10434v1 [cs.FL])

    [http://arxiv.org/abs/2307.10434](http://arxiv.org/abs/2307.10434)

    该论文提出了一种新的框架，通过请求成员标签和成对偏好来扩展主动规范学习，提高学习形式规范的灵活性。在两个不同领域的实验中，结果表明通过学习成员和偏好的组合可以稳定和方便地识别规范。

    

    主动学习是一种研究广泛的学习形式规范的方法，例如自动机。在这项工作中，我们通过提出一种新颖的框架，将主动规范学习扩展到请求组合成员标签和成对偏好（对成员标签的一种流行替代方式）。成对偏好和成员标签的组合允许更灵活的主动规范学习方法，它先前仅依赖成员标签。我们将我们的框架应用于两个不同的领域，证明了我们方法的广泛性。我们的结果表明，从两种模式学习可以通过成员和偏好来稳健和方便地识别规范。

    Active learning is a well-studied approach to learning formal specifications, such as automata. In this work, we extend active specification learning by proposing a novel framework that strategically requests a combination of membership labels and pair-wise preferences, a popular alternative to membership labels. The combination of pair-wise preferences and membership labels allows for a more flexible approach to active specification learning, which previously relied on membership labels only. We instantiate our framework in two different domains, demonstrating the generality of our approach. Our results suggest that learning from both modalities allows us to robustly and conveniently identify specifications via membership and preferences.
    
[^5]: MDI+:一种灵活的基于随机森林的特征重要性框架

    MDI+: A Flexible Random Forest-Based Feature Importance Framework. (arXiv:2307.01932v1 [stat.ME])

    [http://arxiv.org/abs/2307.01932](http://arxiv.org/abs/2307.01932)

    MDI+是一种灵活的基于随机森林的特征重要性框架，通过替换线性回归模型和度量，利用正则化的广义线性模型和更适合数据结构的度量来推广MDI。此外，MDI+还引入了其他特征来减轻决策树对加法或平滑模型的已知偏差。

    

    以不纯度减少的平均值(MDI)是随机森林(RF)中一种流行的特征重要性评估方法。我们展示了在RF中每个树的特征$X_k$的MDI等价于响应变量在决策树集合上的线性回归的未归一化$R^2$值。我们利用这种解释提出了一种灵活的特征重要性框架MDI+，MDI+通过允许分析人员将线性回归模型和$R^2$度量替换为正则化的广义线性模型(GLM)和更适合给定数据结构的度量来推广MDI。此外，MDI+还引入了其他特征来减轻决策树对加法或平滑模型的已知偏差。我们进一步提供了关于如何基于可预测性、可计算性和稳定性框架选择适当的GLM和度量的指导，以进行真实数据科学研究。大量基于数据的模拟结果显示，MDI+在性能上显著优于传统的MDI。

    Mean decrease in impurity (MDI) is a popular feature importance measure for random forests (RFs). We show that the MDI for a feature $X_k$ in each tree in an RF is equivalent to the unnormalized $R^2$ value in a linear regression of the response on the collection of decision stumps that split on $X_k$. We use this interpretation to propose a flexible feature importance framework called MDI+. Specifically, MDI+ generalizes MDI by allowing the analyst to replace the linear regression model and $R^2$ metric with regularized generalized linear models (GLMs) and metrics better suited for the given data structure. Moreover, MDI+ incorporates additional features to mitigate known biases of decision trees against additive or smooth models. We further provide guidance on how practitioners can choose an appropriate GLM and metric based upon the Predictability, Computability, Stability framework for veridical data science. Extensive data-inspired simulations show that MDI+ significantly outperfor
    
[^6]: 通过寻找基于任务的平坦区域来改进多任务学习

    Improving Multi-task Learning via Seeking Task-based Flat Regions. (arXiv:2211.13723v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.13723](http://arxiv.org/abs/2211.13723)

    通过寻找基于任务的平坦区域，可以改进多任务学习并提高模型性能，但需要正确使用正则化技术以避免次优解。

    

    多任务学习（MTL）是一种广泛使用且强大的学习范式，用于训练深度神经网络，可以通过单个骨干学习多个目标。与单独训练任务相比，MTL显着降低了计算成本，提高了数据效率，并通过利用任务之间的知识来潜在地提高模型性能。因此，它已经被应用于各种应用领域，从计算机视觉到自然语言处理和语音识别。其中，MTL的一个新兴研究方向集中在操纵任务梯度以推导出对所有任务有益的最终梯度下降方向。尽管在许多基准测试上取得了令人印象深刻的结果，但是在实际问题上直接应用这些方法而不使用适当的正则化技术可能会导致次优解。特别是，标准训练在训练数据上最小化经验损失，很容易遭受过拟合问题。

    Multi-Task Learning (MTL) is a widely-used and powerful learning paradigm for training deep neural networks that allows learning more than one objective by a single backbone. Compared to training tasks separately, MTL significantly reduces computational costs, improves data efficiency, and potentially enhances model performance by leveraging knowledge across tasks. Hence, it has been adopted in a variety of applications, ranging from computer vision to natural language processing and speech recognition. Among them, there is an emerging line of work in MTL that focuses on manipulating the task gradient to derive an ultimate gradient descent direction to benefit all tasks. Despite achieving impressive results on many benchmarks, directly applying these approaches without using appropriate regularization techniques might lead to suboptimal solutions on real-world problems. In particular, standard training that minimizes the empirical loss on the training data can easily suffer from overfi
    


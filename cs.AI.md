# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Semantic Augmentation in Images using Language](https://arxiv.org/abs/2404.02353) | 深度学习模型需要大规模标记数据集，本文提出利用生成图像增强数据集以改进模型跨领域泛化能力。 |
| [^2] | [Algorithmic Collusion by Large Language Models](https://arxiv.org/abs/2404.00806) | 大型语言模型的算法定价代理在寡头市场环境中自主勾结，对消费者利益有害，其说明书中的短语变化可能增加勾结。 |
| [^3] | [A minimal coalition logic](https://arxiv.org/abs/2403.14704) | 提出了一个基于一般并发博弈模型的最小联盟逻辑，不具备传统模型中的独立性、序列性和确定性假设，展示了其完备性并与传统模型进行了比较 |
| [^4] | [Inconsistency Handling in Prioritized Databases with Universal Constraints: Complexity Analysis and Links with Active Integrity Constraints.](http://arxiv.org/abs/2306.03523) | 本文研究解决了具有全局约束的不一致数据库的修复和查询问题，通过对称差分修复并指定首选修复行动，扩展了现有的最优修复概念，并且研究了修复概念的计算属性，同时澄清了与主动完整性约束框架中引入的修复概念之间的关系。 |

# 详细

[^1]: 利用语言在图像中进行语义增强

    Semantic Augmentation in Images using Language

    [https://arxiv.org/abs/2404.02353](https://arxiv.org/abs/2404.02353)

    深度学习模型需要大规模标记数据集，本文提出利用生成图像增强数据集以改进模型跨领域泛化能力。

    

    深度学习模型需要非常庞大的标记数据集进行监督学习，缺乏这些数据集会导致过拟合并限制其泛化到现实世界示例的能力。最近扩散模型的进展使得能够基于文本输入生成逼真的图像。利用用于训练这些扩散模型的大规模数据集，我们提出一种利用生成的图像来增强现有数据集的技术。本文探讨了各种有效数据增强策略，以提高深度学习模型的跨领域泛化能力。

    arXiv:2404.02353v1 Announce Type: cross  Abstract: Deep Learning models are incredibly data-hungry and require very large labeled datasets for supervised learning. As a consequence, these models often suffer from overfitting, limiting their ability to generalize to real-world examples. Recent advancements in diffusion models have enabled the generation of photorealistic images based on textual inputs. Leveraging the substantial datasets used to train these diffusion models, we propose a technique to utilize generated images to augment existing datasets. This paper explores various strategies for effective data augmentation to improve the out-of-domain generalization capabilities of deep learning models.
    
[^2]: 大型语言模型的算法勾结

    Algorithmic Collusion by Large Language Models

    [https://arxiv.org/abs/2404.00806](https://arxiv.org/abs/2404.00806)

    大型语言模型的算法定价代理在寡头市场环境中自主勾结，对消费者利益有害，其说明书中的短语变化可能增加勾结。

    

    arXiv:2404.00806v1 公告类型:交叉摘要:算法定价的兴起引起了对算法勾结的担忧。我们对基于大型语言模型（LLMs）特别是GPT-4的算法定价代理进行实验。我们发现：（1）基于LLM的代理在定价任务上表现出色，（2）基于LLM的定价代理在寡头市场环境中自主勾结，损害消费者利益，（3）LLM说明书中看似无害短语("提示")的变化可能会增加勾结。这些结果也适用于拍卖设置。我们的发现强调了有关算法定价的反垄断监管的必要性，并发现了基于LLM的定价代理所面临的监管挑战。

    arXiv:2404.00806v1 Announce Type: cross  Abstract: The rise of algorithmic pricing raises concerns of algorithmic collusion. We conduct experiments with algorithmic pricing agents based on Large Language Models (LLMs), and specifically GPT-4. We find that (1) LLM-based agents are adept at pricing tasks, (2) LLM-based pricing agents autonomously collude in oligopoly settings to the detriment of consumers, and (3) variation in seemingly innocuous phrases in LLM instructions ("prompts") may increase collusion. These results extend to auction settings. Our findings underscore the need for antitrust regulation regarding algorithmic pricing, and uncover regulatory challenges unique to LLM-based pricing agents.
    
[^3]: 一个最小联盟逻辑

    A minimal coalition logic

    [https://arxiv.org/abs/2403.14704](https://arxiv.org/abs/2403.14704)

    提出了一个基于一般并发博弈模型的最小联盟逻辑，不具备传统模型中的独立性、序列性和确定性假设，展示了其完备性并与传统模型进行了比较

    

    联盟逻辑是战略推理研究中的一个中心逻辑。本文首先指出联盟逻辑模型，即并发博弈模型，存在三个过于强的假设。其一是代理的独立性；即，两个不同联盟的两个可用联合动作的合并总是可以合并成两个联盟的联盟。其二是序列性；即，联盟总是有可用的联合动作。其三是确定性，即，大联盟的合作动作总是有唯一结果。其次，我们提出了一个基于一般并发博弈模型的联盟逻辑，该模型不具备这三个假设。我们展示了这一逻辑的完备性，并与联盟逻辑进行了详细比较。在战略推理的背景下，这一逻辑似乎是最小的。

    arXiv:2403.14704v1 Announce Type: cross  Abstract: Coalition logic is a central logic in strategic reasoning studies. In this paper, we first argue that Coalition Logic models, concurrent game models, have three too-strong assumptions. The first one is the independence of agents; that is, the merge of two available joint actions of two disjoint coalitions is always available for the union of the two coalitions. The second one is seriality; that is, coalitions always have available joint actions. The third one is determinism, that is, the grand coalition's joint actions always have a unique outcome. Second, we present a coalition logic based on general concurrent game models, which do not have the three assumptions. We show the completeness of this logic and compare it with Coalition Logic in detail. This logic seems minimal in the context of strategic reasoning.
    
[^4]: 具有全局约束的优先数据库中的不一致性处理：复杂度分析和与主动完整性约束的联系(arXiv:2306.03523v1 [cs.DB])

    Inconsistency Handling in Prioritized Databases with Universal Constraints: Complexity Analysis and Links with Active Integrity Constraints. (arXiv:2306.03523v1 [cs.DB])

    [http://arxiv.org/abs/2306.03523](http://arxiv.org/abs/2306.03523)

    本文研究解决了具有全局约束的不一致数据库的修复和查询问题，通过对称差分修复并指定首选修复行动，扩展了现有的最优修复概念，并且研究了修复概念的计算属性，同时澄清了与主动完整性约束框架中引入的修复概念之间的关系。

    

    本文重新审视了带有全局约束的不一致数据库的修复和查询问题。采用对称差分修复，即通过删除和添加事实来恢复一致性，并假设通过对（否定）事实的二元优先关系来指定首选修复行动。我们的第一个贡献是展示如何适当地将现有的最优修复概念（仅对基于事实删除的简单拒绝约束和修复定义）扩展到我们更丰富的设置中。接下来，我们研究了所得到的修复概念的计算属性，特别是修复检查和容忍不一致查询的数据复杂性。最后，我们澄清了优先数据库的最优修复与在主动完整性约束框架中引入的修复概念之间的关系。特别地，我们表明在我们的设置中的帕累托最优修复对应于 founded、grounded 和 just。

    This paper revisits the problem of repairing and querying inconsistent databases equipped with universal constraints. We adopt symmetric difference repairs, in which both deletions and additions of facts can be used to restore consistency, and suppose that preferred repair actions are specified via a binary priority relation over (negated) facts. Our first contribution is to show how existing notions of optimal repairs, defined for simpler denial constraints and repairs solely based on fact deletion, can be suitably extended to our richer setting. We next study the computational properties of the resulting repair notions, in particular, the data complexity of repair checking and inconsistency-tolerant query answering. Finally, we clarify the relationship between optimal repairs of prioritized databases and repair notions introduced in the framework of active integrity constraints. In particular, we show that Pareto-optimal repairs in our setting correspond to founded, grounded and just
    


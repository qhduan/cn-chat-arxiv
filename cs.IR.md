# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Conversational Financial Information Retrieval Model (ConFIRM).](http://arxiv.org/abs/2310.13001) | ConFIRM是一种会话式金融信息检索模型，通过合成金融领域特定问答对和评估参数微调方法，实现了超过90%的准确性，为金融对话系统提供了数据高效的解决方案。 |
| [^2] | [Adaptive Preferential Attached kNN Graph With Distribution-Awareness.](http://arxiv.org/abs/2308.02442) | 本文提出了一种名为paNNG的算法，它结合了自适应kNN和基于分布的图构建。通过包含分布信息，paNNG能够有效提升模糊样本的性能，并实现更好的准确性和泛化能力。 |
| [^3] | [(Un)likelihood Training for Interpretable Embedding.](http://arxiv.org/abs/2207.00282) | 该论文提出了两种新的训练方法：可能性训练和不可能性训练，以解释嵌入向量背后的语义并解决标签稀疏问题。这些方法在图像和视频分类、检索和生成任务中表现出色，提高了学习嵌入的可解释性。 |
| [^4] | [Reinforcement Re-ranking with 2D Grid-based Recommendation Panels.](http://arxiv.org/abs/2204.04954) | 该论文提出了一种名为Panel-MDP的新型模型，通过采用强化学习策略，以用户喜好为导向，能够有效解决网格面板排列物品的问题，提高用户体验。 |

# 详细

[^1]: 会话式金融信息检索模型（ConFIRM）

    Conversational Financial Information Retrieval Model (ConFIRM). (arXiv:2310.13001v1 [cs.IR])

    [http://arxiv.org/abs/2310.13001](http://arxiv.org/abs/2310.13001)

    ConFIRM是一种会话式金融信息检索模型，通过合成金融领域特定问答对和评估参数微调方法，实现了超过90%的准确性，为金融对话系统提供了数据高效的解决方案。

    

    随着大型语言模型（LLM）的指数级增长，利用它们在金融等专门领域的新兴特性具有探索的价值。然而，金融等受监管领域具有独特的约束条件，需要具备针对该领域的优化框架。我们提出了ConFIRM，一种基于LLM的会话式金融信息检索模型，用于查询意图分类和知识库标记。ConFIRM包括两个模块：1）一种合成金融领域特定问答对的方法，以及2）评估参数高效的微调方法来进行查询分类任务。我们生成了一个包含4000多个样本的数据集，并在单独的测试集上评估了准确性。ConFIRM实现了超过90%的准确性，这对于符合监管要求至关重要。ConFIRM提供了一种数据高效的解决方案，用于提取金融对话系统的精确查询意图。

    With the exponential growth in large language models (LLMs), leveraging their emergent properties for specialized domains like finance merits exploration. However, regulated fields such as finance pose unique constraints, requiring domain-optimized frameworks. We present ConFIRM, an LLM-based conversational financial information retrieval model tailored for query intent classification and knowledge base labeling.  ConFIRM comprises two modules:  1) a method to synthesize finance domain-specific question-answer pairs, and  2) evaluation of parameter efficient fine-tuning approaches for the query classification task. We generate a dataset of over 4000 samples, assessing accuracy on a separate test set.  ConFIRM achieved over 90% accuracy, essential for regulatory compliance. ConFIRM provides a data-efficient solution to extract precise query intent for financial dialog systems.
    
[^2]: 使用分布感知的自适应优先级附加kNN图

    Adaptive Preferential Attached kNN Graph With Distribution-Awareness. (arXiv:2308.02442v1 [cs.LG])

    [http://arxiv.org/abs/2308.02442](http://arxiv.org/abs/2308.02442)

    本文提出了一种名为paNNG的算法，它结合了自适应kNN和基于分布的图构建。通过包含分布信息，paNNG能够有效提升模糊样本的性能，并实现更好的准确性和泛化能力。

    

    基于图的kNN算法因其简单性和有效性在机器学习任务中广受欢迎。然而，传统的kNN图对于k值的固定依赖可能会影响其性能，特别是在涉及复杂数据分布的情况下。此外，与其他分类模型类似，决策边界上存在的模糊样本常常是一个挑战，因为它们更容易被错误分类。为了解决这些问题，我们提出了优先级附加k-最近邻图（paNNG），它将自适应的kNN与基于分布的图构建相结合。通过结合分布信息，paNNG可以显著提高模糊样本的性能，通过“拉”它们回到原始类别，从而实现改进的整体准确性和泛化能力。通过在多样化的基准数据集上进行严格评估，paNNG的性能超越了现有算法，展示了它的优越性。

    Graph-based kNN algorithms have garnered widespread popularity for machine learning tasks, due to their simplicity and effectiveness. However, the conventional kNN graph's reliance on a fixed value of k can hinder its performance, especially in scenarios involving complex data distributions. Moreover, like other classification models, the presence of ambiguous samples along decision boundaries often presents a challenge, as they are more prone to incorrect classification. To address these issues, we propose the Preferential Attached k-Nearest Neighbors Graph (paNNG), which combines adaptive kNN with distribution-based graph construction. By incorporating distribution information, paNNG can significantly improve performance for ambiguous samples by "pulling" them towards their original classes and hence enable enhanced overall accuracy and generalization capability. Through rigorous evaluations on diverse benchmark datasets, paNNG outperforms state-of-the-art algorithms, showcasing its 
    
[^3]: 可解释嵌入的(不)可能性训练

    (Un)likelihood Training for Interpretable Embedding. (arXiv:2207.00282v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2207.00282](http://arxiv.org/abs/2207.00282)

    该论文提出了两种新的训练方法：可能性训练和不可能性训练，以解释嵌入向量背后的语义并解决标签稀疏问题。这些方法在图像和视频分类、检索和生成任务中表现出色，提高了学习嵌入的可解释性。

    

    跨模态表示学习已成为弥合文本和视觉数据语义差距的新常态。然而，在连续潜在空间中学习模态不可知表示经常被视为黑匣子数据驱动的训练过程。深度表示学习的有效性严重依赖于训练数据的质量和规模。对于视频表示学习，要完整地标注视频内容的数据集是高度困难甚至不可能的。这些问题，黑匣子训练和数据集偏差，使得解释性较差和结果难以预测，难以在视频理解方面进行实际应用。在本文中，我们提出了两种新的训练方法，可能性和不可能性函数，以展示嵌入背后的语义，并解决训练中的标签稀疏问题。可能性训练旨在通过学习数据分布来解释嵌入向量的语义，而不可能性训练则强调正负对之间的差异。我们将所提出的方法应用于各种任务，包括图像和视频分类、检索和生成，并展示了它们在提高学到的嵌入的可解释性以及在基准数据集上实现竞争性性能方面的有效性。

    Cross-modal representation learning has become a new normal for bridging the semantic gap between text and visual data. Learning modality agnostic representations in a continuous latent space, however, is often treated as a black-box data-driven training process. It is well-known that the effectiveness of representation learning depends heavily on the quality and scale of training data. For video representation learning, having a complete set of labels that annotate the full spectrum of video content for training is highly difficult if not impossible. These issues, black-box training and dataset bias, make representation learning practically challenging to be deployed for video understanding due to unexplainable and unpredictable results. In this paper, we propose two novel training objectives, likelihood and unlikelihood functions, to unroll semantics behind embeddings while addressing the label sparsity problem in training. The likelihood training aims to interpret semantics of embed
    
[^4]: 基于2D网格推荐面板的强化再排序

    Reinforcement Re-ranking with 2D Grid-based Recommendation Panels. (arXiv:2204.04954v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2204.04954](http://arxiv.org/abs/2204.04954)

    该论文提出了一种名为Panel-MDP的新型模型，通过采用强化学习策略，以用户喜好为导向，能够有效解决网格面板排列物品的问题，提高用户体验。

    

    现代推荐系统通常作为一个流式的单维排序列表呈现物品。近年来，在电子商务中有一种趋势，即推荐的物品以二维网格面板的形式组织，用户可以在竖直和水平方向上查看物品。在网格形式的结果面板中呈现物品对于推荐系统提出了新的挑战，因为现有模型都是设计用于输出序列列表，而网格面板中的插槽没有明确的顺序。直接将物品排名转换为网格（例如，预定义插槽的顺序）忽略了网格面板上用户特定的行为模式，并且不可避免地影响用户体验。为了解决这个问题，我们提出了一种新的马尔可夫决策过程（MDP），用于在推荐系统的最终再排序阶段中放置物品到二维网格结果面板中。该模型被称为Panel-MDP，它以早期阶段的初始物品排序为输入。然后，模型将以用户喜好为导向，采用强化学习策略来决定如何排列物品。

    Modern recommender systems usually present items as a streaming, one-dimensional ranking list. Recently there is a trend in e-commerce that the recommended items are organized grid-based panels with two dimensions where users can view the items in both vertical and horizontal directions. Presenting items in grid-based result panels poses new challenges to recommender systems because existing models are all designed to output sequential lists while the slots in a grid-based panel have no explicit order. Directly converting the item rankings into grids (e.g., pre-defining an order on the slots) overlooks the user-specific behavioral patterns on grid-based panels and inevitably hurts the user experiences. To address this issue, we propose a novel Markov decision process (MDP) to place the items in 2D grid-based result panels at the final re-ranking stage of the recommender systems. The model, referred to as Panel-MDP, takes an initial item ranking from the early stages as the input. Then,
    


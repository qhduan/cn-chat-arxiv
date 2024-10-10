# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Decoy Effect In Search Interaction: Understanding User Behavior and Measuring System Vulnerability](https://arxiv.org/abs/2403.18462) | 该研究探讨了诱饵效应对用户搜索交互的影响，提出了新的衡量IR系统脆弱性的方法，并介绍了DEJA-VU指标来评估系统对诱饵效应的易感性。 |
| [^2] | [One Backpropagation in Two Tower Recommendation Models](https://arxiv.org/abs/2403.18227) | 该论文提出了一种在两塔推荐模型中使用单次反向传播更新策略的方法，挑战了现有算法中平等对待用户和物品的假设。 |
| [^3] | [Conversational Financial Information Retrieval Model (ConFIRM).](http://arxiv.org/abs/2310.13001) | ConFIRM是一种会话式金融信息检索模型，通过合成金融领域特定问答对和评估参数微调方法，实现了超过90%的准确性，为金融对话系统提供了数据高效的解决方案。 |
| [^4] | [SR-PredictAO: Session-based Recommendation with High-Capability Predictor Add-On.](http://arxiv.org/abs/2309.12218) | SR-PredictAO是一种基于会话的推荐系统，通过引入高能力预测器模块，解决了现有模型中低能力预测器模块的问题，可以在存在随机用户行为的情况下预测用户的下一个动作。 |
| [^5] | [Invariant representation learning for sequential recommendation.](http://arxiv.org/abs/2308.11728) | 本论文介绍了一种名为Irl4Rec的新颖序列推荐框架，利用不变表示学习和考虑虚假关系，提高了推荐准确性。该框架在比较分析和消融研究中都表现出了优越性能。 |

# 详细

[^1]: 搜索交互中的诱饵效应：理解用户行为和测量系统脆弱性

    Decoy Effect In Search Interaction: Understanding User Behavior and Measuring System Vulnerability

    [https://arxiv.org/abs/2403.18462](https://arxiv.org/abs/2403.18462)

    该研究探讨了诱饵效应对用户搜索交互的影响，提出了新的衡量IR系统脆弱性的方法，并介绍了DEJA-VU指标来评估系统对诱饵效应的易感性。

    

    本研究考察了诱饵效应对用户搜索交互的影响，以及衡量信息检索（IR）系统对这种影响的脆弱性的方法。它探讨了诱饵结果如何改变用户在搜索引擎结果页面上的交互，关注点击概率、浏览时间和感知文档有用性等指标。通过分析来自多个数据集的用户交互日志，研究表明诱饵结果显著影响用户行为和感知。此外，研究还调查了不同任务难度和用户知识水平如何修改诱饵效应的影响，发现更容易的任务和较低的知识水平会导致更高的目标文档参与度。在IR系统评估方面，研究引入了DEJA-VU指标来评估系统对诱饵效应的易感性，并在特定检索任务上进行测试。结果显示在系统上存在差异

    arXiv:2403.18462v1 Announce Type: new  Abstract: This study examines the decoy effect's underexplored influence on user search interactions and methods for measuring information retrieval (IR) systems' vulnerability to this effect. It explores how decoy results alter users' interactions on search engine result pages, focusing on metrics like click-through likelihood, browsing time, and perceived document usefulness. By analyzing user interaction logs from multiple datasets, the study demonstrates that decoy results significantly affect users' behavior and perceptions. Furthermore, it investigates how different levels of task difficulty and user knowledge modify the decoy effect's impact, finding that easier tasks and lower knowledge levels lead to higher engagement with target documents. In terms of IR system evaluation, the study introduces the DEJA-VU metric to assess systems' susceptibility to the decoy effect, testing it on specific retrieval tasks. The results show differences in 
    
[^2]: 两塔推荐模型中的单次反向传播

    One Backpropagation in Two Tower Recommendation Models

    [https://arxiv.org/abs/2403.18227](https://arxiv.org/abs/2403.18227)

    该论文提出了一种在两塔推荐模型中使用单次反向传播更新策略的方法，挑战了现有算法中平等对待用户和物品的假设。

    

    最近几年，已经看到为了减轻信息过载而开发两塔推荐模型的广泛研究。这种模型中可以识别出四个构建模块，分别是用户-物品编码、负采样、损失计算和反向传播更新。据我们所知，现有算法仅研究了前三个模块，却忽略了反向传播模块。他们都采用某种形式的双反向传播策略，基于一个隐含的假设，即在训练阶段平等对待用户和物品。在本文中，我们挑战了这种平等训练假设，并提出了一种新颖的单次反向传播更新策略，这种策略保留了物品编码塔的正常梯度反向传播，但削减了用户编码塔的反向传播。相反，我们提出了一种移动聚合更新策略来更新每个训练周期中的用户编码。

    arXiv:2403.18227v1 Announce Type: new  Abstract: Recent years have witnessed extensive researches on developing two tower recommendation models for relieving information overload. Four building modules can be identified in such models, namely, user-item encoding, negative sampling, loss computing and back-propagation updating. To the best of our knowledge, existing algorithms have researched only on the first three modules, yet neglecting the backpropagation module. They all adopt a kind of two backpropagation strategy, which are based on an implicit assumption of equally treating users and items in the training phase. In this paper, we challenge such an equal training assumption and propose a novel one backpropagation updating strategy, which keeps the normal gradient backpropagation for the item encoding tower, but cuts off the backpropagation for the user encoding tower. Instead, we propose a moving-aggregation updating strategy to update a user encoding in each training epoch. Exce
    
[^3]: 会话式金融信息检索模型（ConFIRM）

    Conversational Financial Information Retrieval Model (ConFIRM). (arXiv:2310.13001v1 [cs.IR])

    [http://arxiv.org/abs/2310.13001](http://arxiv.org/abs/2310.13001)

    ConFIRM是一种会话式金融信息检索模型，通过合成金融领域特定问答对和评估参数微调方法，实现了超过90%的准确性，为金融对话系统提供了数据高效的解决方案。

    

    随着大型语言模型（LLM）的指数级增长，利用它们在金融等专门领域的新兴特性具有探索的价值。然而，金融等受监管领域具有独特的约束条件，需要具备针对该领域的优化框架。我们提出了ConFIRM，一种基于LLM的会话式金融信息检索模型，用于查询意图分类和知识库标记。ConFIRM包括两个模块：1）一种合成金融领域特定问答对的方法，以及2）评估参数高效的微调方法来进行查询分类任务。我们生成了一个包含4000多个样本的数据集，并在单独的测试集上评估了准确性。ConFIRM实现了超过90%的准确性，这对于符合监管要求至关重要。ConFIRM提供了一种数据高效的解决方案，用于提取金融对话系统的精确查询意图。

    With the exponential growth in large language models (LLMs), leveraging their emergent properties for specialized domains like finance merits exploration. However, regulated fields such as finance pose unique constraints, requiring domain-optimized frameworks. We present ConFIRM, an LLM-based conversational financial information retrieval model tailored for query intent classification and knowledge base labeling.  ConFIRM comprises two modules:  1) a method to synthesize finance domain-specific question-answer pairs, and  2) evaluation of parameter efficient fine-tuning approaches for the query classification task. We generate a dataset of over 4000 samples, assessing accuracy on a separate test set.  ConFIRM achieved over 90% accuracy, essential for regulatory compliance. ConFIRM provides a data-efficient solution to extract precise query intent for financial dialog systems.
    
[^4]: SR-PredictAO: 具有高能力预测器附加件的基于会话的推荐系统

    SR-PredictAO: Session-based Recommendation with High-Capability Predictor Add-On. (arXiv:2309.12218v1 [cs.IR])

    [http://arxiv.org/abs/2309.12218](http://arxiv.org/abs/2309.12218)

    SR-PredictAO是一种基于会话的推荐系统，通过引入高能力预测器模块，解决了现有模型中低能力预测器模块的问题，可以在存在随机用户行为的情况下预测用户的下一个动作。

    

    基于会话的推荐系统旨在通过仅基于单个会话中的信息来预测用户的下一个项目点击，即使在存在某些随机用户行为的情况下，这是一个复杂的问题。这个复杂的问题需要一个高能力的预测用户下一个动作的模型。大多数（如果不是全部）现有模型遵循编码器-预测器范式，在这个范式中所有的研究都集中在如何广泛优化编码器模块，但它们忽视了如何优化预测器模块。在本文中，我们发现了现有模型中低能力预测器模块存在的关键问题。受此启发，我们提出了一种新颖的框架称为\emph{\underline{S}ession-based \underline{R}ecommendation with \underline{Pred}ictor \underline{A}dd-\underline{O}n} (SR-PredictAO)。在这个框架中，我们提出了一个高能力的预测器模块，可以减轻随机用户行为对预测的影响。值得一提的是，

    Session-based recommendation, aiming at making the prediction of the user's next item click based on the information in a single session only even in the presence of some random user's behavior, is a complex problem. This complex problem requires a high-capability model of predicting the user's next action. Most (if not all) existing models follow the encoder-predictor paradigm where all studies focus on how to optimize the encoder module extensively in the paradigm but they ignore how to optimize the predictor module. In this paper, we discover the existing critical issue of the low-capability predictor module among existing models. Motivated by this, we propose a novel framework called \emph{\underline{S}ession-based \underline{R}ecommendation with \underline{Pred}ictor \underline{A}dd-\underline{O}n} (SR-PredictAO). In this framework, we propose a high-capability predictor module which could alleviate the effect of random user's behavior for prediction. It is worth mentioning that t
    
[^5]: 序列推荐的不变表示学习

    Invariant representation learning for sequential recommendation. (arXiv:2308.11728v1 [cs.IR])

    [http://arxiv.org/abs/2308.11728](http://arxiv.org/abs/2308.11728)

    本论文介绍了一种名为Irl4Rec的新颖序列推荐框架，利用不变表示学习和考虑虚假关系，提高了推荐准确性。该框架在比较分析和消融研究中都表现出了优越性能。

    

    序列推荐涉及根据用户的历史物品序列自动推荐下一个物品。虽然大多数先前的研究采用RNN或transformer方法从物品序列中获取信息，为每个用户-物品对生成概率，并推荐前几个物品，但这些方法通常忽视了虚假关系带来的挑战。本文特别解决了这些虚假关系问题。我们介绍了一个新颖的序列推荐框架称为Irl4Rec。该框架利用不变表示学习，并在模型训练过程中考虑了虚假变量和调整变量之间的关系，有助于识别虚假关系。比较分析表明，我们的框架优于三种典型方法，凸显了我们模型的有效性。此外，消融研究进一步证明了我们的模型在检测虚假关系中的关键作用。

    Sequential recommendation involves automatically recommending the next item to users based on their historical item sequence. While most prior research employs RNN or transformer methods to glean information from the item sequence-generating probabilities for each user-item pair and recommending the top items, these approaches often overlook the challenge posed by spurious relationships. This paper specifically addresses these spurious relations. We introduce a novel sequential recommendation framework named Irl4Rec. This framework harnesses invariant learning and employs a new objective that factors in the relationship between spurious variables and adjustment variables during model training. This approach aids in identifying spurious relations. Comparative analyses reveal that our framework outperforms three typical methods, underscoring the effectiveness of our model. Moreover, an ablation study further demonstrates the critical role our model plays in detecting spurious relations.
    


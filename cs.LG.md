# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Wisdom of Committee: Distilling from Foundation Model to SpecializedApplication Model](https://arxiv.org/abs/2402.14035) | 将基础模型的知识转移到专用应用模型中存在挑战，提出了通过创建教学委员会来应对这些挑战。 |
| [^2] | [Group-Sparse Matrix Factorization for Transfer Learning of Word Embeddings](https://arxiv.org/abs/2104.08928) | 提出了一种基于群稀疏矩阵分解的方法，用于在新领域进行词嵌入的传递学习，以解决不同领域单词含义差异的挑战。 |
| [^3] | [Algebraic and Statistical Properties of the Ordinary Least Squares Interpolator.](http://arxiv.org/abs/2309.15769) | 本文研究了普通最小二乘插值器在高维环境中的代数和统计属性，并为最小l2范数OLS插值器提供了基本结果。这些结果对理解OLS插值器的泛化能力具有重要意义。 |

# 详细

[^1]: 委员会的智慧：从基础模型到专用应用模型的提取

    Wisdom of Committee: Distilling from Foundation Model to SpecializedApplication Model

    [https://arxiv.org/abs/2402.14035](https://arxiv.org/abs/2402.14035)

    将基础模型的知识转移到专用应用模型中存在挑战，提出了通过创建教学委员会来应对这些挑战。

    

    最近基础模型的进展在各种任务上取得了令人印象深刻的性能，与此同时，为特定应用，从业者们一直在开发专门的应用模型。为了享受这两种模型的好处，一个自然的路径是将基础模型中的知识转移到专用应用模型中，后者通常更有效地提供服务。知识蒸馏的技术可以在这里应用，其中应用模型学会模仿基础模型。然而，专用应用模型和基础模型在容量上存在实质性差距，采用不同的架构，使用来自不同模态的不同输入特征，并在不同的分布上进行优化。模型特征上的这些差异导致了蒸馏方法面临重大挑战。在这项工作中，我们提出创建一个教学委员会，包括基础模型和专用应用模型。

    arXiv:2402.14035v1 Announce Type: cross  Abstract: Recent advancements in foundation models have yielded impressive performance across a wide range of tasks. Meanwhile, for specific applications, practitioners have been developing specialized application models. To enjoy the benefits of both kinds of models, one natural path is to transfer the knowledge in foundation models into specialized application models, which are generally more efficient for serving. Techniques from knowledge distillation may be applied here, where the application model learns to mimic the foundation model. However, specialized application models and foundation models have substantial gaps in capacity, employing distinct architectures, using different input features from different modalities, and being optimized on different distributions. These differences in model characteristics lead to significant challenges for distillation methods. In this work, we propose creating a teaching committee comprising both foun
    
[^2]: 基于群稀疏矩阵分解的词嵌入传递学习

    Group-Sparse Matrix Factorization for Transfer Learning of Word Embeddings

    [https://arxiv.org/abs/2104.08928](https://arxiv.org/abs/2104.08928)

    提出了一种基于群稀疏矩阵分解的方法，用于在新领域进行词嵌入的传递学习，以解决不同领域单词含义差异的挑战。

    

    非结构化文本为许多领域的决策者提供了丰富的数据源，涵盖范围从零售中的产品评论到医疗保健中的护理记录。为了利用这些信息，通常会通过无监督学习算法（如矩阵分解）将单词转换为词嵌入——编码单词之间语义关系的向量。然而，从具有有限训练数据的新领域学习单词嵌入可能具有挑战性，因为在新领域中，单词的含义/用法可能不同，例如，“positive”一词通常具有正面情绪，但在医疗记录中往往具有负面情绪，因为它可能意味着患者检测呈阳性。在实践中，我们预计只有少量领域特定单词可能具有新含义。我们提出了一个直观的两阶段估计器，通过群稀疏惩罚来有效地传递学习领域特定的新含义。

    arXiv:2104.08928v3 Announce Type: replace-cross  Abstract: Unstructured text provides decision-makers with a rich data source in many domains, ranging from product reviews in retail to nursing notes in healthcare. To leverage this information, words are typically translated into word embeddings -- vectors that encode the semantic relationships between words -- through unsupervised learning algorithms such as matrix factorization. However, learning word embeddings from new domains with limited training data can be challenging, because the meaning/usage may be different in the new domain, e.g., the word ``positive'' typically has positive sentiment, but often has negative sentiment in medical notes since it may imply that a patient tested positive for a disease. In practice, we expect that only a small number of domain-specific words may have new meanings. We propose an intuitive two-stage estimator that exploits this structure via a group-sparse penalty to efficiently transfer learn dom
    
[^3]: 普通最小二乘插值器的代数和统计属性

    Algebraic and Statistical Properties of the Ordinary Least Squares Interpolator. (arXiv:2309.15769v1 [math.ST])

    [http://arxiv.org/abs/2309.15769](http://arxiv.org/abs/2309.15769)

    本文研究了普通最小二乘插值器在高维环境中的代数和统计属性，并为最小l2范数OLS插值器提供了基本结果。这些结果对理解OLS插值器的泛化能力具有重要意义。

    

    深度学习研究揭示了对超参数化统计模型的良性过拟合现象，近年来引起了重大的理论兴趣。鉴于其简单性和实用性，普通最小二乘（OLS）插值器已成为获得对这种现象基础洞察力的关键所在。尽管OLS在经典环境中的性质已经得到了很好的建立，但在高维环境中的行为还没有像岭回归或套索回归那样被探索得那么透彻，尽管近年来已取得了显著进展。我们通过为最小l2范数OLS插值器提供基本的代数和统计结果来贡献于这一日益增长的文献。特别地，我们提供了（i）留-k-out残差公式的高维代数等价物，（ii） Cochran公式，以及（iii）Frisch-Waugh-Lovell定理。这些结果有助于理解OLS插值器的泛化能力并具有实质性的影响。

    Deep learning research has uncovered the phenomenon of benign overfitting for over-parameterized statistical models, which has drawn significant theoretical interest in recent years. Given its simplicity and practicality, the ordinary least squares (OLS) interpolator has become essential to gain foundational insights into this phenomenon. While properties of OLS are well established in classical settings, its behavior in high-dimensional settings is less explored (unlike for ridge or lasso regression) though significant progress has been made of late. We contribute to this growing literature by providing fundamental algebraic and statistical results for the minimum $\ell_2$-norm OLS interpolator. In particular, we provide high-dimensional algebraic equivalents of (i) the leave-$k$-out residual formula, (ii) Cochran's formula, and (iii) the Frisch-Waugh-Lovell theorem. These results aid in understanding the OLS interpolator's ability to generalize and have substantive implications for c
    


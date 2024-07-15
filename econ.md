# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Impact of Equal Opportunity on Statistical Discrimination.](http://arxiv.org/abs/2310.04585) | 本文通过修改统计性歧视模型，考虑了由机器学习生成的可合同化信念，给监管者提供了一种超过肯定行动的工具，通过要求公司选取一个平衡不同群体真正阳性率的决策策略，实现机会平等来消除统计性歧视。 |
| [^2] | [Probabilistic Verification in Mechanism Design.](http://arxiv.org/abs/1908.05556) | 该论文介绍了机制设计中的概率验证模型，通过选择统计测试来验证代理人的声明，并判断是否有最佳测试筛选所有其他类型。这个方法能够利润最大化并不断插值解决验证的问题。 |

# 详细

[^1]: 机会平等对统计性歧视的影响

    The Impact of Equal Opportunity on Statistical Discrimination. (arXiv:2310.04585v1 [econ.TH])

    [http://arxiv.org/abs/2310.04585](http://arxiv.org/abs/2310.04585)

    本文通过修改统计性歧视模型，考虑了由机器学习生成的可合同化信念，给监管者提供了一种超过肯定行动的工具，通过要求公司选取一个平衡不同群体真正阳性率的决策策略，实现机会平等来消除统计性歧视。

    

    本文修改了Coate和Loury（1993）的经典统计性歧视模型，假设公司对个体未观察到的类别的信念是由机器学习生成的，因此是可合同化的。这扩展了监管者的工具箱，超出了像肯定行动这样的无信念规定。可合同化的信念使得要求公司选择一个决策策略，使得不同群体之间的真正阳性率相等（算法公平文献中所称的机会平等）成为可能。尽管肯定行动不一定能消除统计性歧视，但本文表明实施机会平等可以做到。

    I modify the canonical statistical discrimination model of Coate and Loury (1993) by assuming the firm's belief about an individual's unobserved class is machine learning-generated and, therefore, contractible. This expands the toolkit of a regulator beyond belief-free regulations like affirmative action. Contractible beliefs make it feasible to require the firm to select a decision policy that equalizes true positive rates across groups -- what the algorithmic fairness literature calls equal opportunity. While affirmative action does not necessarily end statistical discrimination, I show that imposing equal opportunity does.
    
[^2]: 机制设计中的概率验证

    Probabilistic Verification in Mechanism Design. (arXiv:1908.05556v2 [econ.TH] UPDATED)

    [http://arxiv.org/abs/1908.05556](http://arxiv.org/abs/1908.05556)

    该论文介绍了机制设计中的概率验证模型，通过选择统计测试来验证代理人的声明，并判断是否有最佳测试筛选所有其他类型。这个方法能够利润最大化并不断插值解决验证的问题。

    

    我们在机制设计背景下引入了一个概率验证模型。委托人选择一个统计测试来验证代理人的声明。代理人的真实类型决定了他可以通过每个测试的概率。我们刻画了每个类型是否有一个相关的测试，最好地筛选出所有其他类型，无论社会选择规则如何。如果这个条件成立，那么测试技术可以用一个易于处理的简化形式表示。我们使用这个简化形式来解决验证的利润最大化机制。随着验证的改进，解决方案从无验证解决方案到完全剩余提取不断插值。

    We introduce a model of probabilistic verification in a mechanism design setting. The principal selects a statistical test to verify the agent's claim. The agent's true type determines the probability with which he can pass each test. We characterize whether each type has an associated test that best screens out all other types, no matter the social choice rule. If this condition holds, then the testing technology can be represented in a tractable reduced form. We use this reduced form to solve for profit-maximizing mechanisms with verification. As verification improves, the solution continuously interpolates from the no-verification solution to full surplus extraction.
    


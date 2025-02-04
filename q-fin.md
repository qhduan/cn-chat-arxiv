# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Global universal approximation of functional input maps on weighted spaces.](http://arxiv.org/abs/2306.03303) | 本文提出了功能性输入神经网络，可以在带权重空间上完成全局函数逼近。这一方法适用于连续函数的推广，还可用于路径空间函数的逼近，同时也可以逼近线性函数签名。 |
| [^2] | [Axioms for Automated Market Makers: A Mathematical Framework in FinTech and Decentralized Finance.](http://arxiv.org/abs/2210.01227) | 本文提出了一个自动市场制造商（AMMs）的公理框架，通过对底层效用函数施加合理的公理，描述了资产互换规模和结果定价预测的特性，并定义了一种新的价格影响度量方法。分析表明，大多数现有的AMMs满足这些公理。此外，还提出了一种新的费用结构，使得AMM对交易拆分不敏感，并提出了一种具有良好分析特性并提供较大范围内无发散损失的新型AMM。 |

# 详细

[^1]: 带权重空间上功能性输入映射的全局普适逼近

    Global universal approximation of functional input maps on weighted spaces. (arXiv:2306.03303v1 [stat.ML])

    [http://arxiv.org/abs/2306.03303](http://arxiv.org/abs/2306.03303)

    本文提出了功能性输入神经网络，可以在带权重空间上完成全局函数逼近。这一方法适用于连续函数的推广，还可用于路径空间函数的逼近，同时也可以逼近线性函数签名。

    

    我们引入了所谓的功能性输入神经网络，定义在可能是无限维带权重空间上，其值也在可能是无限维的输出空间中。为此，我们使用一个加性族作为隐藏层映射，以及一个非线性激活函数应用于每个隐藏层。依靠带权重空间上的Stone-Weierstrass定理，我们可以证明连续函数的推广的全局普适逼近结果，超越了常规紧集逼近。这特别适用于通过功能性输入神经网络逼近（非先见之明的）路径空间函数。作为带权Stone-Weierstrass定理的进一步应用，我们证明了线性函数签名的全局普适逼近结果。我们还在这个设置中引入了高斯过程回归的观点，并展示了签名内核的再生核希尔伯特空间是某些高斯过程的Cameron-Martin空间。

    We introduce so-called functional input neural networks defined on a possibly infinite dimensional weighted space with values also in a possibly infinite dimensional output space. To this end, we use an additive family as hidden layer maps and a non-linear activation function applied to each hidden layer. Relying on Stone-Weierstrass theorems on weighted spaces, we can prove a global universal approximation result for generalizations of continuous functions going beyond the usual approximation on compact sets. This then applies in particular to approximation of (non-anticipative) path space functionals via functional input neural networks. As a further application of the weighted Stone-Weierstrass theorem we prove a global universal approximation result for linear functions of the signature. We also introduce the viewpoint of Gaussian process regression in this setting and show that the reproducing kernel Hilbert space of the signature kernels are Cameron-Martin spaces of certain Gauss
    
[^2]: 自动市场制造商的公理：金融科技和去中心化金融领域的数学框架

    Axioms for Automated Market Makers: A Mathematical Framework in FinTech and Decentralized Finance. (arXiv:2210.01227v2 [q-fin.MF] UPDATED)

    [http://arxiv.org/abs/2210.01227](http://arxiv.org/abs/2210.01227)

    本文提出了一个自动市场制造商（AMMs）的公理框架，通过对底层效用函数施加合理的公理，描述了资产互换规模和结果定价预测的特性，并定义了一种新的价格影响度量方法。分析表明，大多数现有的AMMs满足这些公理。此外，还提出了一种新的费用结构，使得AMM对交易拆分不敏感，并提出了一种具有良好分析特性并提供较大范围内无发散损失的新型AMM。

    

    在这项工作中，我们考虑了自动市场制造商（AMMs）的公理框架。通过对底层效用函数施加合理的公理，我们能够描述资产互换的规模和结果定价预测的特性。通过提供这些通用的AMM公理，我们定义了一种新的价格影响度量方法，可以用来量化不同构造之间的成本。我们分析了许多现有的AMMs，并证明了其中大多数满足我们的公理。我们还考虑了费用和发散损失的问题。在此过程中，我们提出了一种新的费用结构，使得AMM对交易拆分不敏感。最后，我们提出了一种具有良好分析特性并提供较大范围内无发散损失的新型AMM。

    Within this work we consider an axiomatic framework for Automated Market Makers (AMMs). By imposing reasonable axioms on the underlying utility function, we are able to characterize the properties of the swap size of the assets and of the resulting pricing oracle. In providing these general AMM axioms, we define a novel measure of price impacts that can be used to quantify those costs between different constructions. We have analyzed many existing AMMs and shown that the vast majority of them satisfy our axioms. We have also considered the question of fees and divergence loss. In doing so, we have proposed a new fee structure so as to make the AMM indifferent to transaction splitting. Finally, we have proposed a novel AMM that has nice analytical properties and provides a large range over which there is no divergence loss.
    


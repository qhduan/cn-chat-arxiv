# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Modeling Latent Selection with Structural Causal Models.](http://arxiv.org/abs/2401.06925) | 本文介绍了一种在结构因果模型中对潜在选择进行建模的方法，并展示了它如何帮助进行因果推理任务，包括处理选择偏差。 |
| [^2] | [Global universal approximation of functional input maps on weighted spaces.](http://arxiv.org/abs/2306.03303) | 本文提出了功能性输入神经网络，可以在带权重空间上完成全局函数逼近。这一方法适用于连续函数的推广，还可用于路径空间函数的逼近，同时也可以逼近线性函数签名。 |

# 详细

[^1]: 用结构因果模型对潜在选择进行建模

    Modeling Latent Selection with Structural Causal Models. (arXiv:2401.06925v1 [cs.AI])

    [http://arxiv.org/abs/2401.06925](http://arxiv.org/abs/2401.06925)

    本文介绍了一种在结构因果模型中对潜在选择进行建模的方法，并展示了它如何帮助进行因果推理任务，包括处理选择偏差。

    

    选择偏倚在现实世界的数据中是普遍存在的，如果不正确处理可能导致误导性结果。我们引入了对结构因果模型（SCMs）进行条件操作的方法，以从因果的角度对潜在选择进行建模。我们展示了条件操作将具有明确潜在选择机制的SCM转换为没有此类选择机制的SCM，这在一定程度上编码了根据原始SCM选择的亚总体的因果语义。此外，我们还展示了该条件操作保持SCMs的简洁性，无环性和线性性，并与边际化操作相符合。由于这些特性与边际化和干预结合起来，条件操作为在潜在细节已经去除的因果模型中进行因果推理任务提供了一个有价值的工具。我们通过例子演示了如何将因果推断的经典结果推广以包括选择偏倚。

    Selection bias is ubiquitous in real-world data, and can lead to misleading results if not dealt with properly. We introduce a conditioning operation on Structural Causal Models (SCMs) to model latent selection from a causal perspective. We show that the conditioning operation transforms an SCM with the presence of an explicit latent selection mechanism into an SCM without such selection mechanism, which partially encodes the causal semantics of the selected subpopulation according to the original SCM. Furthermore, we show that this conditioning operation preserves the simplicity, acyclicity, and linearity of SCMs, and commutes with marginalization. Thanks to these properties, combined with marginalization and intervention, the conditioning operation offers a valuable tool for conducting causal reasoning tasks within causal models where latent details have been abstracted away. We demonstrate by example how classical results of causal inference can be generalized to include selection b
    
[^2]: 带权重空间上功能性输入映射的全局普适逼近

    Global universal approximation of functional input maps on weighted spaces. (arXiv:2306.03303v1 [stat.ML])

    [http://arxiv.org/abs/2306.03303](http://arxiv.org/abs/2306.03303)

    本文提出了功能性输入神经网络，可以在带权重空间上完成全局函数逼近。这一方法适用于连续函数的推广，还可用于路径空间函数的逼近，同时也可以逼近线性函数签名。

    

    我们引入了所谓的功能性输入神经网络，定义在可能是无限维带权重空间上，其值也在可能是无限维的输出空间中。为此，我们使用一个加性族作为隐藏层映射，以及一个非线性激活函数应用于每个隐藏层。依靠带权重空间上的Stone-Weierstrass定理，我们可以证明连续函数的推广的全局普适逼近结果，超越了常规紧集逼近。这特别适用于通过功能性输入神经网络逼近（非先见之明的）路径空间函数。作为带权Stone-Weierstrass定理的进一步应用，我们证明了线性函数签名的全局普适逼近结果。我们还在这个设置中引入了高斯过程回归的观点，并展示了签名内核的再生核希尔伯特空间是某些高斯过程的Cameron-Martin空间。

    We introduce so-called functional input neural networks defined on a possibly infinite dimensional weighted space with values also in a possibly infinite dimensional output space. To this end, we use an additive family as hidden layer maps and a non-linear activation function applied to each hidden layer. Relying on Stone-Weierstrass theorems on weighted spaces, we can prove a global universal approximation result for generalizations of continuous functions going beyond the usual approximation on compact sets. This then applies in particular to approximation of (non-anticipative) path space functionals via functional input neural networks. As a further application of the weighted Stone-Weierstrass theorem we prove a global universal approximation result for linear functions of the signature. We also introduce the viewpoint of Gaussian process regression in this setting and show that the reproducing kernel Hilbert space of the signature kernels are Cameron-Martin spaces of certain Gauss
    


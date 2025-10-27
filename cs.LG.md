# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FlexLLM: A System for Co-Serving Large Language Model Inference and Parameter-Efficient Finetuning](https://arxiv.org/abs/2402.18789) | FlexLLM是第一个可以在同一迭代中共同提供推理和参数高效微调请求的系统，通过引入标记级微调机制实现共享GPU资源的高效利用 |
| [^2] | [Factor Fitting, Rank Allocation, and Partitioning in Multilevel Low Rank Matrices.](http://arxiv.org/abs/2310.19214) | 本文研究了多级低秩矩阵中的因子拟合、秩分配和分割问题，提出了相应的解决方法，并开发了一个开源软件包。 |
| [^3] | [Interpretability and Transparency-Driven Detection and Transformation of Textual Adversarial Examples (IT-DT).](http://arxiv.org/abs/2307.01225) | 通过提出的解释性和透明性驱动的检测与转换（IT-DT）框架，我们在检测和转换文本对抗示例方面注重解释性和透明性。这个框架利用了注意力图、集成梯度和模型反馈等技术，在检测阶段有助于识别对对抗性分类有贡献的显著特征和扰动词语，并在转换阶段使用预训练的嵌入和模型反馈来生成扰动词语的最佳替代，以将对抗性示例转换为正常示例。 |
| [^4] | [Regret Distribution in Stochastic Bandits: Optimal Trade-off between Expectation and Tail Risk.](http://arxiv.org/abs/2304.04341) | 该论文探讨了随机多臂赌博问题中，如何在期望和尾部风险之间做出最优权衡。提出了一种新的策略，能够实现最坏和实例相关的优异表现，并且能够最小化遗憾尾部概率。 |

# 详细

[^1]: FlexLLM：一种用于共同提供大型语言模型推理和参数高效微调的系统

    FlexLLM: A System for Co-Serving Large Language Model Inference and Parameter-Efficient Finetuning

    [https://arxiv.org/abs/2402.18789](https://arxiv.org/abs/2402.18789)

    FlexLLM是第一个可以在同一迭代中共同提供推理和参数高效微调请求的系统，通过引入标记级微调机制实现共享GPU资源的高效利用

    

    Parameter-efficient finetuning（PEFT）是一种广泛使用的技术，用于为不同任务调整大型语言模型。通常，服务提供商会为用户创建单独的系统，以执行PEFT模型微调和推理任务。这是因为现有系统无法处理包含推理和PEFT微调请求混合的工作负载。因此，共享的GPU资源利用不足，导致效率低下。为解决这一问题，我们提出了FlexLLM，这是第一个可以在同一迭代中为推理和参数高效微调请求提供服务的系统。我们的系统利用这两个任务的互补性质，并利用共享的GPU资源来共同运行它们，使用一种称为共同提供的方法。为实现这一目标，FlexLLM引入了一种新颖的标记级微调机制，将序列的微调计算分解为更小的标记级计算，并使用依赖并行化。

    arXiv:2402.18789v1 Announce Type: cross  Abstract: Parameter-efficient finetuning (PEFT) is a widely used technique to adapt large language models for different tasks. Service providers typically create separate systems for users to perform PEFT model finetuning and inference tasks. This is because existing systems cannot handle workloads that include a mix of inference and PEFT finetuning requests. As a result, shared GPU resources are underutilized, leading to inefficiencies. To address this problem, we present FlexLLM, the first system that can serve inference and parameter-efficient finetuning requests in the same iteration. Our system leverages the complementary nature of these two tasks and utilizes shared GPU resources to run them jointly, using a method called co-serving. To achieve this, FlexLLM introduces a novel token-level finetuning mechanism, which breaks down the finetuning computation of a sequence into smaller token-level computations and uses dependent parallelization
    
[^2]: 在多级低秩矩阵中进行因子拟合、秩分配和分割

    Factor Fitting, Rank Allocation, and Partitioning in Multilevel Low Rank Matrices. (arXiv:2310.19214v1 [stat.ML])

    [http://arxiv.org/abs/2310.19214](http://arxiv.org/abs/2310.19214)

    本文研究了多级低秩矩阵中的因子拟合、秩分配和分割问题，提出了相应的解决方法，并开发了一个开源软件包。

    

    我们考虑多级低秩（MLR）矩阵，定义为一系列矩阵的行和列的排列，每个矩阵都是前一个矩阵的块对角修正，所有块以因子形式给出低秩矩阵。MLR矩阵扩展了低秩矩阵的概念，但它们共享许多特性，例如所需总存储空间和矩阵向量乘法的复杂度。我们解决了用Frobenius范数拟合给定矩阵到MLR矩阵的三个问题。第一个问题是因子拟合，通过调整MLR矩阵的因子来解决。第二个问题是秩分配，在每个级别中选择块的秩，满足总秩的给定值，以保持MLR矩阵所需的总存储空间。最后一个问题是选择行和列的层次分割，以及秩和因子。本文附带了一个开源软件包，实现了所提出的方法。

    We consider multilevel low rank (MLR) matrices, defined as a row and column permutation of a sum of matrices, each one a block diagonal refinement of the previous one, with all blocks low rank given in factored form. MLR matrices extend low rank matrices but share many of their properties, such as the total storage required and complexity of matrix-vector multiplication. We address three problems that arise in fitting a given matrix by an MLR matrix in the Frobenius norm. The first problem is factor fitting, where we adjust the factors of the MLR matrix. The second is rank allocation, where we choose the ranks of the blocks in each level, subject to the total rank having a given value, which preserves the total storage needed for the MLR matrix. The final problem is to choose the hierarchical partition of rows and columns, along with the ranks and factors. This paper is accompanied by an open source package that implements the proposed methods.
    
[^3]: 解释性和透明性驱动的文本对抗示例的检测与转换（IT-DT）

    Interpretability and Transparency-Driven Detection and Transformation of Textual Adversarial Examples (IT-DT). (arXiv:2307.01225v1 [cs.CL])

    [http://arxiv.org/abs/2307.01225](http://arxiv.org/abs/2307.01225)

    通过提出的解释性和透明性驱动的检测与转换（IT-DT）框架，我们在检测和转换文本对抗示例方面注重解释性和透明性。这个框架利用了注意力图、集成梯度和模型反馈等技术，在检测阶段有助于识别对对抗性分类有贡献的显著特征和扰动词语，并在转换阶段使用预训练的嵌入和模型反馈来生成扰动词语的最佳替代，以将对抗性示例转换为正常示例。

    

    基于Transformer的文本分类器如BERT、Roberta、T5和GPT-3在自然语言处理方面展示了令人印象深刻的性能。然而，它们对于对抗性示例的脆弱性提出了安全风险。现有的防御方法缺乏解释性，很难理解对抗性分类并识别模型的漏洞。为了解决这个问题，我们提出了解释性和透明性驱动的检测与转换（IT-DT）框架。它专注于在检测和转换文本对抗示例时的解释性和透明性。IT-DT利用注意力图、集成梯度和模型反馈等技术进行解释性检测。这有助于识别对对抗性分类有贡献的显著特征和扰动词语。在转换阶段，IT-DT利用预训练的嵌入和模型反馈来生成扰动词语的最佳替代。通过找到合适的替换，我们的目标是将对抗性示例转换为正常示例。

    Transformer-based text classifiers like BERT, Roberta, T5, and GPT-3 have shown impressive performance in NLP. However, their vulnerability to adversarial examples poses a security risk. Existing defense methods lack interpretability, making it hard to understand adversarial classifications and identify model vulnerabilities. To address this, we propose the Interpretability and Transparency-Driven Detection and Transformation (IT-DT) framework. It focuses on interpretability and transparency in detecting and transforming textual adversarial examples. IT-DT utilizes techniques like attention maps, integrated gradients, and model feedback for interpretability during detection. This helps identify salient features and perturbed words contributing to adversarial classifications. In the transformation phase, IT-DT uses pre-trained embeddings and model feedback to generate optimal replacements for perturbed words. By finding suitable substitutions, we aim to convert adversarial examples into
    
[^4]: 随机赌博机中的遗憾分布：期望和尾部风险之间的最优权衡

    Regret Distribution in Stochastic Bandits: Optimal Trade-off between Expectation and Tail Risk. (arXiv:2304.04341v1 [stat.ML])

    [http://arxiv.org/abs/2304.04341](http://arxiv.org/abs/2304.04341)

    该论文探讨了随机多臂赌博问题中，如何在期望和尾部风险之间做出最优权衡。提出了一种新的策略，能够实现最坏和实例相关的优异表现，并且能够最小化遗憾尾部概率。

    

    本文研究了随机多臂赌博问题中，遗憾分布的期望和尾部风险之间的权衡问题。我们完全刻画了策略设计中三个期望性质之间的相互作用：最坏情况下的最优性，实例相关的一致性和轻尾风险。我们展示了期望遗憾的顺序如何影响遗憾尾部概率的衰减率，同时包括了最坏情况和实例相关的情况。我们提出了一种新的策略，以表征对于任何遗憾阈值的最优遗憾尾部概率。具体地，对于任何给定的$\alpha \in [1/2, 1)$和$\beta \in [0, \alpha]$，我们的策略可以实现平均期望遗憾$\tilde O(T^\alpha)$的最坏情况下$\alpha$-最优和期望遗憾$\tilde O(T^\beta)$的实例相关的$\beta$-一致性，并且享有一定的概率可以避免$\tilde O(T^\delta)$的遗憾($\delta \geq \alpha$在最坏情况下和$\delta \geq \beta$在实例相关的情况下)。

    We study the trade-off between expectation and tail risk for regret distribution in the stochastic multi-armed bandit problem. We fully characterize the interplay among three desired properties for policy design: worst-case optimality, instance-dependent consistency, and light-tailed risk. We show how the order of expected regret exactly affects the decaying rate of the regret tail probability for both the worst-case and instance-dependent scenario. A novel policy is proposed to characterize the optimal regret tail probability for any regret threshold. Concretely, for any given $\alpha\in[1/2, 1)$ and $\beta\in[0, \alpha]$, our policy achieves a worst-case expected regret of $\tilde O(T^\alpha)$ (we call it $\alpha$-optimal) and an instance-dependent expected regret of $\tilde O(T^\beta)$ (we call it $\beta$-consistent), while enjoys a probability of incurring an $\tilde O(T^\delta)$ regret ($\delta\geq\alpha$ in the worst-case scenario and $\delta\geq\beta$ in the instance-dependent s
    


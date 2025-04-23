# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [QuaCer-C: Quantitative Certification of Knowledge Comprehension in LLMs](https://arxiv.org/abs/2402.15929) | 本文提出了一种新颖的认证框架QuaCer-C，用于正式认证大型语言模型中知识理解的能力，证书定量化且包含高置信度的概率界限，研究发现，随着参数数量的增加，知识理解能力提高，Mistral模型在这一评估中表现不如其他模型。 |
| [^2] | [Multi-View Knowledge Distillation from Crowd Annotations for Out-of-Domain Generalization.](http://arxiv.org/abs/2212.09409) | 本文提出了一种新方法，通过聚合多个视图的众包注释来获取软标签，从而进行跨领域泛化。 |

# 详细

[^1]: QuaCer-C：大型语言模型中知识理解的定量认证

    QuaCer-C: Quantitative Certification of Knowledge Comprehension in LLMs

    [https://arxiv.org/abs/2402.15929](https://arxiv.org/abs/2402.15929)

    本文提出了一种新颖的认证框架QuaCer-C，用于正式认证大型语言模型中知识理解的能力，证书定量化且包含高置信度的概率界限，研究发现，随着参数数量的增加，知识理解能力提高，Mistral模型在这一评估中表现不如其他模型。

    

    大型语言模型（LLMs）在多个基准测试中展现出令人印象深刻的表现。然而，传统研究并未对LLMs的表现提供正式的保证。本文提出了一种新颖的LLM认证框架QuaCer-C，我们在此对知名LLMs的知识理解能力进行正式认证。我们的证书是定量的 - 它们包括对目标LLM在任何相关知识理解提示上给出正确答案的概率的高置信度紧密界限。我们针对Llama、Vicuna和Mistral LLMs的证书表明，知识理解能力随参数数量的增加而提高，并且Mistral模型在这一评估中表现不如其他模型。

    arXiv:2402.15929v1 Announce Type: new  Abstract: Large Language Models (LLMs) have demonstrated impressive performance on several benchmarks. However, traditional studies do not provide formal guarantees on the performance of LLMs. In this work, we propose a novel certification framework for LLM, QuaCer-C, wherein we formally certify the knowledge-comprehension capabilities of popular LLMs. Our certificates are quantitative - they consist of high-confidence, tight bounds on the probability that the target LLM gives the correct answer on any relevant knowledge comprehension prompt. Our certificates for the Llama, Vicuna, and Mistral LLMs indicate that the knowledge comprehension capability improves with an increase in the number of parameters and that the Mistral model is less performant than the rest in this evaluation.
    
[^2]: 从众包注释中进行多视角知识蒸馏以实现跨领域泛化

    Multi-View Knowledge Distillation from Crowd Annotations for Out-of-Domain Generalization. (arXiv:2212.09409v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.09409](http://arxiv.org/abs/2212.09409)

    本文提出了一种新方法，通过聚合多个视图的众包注释来获取软标签，从而进行跨领域泛化。

    

    在自然语言处理任务中选择有效的训练信号很困难：专家注释很昂贵，而众包注释可能不可靠。最近的NLP研究表明，从众包注释中获取标签分布的学习可以是有效的。然而，有很多获取这种分布的方法，任何一种方法的性能都可能因任务和可用众包注释量而波动，这使得事先不知道哪种分布最好。本文在领域外系统地分析了这个问题，并提出了通过聚合现有方法产生的分布来获取来自众包注释的软标签的新方法。特别地，我们建议通过温度缩放和找到它们的Jensen-Shannon中心来聚合众包注释的多个视图。

    Selecting an effective training signal for tasks in natural language processing is difficult: expert annotations are expensive, and crowd-sourced annotations may not be reliable. At the same time, recent work in NLP has demonstrated that learning from a distribution over labels acquired from crowd annotations can be effective. However, there are many ways to acquire such a distribution, and the performance allotted by any one method can fluctuate based on the task and the amount of available crowd annotations, making it difficult to know a priori which distribution is best. This paper systematically analyzes this in the out-of-domain setting, adding to the NLP literature which has focused on in-domain evaluation, and proposes new methods for acquiring soft-labels from crowd-annotations by aggregating the distributions produced by existing methods. In particular, we propose to aggregate multiple-views of crowd annotations via temperature scaling and finding their Jensen-Shannon centroid
    


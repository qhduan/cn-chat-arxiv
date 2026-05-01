# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Framework for Variational Inference of Lightweight Bayesian Neural Networks with Heteroscedastic Uncertainties](https://arxiv.org/abs/2402.14532) | 提出了一种新框架，通过将异方差Aleatoric和认知方差嵌入到学习BNN参数的方差中，改善了轻量级网络的预测性能。 |
| [^2] | [Junk DNA Hypothesis: A Task-Centric Angle of LLM Pre-trained Weights through Sparsity.](http://arxiv.org/abs/2310.02277) | 本文研究通过稀疏性分析LLM预训练权重的任务中心角度，挑战了传统对于权重中冗余性的观点，并提出了"垃圾DNA假设"。 |

# 详细

[^1]: 一种用于具有异方差不确定性的轻量级贝叶斯神经网络变分推断的框架

    A Framework for Variational Inference of Lightweight Bayesian Neural Networks with Heteroscedastic Uncertainties

    [https://arxiv.org/abs/2402.14532](https://arxiv.org/abs/2402.14532)

    提出了一种新框架，通过将异方差Aleatoric和认知方差嵌入到学习BNN参数的方差中，改善了轻量级网络的预测性能。

    

    从贝叶斯神经网络（BNN）中获得异方差预测不确定性对许多应用至关重要。通常，除了预测均值外，异方差Aleatoric不确定性作为BNN的输出进行学习，然而这样做可能需要向网络中添加更多可学习参数。在这项工作中，我们展示了异方差Aleatoric和认知方差均可以嵌入到学习BNN参数的方差中，从而提高轻量级网络的预测性能。通过将这种方法与矩传播方法相结合，我们引入了一个适用于轻量级BNNs的无需取样的变分推断相对简单的框架。

    arXiv:2402.14532v1 Announce Type: new  Abstract: Obtaining heteroscedastic predictive uncertainties from a Bayesian Neural Network (BNN) is vital to many applications. Often, heteroscedastic aleatoric uncertainties are learned as outputs of the BNN in addition to the predictive means, however doing so may necessitate adding more learnable parameters to the network. In this work, we demonstrate that both the heteroscedastic aleatoric and epistemic variance can be embedded into the variances of learned BNN parameters, improving predictive performance for lightweight networks. By complementing this approach with a moment propagation approach to inference, we introduce a relatively simple framework for sampling-free variational inference suitable for lightweight BNNs.
    
[^2]: "垃圾DNA假设：通过稀疏性对LLM预训练权重进行任务中心角度分析"

    Junk DNA Hypothesis: A Task-Centric Angle of LLM Pre-trained Weights through Sparsity. (arXiv:2310.02277v1 [cs.LG])

    [http://arxiv.org/abs/2310.02277](http://arxiv.org/abs/2310.02277)

    本文研究通过稀疏性分析LLM预训练权重的任务中心角度，挑战了传统对于权重中冗余性的观点，并提出了"垃圾DNA假设"。

    

    传统对"垃圾DNA"的概念长期以来与人类基因组中的非编码片段相关联，占其组成的大约98%。然而，最近的研究揭示了一些这些看似无功能的DNA序列在细胞过程中起到的关键作用。有趣的是，深度神经网络中的权重与人类基因中观察到的冗余性有着显著的相似性。人们认为，庞大模型中的权重包含了过多的冗余，可以在不影响性能的情况下去除。本文通过提出一个令人信服的反论来挑战这个传统观点。我们使用稀疏性作为一种工具，来独立而准确地量化预训练大语言模型(LLM)中低幅度权重的细微重要性，从下游任务中心的角度理解它们包含的知识。我们提出了支持我们深入研究的"垃圾DNA假设"。

    The traditional notion of "Junk DNA" has long been linked to non-coding segments within the human genome, constituting roughly 98% of its composition. However, recent research has unveiled the critical roles some of these seemingly non-functional DNA sequences play in cellular processes. Intriguingly, the weights within deep neural networks exhibit a remarkable similarity to the redundancy observed in human genes. It was believed that weights in gigantic models contained excessive redundancy, and could be removed without compromising performance. This paper challenges this conventional wisdom by presenting a compelling counter-argument. We employ sparsity as a tool to isolate and quantify the nuanced significance of low-magnitude weights in pre-trained large language models (LLMs). Our study demonstrates a strong correlation between these weight magnitudes and the knowledge they encapsulate, from a downstream task-centric angle. we raise the "Junk DNA Hypothesis" backed by our in-depth
    


# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Securing GNNs: Explanation-Based Identification of Backdoored Training Graphs](https://arxiv.org/abs/2403.18136) | 提出了一种基于解释的方法来识别GNN中的后门训练图，设计了七种新的度量指标以更有效地检测后门攻击，并且通过自适应攻击进行了方法评估。 |
| [^2] | [Multi-Level Feature Aggregation and Recursive Alignment Network for Real-Time Semantic Segmentation](https://arxiv.org/abs/2402.02286) | 该论文提出了一种在实时推理速度下实现高分割准确性的多级特征聚合和递归对齐网络。使用ResNet-18作为骨干，通过多级特征聚合模块和递归对齐模块来提高模型性能。 |

# 详细

[^1]: 保护GNN：基于解释的后门训练图识别

    Securing GNNs: Explanation-Based Identification of Backdoored Training Graphs

    [https://arxiv.org/abs/2403.18136](https://arxiv.org/abs/2403.18136)

    提出了一种基于解释的方法来识别GNN中的后门训练图，设计了七种新的度量指标以更有效地检测后门攻击，并且通过自适应攻击进行了方法评估。

    

    Graph Neural Networks (GNNs)已经在许多领域流行起来，但它们容易受到后门攻击，这可能会损害它们的性能和道德应用。检测这些攻击对于保持GNN分类任务的可靠性和安全性至关重要，但有效的检测技术并不多见。我们观察到，尽管图级解释能够提供一些有限的见解，但它们在检测后门触发器方面的有效性是不一致且不完整的。为弥补这一差距，我们提取并转换GNN解释机制的次要输出，设计了七种更有效地检测后门攻击的新度量。此外，我们还开发了一种自适应攻击来严格评估我们的方法。我们在多个基准数据集上测试了我们的方法，并检查其对各种攻击模型的有效性。我们的结果表明，我们的方法可以取得较高的效果。

    arXiv:2403.18136v1 Announce Type: cross  Abstract: Graph Neural Networks (GNNs) have gained popularity in numerous domains, yet they are vulnerable to backdoor attacks that can compromise their performance and ethical application. The detection of these attacks is crucial for maintaining the reliability and security of GNN classification tasks, but effective detection techniques are lacking. Following an initial investigation, we observed that while graph-level explanations can offer limited insights, their effectiveness in detecting backdoor triggers is inconsistent and incomplete. To bridge this gap, we extract and transform secondary outputs of GNN explanation mechanisms, designing seven novel metrics that more effectively detect backdoor attacks. Additionally, we develop an adaptive attack to rigorously evaluate our approach. We test our method on multiple benchmark datasets and examine its efficacy against various attack models. Our results show that our method can achieve high de
    
[^2]: 多级特征聚合和递归对齐网络用于实时语义分割

    Multi-Level Feature Aggregation and Recursive Alignment Network for Real-Time Semantic Segmentation

    [https://arxiv.org/abs/2402.02286](https://arxiv.org/abs/2402.02286)

    该论文提出了一种在实时推理速度下实现高分割准确性的多级特征聚合和递归对齐网络。使用ResNet-18作为骨干，通过多级特征聚合模块和递归对齐模块来提高模型性能。

    

    实时语义分割对于实际应用非常重要。然而，许多方法都着重于降低计算复杂性和模型大小，但同时牺牲了准确性。在一些场景下，如自主导航和驾驶员辅助系统，准确性和速度同样重要。为了解决这个问题，我们提出了一种新颖的多级特征聚合和递归对齐网络（MFARANet），旨在实现高分割准确性和实时推理速度。我们使用ResNet-18作为骨干来保证效率，并提出了三个核心组件来弥补浅骨干引起的模型容量减少。具体而言，我们首先设计多级特征聚合模块（MFAM），将编码器中的分层特征聚合到每个尺度，以便于后续的空间对齐和多尺度推理。然后，我们通过结合基于流的对齐来建立递归对齐模块（RAM）。

    Real-time semantic segmentation is a crucial research for real-world applications. However, many methods lay particular emphasis on reducing the computational complexity and model size, while largely sacrificing the accuracy. In some scenarios, such as autonomous navigation and driver assistance system, accuracy and speed are equally important. To tackle this problem, we propose a novel Multi-level Feature Aggregation and Recursive Alignment Network (MFARANet), aiming to achieve high segmentation accuracy at real-time inference speed. We employ ResNet-18 as the backbone to ensure efficiency, and propose three core components to compensate for the reduced model capacity due to the shallow backbone. Specifically, we first design Multi-level Feature Aggregation Module (MFAM) to aggregate the hierarchical features in the encoder to each scale to benefit subsequent spatial alignment and multi-scale inference. Then, we build Recursive Alignment Module (RAM) by combining the flow-based alignm
    


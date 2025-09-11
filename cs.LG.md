# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Transformer approach for Electricity Price Forecasting](https://arxiv.org/abs/2403.16108) | 这种独特的Transformer模型在电力价格预测中取得了更好的表现，为可靠和可持续的电力系统运行提供了有前景的解决方案。 |
| [^2] | [FedComLoc: Communication-Efficient Distributed Training of Sparse and Quantized Models](https://arxiv.org/abs/2403.09904) | FedComLoc利用Scaffnew算法的基础，引入了压缩和本地训练，显著降低了分布式训练中的通信开销。 |
| [^3] | [PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation](https://arxiv.org/abs/2402.04355) | PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。 |
| [^4] | [Adversarially Robust Signed Graph Contrastive Learning from Balance Augmentation.](http://arxiv.org/abs/2401.10590) | 本研究提出了一种名为BA-SGCL的鲁棒SGNN框架，通过结合图对比学习原则和平衡增强技术，解决了带符号图对抗性攻击中平衡相关信息不可逆的问题。 |
| [^5] | [Less is More: On the Feature Redundancy of Pretrained Models When Transferring to Few-shot Tasks.](http://arxiv.org/abs/2310.03843) | 预训练模型在少样本任务中的特征可以极其冗余，仅使用最重要的特征维度的1%就能够达到使用完整表示时的性能。 |
| [^6] | [Calibrating Transformers via Sparse Gaussian Processes.](http://arxiv.org/abs/2303.02444) | 提出了一种通过Sparse Gaussian Process attention (SGPA)来校准Transformer模型不确定性的方法。在文本、图像和图形的预测任务中，SGPA-based Transformers在预测准确性上表现出竞争力，并显著改善了内分布校准和外分布的鲁棒性和检测能力。 |
| [^7] | [FlexFringe: Modeling Software Behavior by Learning Probabilistic Automata.](http://arxiv.org/abs/2203.16331) | FlexFringe提供了高效的概率有限自动机学习方法，可用于建模软件行为。该方法在实践中通过实现改进的状态合并策略实现了显著性能提升，并且能够从软件日志中学习可解释的模型，用于异常检测。与基于神经网络的解决方案相比，学习更小更复杂的模型能够提高FlexFringe在异常检测中的性能。 |

# 详细

[^1]: 一种用于电力价格预测的Transformer方法

    A Transformer approach for Electricity Price Forecasting

    [https://arxiv.org/abs/2403.16108](https://arxiv.org/abs/2403.16108)

    这种独特的Transformer模型在电力价格预测中取得了更好的表现，为可靠和可持续的电力系统运行提供了有前景的解决方案。

    

    本文提出了一种使用纯Transformer模型进行电力价格预测（EPF）的新方法。与其他方法不同，没有使用其他递归网络结合注意力机制。因此，表明注意力层足以捕捉时间模式。该论文还通过使用开源EPF工具进行了对模型的公平比较，并提供了代码以增强EPF研究的可再现性和透明度。结果表明，Transformer模型优于传统方法，为可靠和可持续的电力系统运行提供了一种有希望的解决方案。

    arXiv:2403.16108v1 Announce Type: cross  Abstract: This paper presents a novel approach to electricity price forecasting (EPF) using a pure Transformer model. As opposed to other alternatives, no other recurrent network is used in combination to the attention mechanism. Hence, showing that the attention layer is enough for capturing the temporal patterns. The paper also provides fair comparison of the models using the open-source EPF toolbox and provide the code to enhance reproducibility and transparency in EPF research. The results show that the Transformer model outperforms traditional methods, offering a promising solution for reliable and sustainable power system operation.
    
[^2]: FedComLoc: 稀疏和量化模型的通信高效分布式训练

    FedComLoc: Communication-Efficient Distributed Training of Sparse and Quantized Models

    [https://arxiv.org/abs/2403.09904](https://arxiv.org/abs/2403.09904)

    FedComLoc利用Scaffnew算法的基础，引入了压缩和本地训练，显著降低了分布式训练中的通信开销。

    

    联邦学习（FL）由于其允许异构客户端在本地处理其私有数据并与中央服务器互动，同时尊重隐私的独特特点而受到越来越多的关注。我们的工作受到了创新的Scaffnew算法的启发，该算法在FL中大大推动了通信复杂性的降低。我们引入了FedComLoc（联邦压缩和本地训练），将实用且有效的压缩集成到Scaffnew中，以进一步增强通信效率。广泛的实验证明，使用流行的TopK压缩器和量化，它在大幅减少异构中的通信开销方面具有卓越的性能。

    arXiv:2403.09904v1 Announce Type: cross  Abstract: Federated Learning (FL) has garnered increasing attention due to its unique characteristic of allowing heterogeneous clients to process their private data locally and interact with a central server, while being respectful of privacy. A critical bottleneck in FL is the communication cost. A pivotal strategy to mitigate this burden is \emph{Local Training}, which involves running multiple local stochastic gradient descent iterations between communication phases. Our work is inspired by the innovative \emph{Scaffnew} algorithm, which has considerably advanced the reduction of communication complexity in FL. We introduce FedComLoc (Federated Compressed and Local Training), integrating practical and effective compression into \emph{Scaffnew} to further enhance communication efficiency. Extensive experiments, using the popular TopK compressor and quantization, demonstrate its prowess in substantially reducing communication overheads in heter
    
[^3]: PQMass: 使用概率质量估计的生成模型质量的概率评估

    PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation

    [https://arxiv.org/abs/2402.04355](https://arxiv.org/abs/2402.04355)

    PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。

    

    我们提出了一种全面的基于样本的方法来评估生成模型的质量。所提出的方法能够估计两个样本集合来自同一分布的概率，为评估单个生成模型的性能或比较在同一数据集上训练的多个竞争模型提供了一个统计上严格的方法。该比较可以通过将空间划分为非重叠的区域并比较每个区域中的数据样本数量来进行。该方法仅需要生成模型和测试数据的样本。它能够直接处理高维数据，无需降维。显著的是，该方法不依赖于关于真实分布密度的假设，并且不依赖于训练或拟合任何辅助模型。相反，它着重于近似计算密度的积分（概率质量）。

    We propose a comprehensive sample-based method for assessing the quality of generative models. The proposed approach enables the estimation of the probability that two sets of samples are drawn from the same distribution, providing a statistically rigorous method for assessing the performance of a single generative model or the comparison of multiple competing models trained on the same dataset. This comparison can be conducted by dividing the space into non-overlapping regions and comparing the number of data samples in each region. The method only requires samples from the generative model and the test data. It is capable of functioning directly on high-dimensional data, obviating the need for dimensionality reduction. Significantly, the proposed method does not depend on assumptions regarding the density of the true distribution, and it does not rely on training or fitting any auxiliary models. Instead, it focuses on approximating the integral of the density (probability mass) acros
    
[^4]: Adversarially Robust Signed Graph Contrastive Learning from Balance Augmentation（从平衡增强中提取对抗性鲁棒的带符号图对比学习）

    Adversarially Robust Signed Graph Contrastive Learning from Balance Augmentation. (arXiv:2401.10590v1 [cs.LG])

    [http://arxiv.org/abs/2401.10590](http://arxiv.org/abs/2401.10590)

    本研究提出了一种名为BA-SGCL的鲁棒SGNN框架，通过结合图对比学习原则和平衡增强技术，解决了带符号图对抗性攻击中平衡相关信息不可逆的问题。

    

    带符号图由边和符号组成，可以分为结构信息和平衡相关信息。现有的带符号图神经网络（SGNN）通常依赖于平衡相关信息来生成嵌入。然而，最近的对抗性攻击对平衡相关信息产生了不利影响。类似于结构学习可以恢复无符号图，通过改进被污染图的平衡度，可以将平衡学习应用于带符号图。然而，这种方法面临着“平衡相关信息的不可逆性”挑战-尽管平衡度得到改善，但恢复的边可能不是最初受到攻击影响的边，导致防御效果差。为了解决这个挑战，我们提出了一个鲁棒的SGNN框架，称为平衡增强带符号图对比学习（BA-SGCL），它将图对比学习原则与平衡增强相结合。

    Signed graphs consist of edges and signs, which can be separated into structural information and balance-related information, respectively. Existing signed graph neural networks (SGNNs) typically rely on balance-related information to generate embeddings. Nevertheless, the emergence of recent adversarial attacks has had a detrimental impact on the balance-related information. Similar to how structure learning can restore unsigned graphs, balance learning can be applied to signed graphs by improving the balance degree of the poisoned graph. However, this approach encounters the challenge "Irreversibility of Balance-related Information" - while the balance degree improves, the restored edges may not be the ones originally affected by attacks, resulting in poor defense effectiveness. To address this challenge, we propose a robust SGNN framework called Balance Augmented-Signed Graph Contrastive Learning (BA-SGCL), which combines Graph Contrastive Learning principles with balance augmentati
    
[^5]: Less is More: 关于预训练模型在少样本任务中特征冗余性的研究

    Less is More: On the Feature Redundancy of Pretrained Models When Transferring to Few-shot Tasks. (arXiv:2310.03843v1 [cs.CV])

    [http://arxiv.org/abs/2310.03843](http://arxiv.org/abs/2310.03843)

    预训练模型在少样本任务中的特征可以极其冗余，仅使用最重要的特征维度的1%就能够达到使用完整表示时的性能。

    

    将预训练模型应用于下游任务可以通过使用目标数据进行线性探测来实现，即对从预训练模型中提取的冻结特征进行训练线性分类器。由于预训练和下游数据集之间可能存在显著差异，我们可以询问是否所有预训练特征的维度对于给定的下游任务都是有用的。我们发现，在线性探测的情况下，当下游数据稀缺或少样本时，预训练特征可能极其冗余。对于一些情况，比如5类1样本任务，只使用最重要的特征维度的1%就能够达到使用完整表示时的性能。有趣的是，大部分特征只在少样本设置下是冗余的，在样本数增加时逐渐变得有用，这表明特征冗余可能是表征少样本转移问题的关键。我们给出了理论解释

    Transferring a pretrained model to a downstream task can be as easy as conducting linear probing with target data, that is, training a linear classifier upon frozen features extracted from the pretrained model. As there may exist significant gaps between pretraining and downstream datasets, one may ask whether all dimensions of the pretrained features are useful for a given downstream task. We show that, for linear probing, the pretrained features can be extremely redundant when the downstream data is scarce, or few-shot. For some cases such as 5-way 1-shot tasks, using only 1\% of the most important feature dimensions is able to recover the performance achieved by using the full representation. Interestingly, most dimensions are redundant only under few-shot settings and gradually become useful when the number of shots increases, suggesting that feature redundancy may be the key to characterizing the "few-shot" nature of few-shot transfer problems. We give a theoretical understanding 
    
[^6]: 通过稀疏高斯过程校准Transformer

    Calibrating Transformers via Sparse Gaussian Processes. (arXiv:2303.02444v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.02444](http://arxiv.org/abs/2303.02444)

    提出了一种通过Sparse Gaussian Process attention (SGPA)来校准Transformer模型不确定性的方法。在文本、图像和图形的预测任务中，SGPA-based Transformers在预测准确性上表现出竞争力，并显著改善了内分布校准和外分布的鲁棒性和检测能力。

    

    Transformer模型在自然语言处理、语音识别和计算机视觉等广泛应用中取得了巨大成功。将Transformer的成功扩展到安全关键领域需要准确估计的不确定性，这方面的研究较少。为了解决这个问题，我们提出了稀疏高斯过程注意力（SGPA），它直接在Transformer的多头自注意力块（MHA）的输出空间中进行贝叶斯推断，以校准其不确定性。它用一个有效的对称核替代了缩放点积操作，并使用稀疏高斯过程（SGP）技术来近似MHA输出的后验过程。经验上，在文本、图像和图形的一系列预测任务中，基于SGPA的Transformer模型实现了有竞争力的预测准确性，同时显著改善了内分布校准和外分布的鲁棒性和检测能力。

    Transformer models have achieved profound success in prediction tasks in a wide range of applications in natural language processing, speech recognition and computer vision. Extending Transformer's success to safety-critical domains requires calibrated uncertainty estimation which remains under-explored. To address this, we propose Sparse Gaussian Process attention (SGPA), which performs Bayesian inference directly in the output space of multi-head attention blocks (MHAs) in transformer to calibrate its uncertainty. It replaces the scaled dot-product operation with a valid symmetric kernel and uses sparse Gaussian processes (SGP) techniques to approximate the posterior processes of MHA outputs. Empirically, on a suite of prediction tasks on text, images and graphs, SGPA-based Transformers achieve competitive predictive accuracy, while noticeably improving both in-distribution calibration and out-of-distribution robustness and detection.
    
[^7]: FlexFringe:通过学习概率有限自动机来建模软件行为

    FlexFringe: Modeling Software Behavior by Learning Probabilistic Automata. (arXiv:2203.16331v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2203.16331](http://arxiv.org/abs/2203.16331)

    FlexFringe提供了高效的概率有限自动机学习方法，可用于建模软件行为。该方法在实践中通过实现改进的状态合并策略实现了显著性能提升，并且能够从软件日志中学习可解释的模型，用于异常检测。与基于神经网络的解决方案相比，学习更小更复杂的模型能够提高FlexFringe在异常检测中的性能。

    

    我们介绍了FlexFringe中可用的概率确定性有限自动机学习方法的高效实现。这些实现了众所周知的状态合并策略，包括几种修改以提高它们在实践中的性能。我们通过实验证明这些算法能够获得有竞争力的结果，并在默认实现上实现了显著的改进。我们还展示了如何使用FlexFringe从软件日志中学习可解释的模型，并将其用于异常检测。虽然这些模型较难解释，但我们展示了学习更小、更复杂的模型如何提高FlexFringe在异常检测中的性能，优于基于神经网络的现有解决方案。

    We present the efficient implementations of probabilistic deterministic finite automaton learning methods available in FlexFringe. These implement well-known strategies for state-merging including several modifications to improve their performance in practice. We show experimentally that these algorithms obtain competitive results and significant improvements over a default implementation. We also demonstrate how to use FlexFringe to learn interpretable models from software logs and use these for anomaly detection. Although less interpretable, we show that learning smaller more convoluted models improves the performance of FlexFringe on anomaly detection, outperforming an existing solution based on neural nets.
    


# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adversarial Fine-tuning of Compressed Neural Networks for Joint Improvement of Robustness and Efficiency](https://arxiv.org/abs/2403.09441) | 本研究探讨了对压缩神经网络进行对抗微调对提高鲁棒性和效率的影响。 |
| [^2] | [MATNet: Multi-Level Fusion Transformer-Based Model for Day-Ahead PV Generation Forecasting](https://arxiv.org/abs/2306.10356) | 提出了MATNet，结合了人工智能范式与光伏发电的物理先验知识，通过多级联合融合方法进行日前光伏发电预测 |
| [^3] | [Matrix Completion with Hypergraphs:Sharp Thresholds and Efficient Algorithms.](http://arxiv.org/abs/2401.08197) | 本论文研究了基于子采样矩阵条目、观察到的社交图和超图的矩阵补全问题。我们发现了一个尖锐阈值，可以精确补全评分矩阵。通过量化超图的“质量”函数，我们可以评估超图利用对样本概率的影响。通过开发高效算法，我们在高概率情况下成功地完成了矩阵补全任务。 |

# 详细

[^1]: 对压缩神经网络进行对抗微调，共同提高鲁棒性和效率

    Adversarial Fine-tuning of Compressed Neural Networks for Joint Improvement of Robustness and Efficiency

    [https://arxiv.org/abs/2403.09441](https://arxiv.org/abs/2403.09441)

    本研究探讨了对压缩神经网络进行对抗微调对提高鲁棒性和效率的影响。

    

    随着深度学习（DL）模型越来越多地融入我们的日常生活中，确保它们的安全性，使其对抗对抗性攻击具有鲁棒性变得越来越关键。我们在这项研究中探讨了两种不同的模型压缩方法 -- 结构化权重剪枝和量化对抗鲁棒性的影响。我们特别研究了对压缩模型进行微调的效果，并提出了一种同时提高鲁棒性和效率的方法。

    arXiv:2403.09441v1 Announce Type: new  Abstract: As deep learning (DL) models are increasingly being integrated into our everyday lives, ensuring their safety by making them robust against adversarial attacks has become increasingly critical. DL models have been found to be susceptible to adversarial attacks which can be achieved by introducing small, targeted perturbations to disrupt the input data. Adversarial training has been presented as a mitigation strategy which can result in more robust models. This adversarial robustness comes with additional computational costs required to design adversarial attacks during training. The two objectives -- adversarial robustness and computational efficiency -- then appear to be in conflict of each other. In this work, we explore the effects of two different model compression methods -- structured weight pruning and quantization -- on adversarial robustness. We specifically explore the effects of fine-tuning on compressed models, and present th
    
[^2]: MATNet: 多级融合变压器模型用于日前光伏发电预测

    MATNet: Multi-Level Fusion Transformer-Based Model for Day-Ahead PV Generation Forecasting

    [https://arxiv.org/abs/2306.10356](https://arxiv.org/abs/2306.10356)

    提出了MATNet，结合了人工智能范式与光伏发电的物理先验知识，通过多级联合融合方法进行日前光伏发电预测

    

    准确预测可再生能源发电对促进可再生能源整合到电力系统中至关重要。针对光伏单元，预测方法主要可分为基于物理和基于数据的策略两类，基于人工智能的模型提供了最先进的性能。然而，虽然这些基于人工智能的模型可以捕捉数据中的复杂模式和关系，但它们忽略了现象的潜在物理先验知识。因此，在本文中，我们提出了MATNet，一种新颖的基于自注意力变压器架构，用于多元多步日前光伏发电预测。它采用一种混合方法，将人工智能范式与基于物理的光伏发电的先验知识相结合。该模型通过多级联合融合方法输入历史光伏数据以及历史和预测天气数据。

    arXiv:2306.10356v2 Announce Type: replace-cross  Abstract: Accurate forecasting of renewable generation is crucial to facilitate the integration of RES into the power system. Focusing on PV units, forecasting methods can be divided into two main categories: physics-based and data-based strategies, with AI-based models providing state-of-the-art performance. However, while these AI-based models can capture complex patterns and relationships in the data, they ignore the underlying physical prior knowledge of the phenomenon. Therefore, in this paper we propose MATNet, a novel self-attention transformer-based architecture for multivariate multi-step day-ahead PV power generation forecasting. It consists of a hybrid approach that combines the AI paradigm with the prior physical knowledge of PV power generation of physics-based methods. The model is fed with historical PV data and historical and forecast weather data through a multi-level joint fusion approach. The effectiveness of the propo
    
[^3]: 基于超图的矩阵补全：尖锐阈值和高效算法

    Matrix Completion with Hypergraphs:Sharp Thresholds and Efficient Algorithms. (arXiv:2401.08197v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2401.08197](http://arxiv.org/abs/2401.08197)

    本论文研究了基于子采样矩阵条目、观察到的社交图和超图的矩阵补全问题。我们发现了一个尖锐阈值，可以精确补全评分矩阵。通过量化超图的“质量”函数，我们可以评估超图利用对样本概率的影响。通过开发高效算法，我们在高概率情况下成功地完成了矩阵补全任务。

    

    该论文研究了基于子采样矩阵条目以及观察到的社交图和超图的补全评分矩阵的问题。我们证明了在样本概率上存在一个尖锐阈值，用于精确完成评分矩阵的任务，当样本概率高于阈值时，任务可完成，反之则不可能，这展示了一个相变现象。阈值可以作为超图的“质量”函数来表示，从而使我们能够量化由于超图利用而导致的样本概率减少量，这也突显了超图在矩阵补全问题中的有用性。在发现尖锐阈值的过程中，我们开发了一种计算高效的矩阵补全算法，该算法有效地利用了观察到的图和超图。理论分析表明，只要样本概率高于某个阈值，我们的算法可以高概率地成功。

    This paper considers the problem of completing a rating matrix based on sub-sampled matrix entries as well as observed social graphs and hypergraphs. We show that there exists a \emph{sharp threshold} on the sample probability for the task of exactly completing the rating matrix -- the task is achievable when the sample probability is above the threshold, and is impossible otherwise -- demonstrating a phase transition phenomenon. The threshold can be expressed as a function of the ``quality'' of hypergraphs, enabling us to \emph{quantify} the amount of reduction in sample probability due to the exploitation of hypergraphs. This also highlights the usefulness of hypergraphs in the matrix completion problem. En route to discovering the sharp threshold, we develop a computationally efficient matrix completion algorithm that effectively exploits the observed graphs and hypergraphs. Theoretical analyses show that our algorithm succeeds with high probability as long as the sample probability
    


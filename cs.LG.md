# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Transparent and Clinically Interpretable AI for Lung Cancer Detection in Chest X-Rays](https://arxiv.org/abs/2403.19444) | 使用概念瓶颈模型的ante-hoc方法将临床概念引入到分类管道中，提供了肺癌检测决策过程中的有价值见解，相较于基线深度学习模型实现了更好的分类性能（F1 > 0.9）。 |
| [^2] | [Approximate Nullspace Augmented Finetuning for Robust Vision Transformers](https://arxiv.org/abs/2403.10476) | 本研究提出了一种启发自线性代数零空间概念的视觉变换器鲁棒性增强微调方法，通过合成近似零空间元素来提高模型的鲁棒性。 |
| [^3] | [Dynamics-Guided Diffusion Model for Robot Manipulator Design](https://arxiv.org/abs/2402.15038) | 该论文提出了动态引导扩散模型，利用共享的动力学网络为不同操作任务生成 manipulator 几何设计，通过设计目标构建的梯度引导手指几何设计的完善过程。 |
| [^4] | [Dataset Size Dependence of Rate-Distortion Curve and Threshold of Posterior Collapse in Linear VAE.](http://arxiv.org/abs/2309.07663) | 本文通过分析在高维限制下的最简化的VAE，提出了一个闭式表达式，评估了beta与VAE中数据集大小、后验坍缩和率失真曲线之间的关系。结果显示，随着beta的增加，产生较大的广义误差平台，并且选择一个小于特定阈值的beta值可以提高模型性能。 |
| [^5] | [Privacy Amplification via Importance Sampling.](http://arxiv.org/abs/2307.10187) | 通过重要性采样进行隐私放大，可以同时增强隐私保护和提高效用。我们提供了一个一般的结果来量化选择概率权重对隐私放大的影响，并展示了异质采样概率可以在保持子采样大小不变的情况下获得更好的隐私和效用。 |
| [^6] | [Attention-Based Transformer Networks for Quantum State Tomography.](http://arxiv.org/abs/2305.05433) | 本研究提出一种基于注意力机制和变压器网络的 QST 方法，可捕捉不同测量之间的相关性，并成功应用于检索量子态的密度矩阵，特别是对于受限测量数据的情况表现良好。 |
| [^7] | [ERSAM: Neural Architecture Search For Energy-Efficient and Real-Time Social Ambiance Measurement.](http://arxiv.org/abs/2303.10727) | 本文提出了一种面向能源高效性和实时性的社交氛围测量神经网络架构搜索框架ERSAM。该框架可以自动搜索适合SAM任务的神经网络架构，并满足能源效率、实时处理和有限标签数据的要求。在基准数据集上，该框架优于现有解决方案，具有更好的精度和效率。 |

# 详细

[^1]: 透明且临床可解释的人工智能用于胸部X射线肺癌检测

    Transparent and Clinically Interpretable AI for Lung Cancer Detection in Chest X-Rays

    [https://arxiv.org/abs/2403.19444](https://arxiv.org/abs/2403.19444)

    使用概念瓶颈模型的ante-hoc方法将临床概念引入到分类管道中，提供了肺癌检测决策过程中的有价值见解，相较于基线深度学习模型实现了更好的分类性能（F1 > 0.9）。

    

    arXiv:2403.19444v1 公告类型：新 简要摘要：透明人工智能（XAI）领域正在迅速发展，旨在解决复杂黑匣子深度学习模型在现实应用中的信任问题。现有的事后XAI技术最近已被证明在医疗数据上表现不佳，产生不可靠的解释，不适合临床使用。为解决这一问题，我们提出了一种基于概念瓶颈模型的ante-hoc方法，首次将临床概念引入分类管道，使用户可以深入了解决策过程。在一个大型公共数据集上，我们聚焦于胸部X射线和相关医疗报告的二元分类任务，即肺癌的检测。与基准深度学习模型相比，我们的方法在肺癌检测中获得了更好的分类性能（F1 > 0.9），同时生成了临床相关且更可靠的解释。

    arXiv:2403.19444v1 Announce Type: new  Abstract: The rapidly advancing field of Explainable Artificial Intelligence (XAI) aims to tackle the issue of trust regarding the use of complex black-box deep learning models in real-world applications. Existing post-hoc XAI techniques have recently been shown to have poor performance on medical data, producing unreliable explanations which are infeasible for clinical use. To address this, we propose an ante-hoc approach based on concept bottleneck models which introduces for the first time clinical concepts into the classification pipeline, allowing the user valuable insight into the decision-making process. On a large public dataset of chest X-rays and associated medical reports, we focus on the binary classification task of lung cancer detection. Our approach yields improved classification performance in lung cancer detection when compared to baseline deep learning models (F1 > 0.9), while also generating clinically relevant and more reliable
    
[^2]: 增强鲁棒性的近似零空间增强微调方法用于视觉变换器

    Approximate Nullspace Augmented Finetuning for Robust Vision Transformers

    [https://arxiv.org/abs/2403.10476](https://arxiv.org/abs/2403.10476)

    本研究提出了一种启发自线性代数零空间概念的视觉变换器鲁棒性增强微调方法，通过合成近似零空间元素来提高模型的鲁棒性。

    

    增强深度学习模型的鲁棒性，特别是在视觉变换器（ViTs）领域中，对于它们在现实世界中的部署至关重要。在这项工作中，我们提供了一种启发自线性代数中零空间概念的视觉变换器鲁棒性增强微调方法。我们的研究集中在一个问题上，即视觉变换器是否可以展现出类似于线性映射中的零空间属性的输入变化韧性，这意味着从该零空间中采样的扰动添加到输入时不会影响模型的输出。首先，我们展示了对于许多预训练的ViTs，存在一个非平凡的零空间，这是由于存在修补嵌入层。其次，由于零空间是与线性代数相关的概念，我们表明可以利用优化策略为ViTs的非线性块合成近似零空间元素。最后，我们提出了一种细致的方法

    arXiv:2403.10476v1 Announce Type: cross  Abstract: Enhancing the robustness of deep learning models, particularly in the realm of vision transformers (ViTs), is crucial for their real-world deployment. In this work, we provide a finetuning approach to enhance the robustness of vision transformers inspired by the concept of nullspace from linear algebra. Our investigation centers on whether a vision transformer can exhibit resilience to input variations akin to the nullspace property in linear mappings, implying that perturbations sampled from this nullspace do not influence the model's output when added to the input. Firstly, we show that for many pretrained ViTs, a non-trivial nullspace exists due to the presence of the patch embedding layer. Secondly, as nullspace is a concept associated with linear algebra, we demonstrate that it is possible to synthesize approximate nullspace elements for the non-linear blocks of ViTs employing an optimisation strategy. Finally, we propose a fine-t
    
[^3]: 动态引导扩散模型用于机器人 manipulator 设计

    Dynamics-Guided Diffusion Model for Robot Manipulator Design

    [https://arxiv.org/abs/2402.15038](https://arxiv.org/abs/2402.15038)

    该论文提出了动态引导扩散模型，利用共享的动力学网络为不同操作任务生成 manipulator 几何设计，通过设计目标构建的梯度引导手指几何设计的完善过程。

    

    我们提出了一个名为动态引导扩散模型的数据驱动框架，用于为给定操作任务生成 manipulator 几何设计。与为每个任务训练不同的设计模型不同，我们的方法采用一个跨任务共享的学习动力学网络。对于新的操作任务，我们首先将其分解为一组称为目标相互作用配置文件的个别运动目标，其中每个个别运动可以由共享的动力学网络建模。从目标和预测的相互作用配置文件构建的设计目标为任务的手指几何设计提供了梯度引导。这个设计过程被执行为一种分类器引导的扩散过程，其中设计目标作为分类器引导。我们在只使用开环平行夹爪运动的无传感器设置下，在各种操作任务上评估了我们的框架。

    arXiv:2402.15038v1 Announce Type: cross  Abstract: We present Dynamics-Guided Diffusion Model, a data-driven framework for generating manipulator geometry designs for a given manipulation task. Instead of training different design models for each task, our approach employs a learned dynamics network shared across tasks. For a new manipulation task, we first decompose it into a collection of individual motion targets which we call target interaction profile, where each individual motion can be modeled by the shared dynamics network. The design objective constructed from the target and predicted interaction profiles provides a gradient to guide the refinement of finger geometry for the task. This refinement process is executed as a classifier-guided diffusion process, where the design objective acts as the classifier guidance. We evaluate our framework on various manipulation tasks, under the sensor-less setting using only an open-loop parallel jaw motion. Our generated designs outperfor
    
[^4]: 线性变分自编码器中数据集大小对率失真曲线和后验坍缩阈值的影响

    Dataset Size Dependence of Rate-Distortion Curve and Threshold of Posterior Collapse in Linear VAE. (arXiv:2309.07663v1 [stat.ML])

    [http://arxiv.org/abs/2309.07663](http://arxiv.org/abs/2309.07663)

    本文通过分析在高维限制下的最简化的VAE，提出了一个闭式表达式，评估了beta与VAE中数据集大小、后验坍缩和率失真曲线之间的关系。结果显示，随着beta的增加，产生较大的广义误差平台，并且选择一个小于特定阈值的beta值可以提高模型性能。

    

    在变分自编码器（VAE）中，变分后验经常与先验密切吻合，这被称为后验坍缩，影响了表示学习的质量。为了缓解这个问题，VAE中引入了一个可调节的超参数beta。本文通过在高维限制下分析最简化的VAE，提出了一个闭式表达式，评估了beta与VAE中数据集大小、后验坍缩和率失真曲线之间的关系。这些结果表明，一个较大的beta会产生一个长的广义误差平台。随着beta的增加，平台的长度延长，超过一定的阈值后变为无穷。这意味着与通常的正则化参数不同，beta的选择可能会导致后验坍缩，而与数据集大小无关。因此，beta是一个需要谨慎调整的风险参数。此外，考虑到数据集大小对率失真曲线的依赖性，我们发现存在一个与数据集大小相关的阈值，选择小于这个阈值的beta值可以提高模型的性能。

    In the Variational Autoencoder (VAE), the variational posterior often aligns closely with the prior, which is known as posterior collapse and hinders the quality of representation learning. To mitigate this problem, an adjustable hyperparameter beta has been introduced in the VAE. This paper presents a closed-form expression to assess the relationship between the beta in VAE, the dataset size, the posterior collapse, and the rate-distortion curve by analyzing a minimal VAE in a high-dimensional limit. These results clarify that a long plateau in the generalization error emerges with a relatively larger beta. As the beta increases, the length of the plateau extends and then becomes infinite beyond a certain beta threshold. This implies that the choice of beta, unlike the usual regularization parameters, can induce posterior collapse regardless of the dataset size. Thus, beta is a risky parameter that requires careful tuning. Furthermore, considering the dataset-size dependence on the ra
    
[^5]: 隐私放大通过重要性采样

    Privacy Amplification via Importance Sampling. (arXiv:2307.10187v1 [cs.CR])

    [http://arxiv.org/abs/2307.10187](http://arxiv.org/abs/2307.10187)

    通过重要性采样进行隐私放大，可以同时增强隐私保护和提高效用。我们提供了一个一般的结果来量化选择概率权重对隐私放大的影响，并展示了异质采样概率可以在保持子采样大小不变的情况下获得更好的隐私和效用。

    

    我们研究了通过重要性采样对数据集进行子采样作为差分隐私机制的预处理步骤来增强隐私保护的性质。这扩展了已有的通过子采样进行隐私放大的结果到重要性采样，其中每个数据点的权重为其被选择概率的倒数。每个点的选择概率的权重对隐私的影响并不明显。一方面，较低的选择概率会导致更强的隐私放大。另一方面，权重越高，在点被选择时，点对机制输出的影响就越强。我们提供了一个一般的结果来量化这两个影响之间的权衡。我们展示了异质采样概率可以同时比均匀子采样具有更强的隐私和更好的效用，并保持子采样大小不变。特别地，我们制定和解决了隐私优化采样的问题，即寻找...

    We examine the privacy-enhancing properties of subsampling a data set via importance sampling as a pre-processing step for differentially private mechanisms. This extends the established privacy amplification by subsampling result to importance sampling where each data point is weighted by the reciprocal of its selection probability. The implications for privacy of weighting each point are not obvious. On the one hand, a lower selection probability leads to a stronger privacy amplification. On the other hand, the higher the weight, the stronger the influence of the point on the output of the mechanism in the event that the point does get selected. We provide a general result that quantifies the trade-off between these two effects. We show that heterogeneous sampling probabilities can lead to both stronger privacy and better utility than uniform subsampling while retaining the subsample size. In particular, we formulate and solve the problem of privacy-optimal sampling, that is, finding
    
[^6]: 基于注意力机制的变压器网络用于量子态重构

    Attention-Based Transformer Networks for Quantum State Tomography. (arXiv:2305.05433v1 [quant-ph])

    [http://arxiv.org/abs/2305.05433](http://arxiv.org/abs/2305.05433)

    本研究提出一种基于注意力机制和变压器网络的 QST 方法，可捕捉不同测量之间的相关性，并成功应用于检索量子态的密度矩阵，特别是对于受限测量数据的情况表现良好。

    

    由于其良好的表达能力，神经网络一直被用于量子态重构（QST）。为了进一步提高重构量子态的效率，本文探讨了语言建模与量子态重构之间的相似性，并提出了一种基于注意力机制和变压器网络的 QST 方法，用于捕捉不同测量之间的相关性。我们的方法直接从测量统计数据中检索量子态的密度矩阵，并辅助使用综合损失函数来帮助最小化实际态与检索态之间的差异。然后，我们系统地跟踪了涉及各种参数调整的常见训练策略对基于注意力机制的 QST 方法的不同影响。结合这些技术，我们建立了一个稳健的基准线，可以有效地重构纯态和混合态。此外，通过比较三种不同的神经网络方法的性能，我们证明了我们的基于注意力机制的方法表现优于其他方法，特别是对于受限测量数据的情况。

    Neural networks have been actively explored for quantum state tomography (QST) due to their favorable expressibility. To further enhance the efficiency of reconstructing quantum states, we explore the similarity between language modeling and quantum state tomography and propose an attention-based QST method that utilizes the Transformer network to capture the correlations between measured results from different measurements. Our method directly retrieves the density matrices of quantum states from measured statistics, with the assistance of an integrated loss function that helps minimize the difference between the actual states and the retrieved states. Then, we systematically trace different impacts within a bag of common training strategies involving various parameter adjustments on the attention-based QST method. Combining these techniques, we establish a robust baseline that can efficiently reconstruct pure and mixed quantum states. Furthermore, by comparing the performance of thre
    
[^7]: ERSAM: 面向能源高效和实时社交氛围测量的神经架构搜索

    ERSAM: Neural Architecture Search For Energy-Efficient and Real-Time Social Ambiance Measurement. (arXiv:2303.10727v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.10727](http://arxiv.org/abs/2303.10727)

    本文提出了一种面向能源高效性和实时性的社交氛围测量神经网络架构搜索框架ERSAM。该框架可以自动搜索适合SAM任务的神经网络架构，并满足能源效率、实时处理和有限标签数据的要求。在基准数据集上，该框架优于现有解决方案，具有更好的精度和效率。

    

    社交氛围描述了社交互动发生的背景，可以使用语音音频通过计算同时发言者的数量来测量。这种测量已经实现了各种心理健康跟踪和面向人类的物联网应用。虽然设备上的社交氛围测量 (SAM) 非常理想，以确保用户隐私并促进上述应用的广泛采用，但最先进的深度神经网络（DNNs）驱动的SAM解决方案所需的计算复杂度与移动设备上的常见资源相矛盾。此外，在临床设置下，由于各种隐私限制和所需的人力劳动，只有有限的标记数据可用或实际可行，这进一步挑战了设备上SAM解决方案的可实现准确性。为此，我们提出了一个专门的神经架构搜索框架，用于面向能源高效和实时SAM的ERSAM。具体而言，我们的ERSAM框架可以自动搜索适合SAM任务的神经网络架构，并满足能源效率、实时处理和有限标签数据的严格要求。我们在基准数据集上展示了我们提出的ERSAM框架的有效性，它优于最先进的SAM解决方案，并提高了精度和效率。

    Social ambiance describes the context in which social interactions happen, and can be measured using speech audio by counting the number of concurrent speakers. This measurement has enabled various mental health tracking and human-centric IoT applications. While on-device Socal Ambiance Measure (SAM) is highly desirable to ensure user privacy and thus facilitate wide adoption of the aforementioned applications, the required computational complexity of state-of-the-art deep neural networks (DNNs) powered SAM solutions stands at odds with the often constrained resources on mobile devices. Furthermore, only limited labeled data is available or practical when it comes to SAM under clinical settings due to various privacy constraints and the required human effort, further challenging the achievable accuracy of on-device SAM solutions. To this end, we propose a dedicated neural architecture search framework for Energy-efficient and Real-time SAM (ERSAM). Specifically, our ERSAM framework can
    


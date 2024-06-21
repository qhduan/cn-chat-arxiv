# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Active Adaptive Experimental Design for Treatment Effect Estimation with Covariate Choices](https://arxiv.org/abs/2403.03589) | 该研究提出了一种更有效地估计处理效应的活跃自适应实验设计方法，通过优化协变量密度和倾向得分来降低渐近方差。 |
| [^2] | [Multi-objective Differentiable Neural Architecture Search](https://arxiv.org/abs/2402.18213) | 提出了一种新颖的NAS算法，可以在一个搜索运行中编码用户对性能和硬件指标之间的权衡偏好，生成精心选择的多设备架构。 |
| [^3] | [The Mirrored Influence Hypothesis: Efficient Data Influence Estimation by Harnessing Forward Passes](https://arxiv.org/abs/2402.08922) | 本文介绍和探讨了镜像影响假设，突出了训练和测试数据之间影响的相互性。具体而言，它指出，评估训练数据对测试预测的影响可以重新表述为一个等效但相反的问题：评估如果模型在特定的测试样本上进行训练，对训练样本的预测将如何改变。通过实证和理论验证，我们演示了这一假设的正确性。 |
| [^4] | [Top-$K$ ranking with a monotone adversary](https://arxiv.org/abs/2402.07445) | 本文针对具有单调对手的Top-K排名问题，提出了一种加权最大似然估计器(MLE)，在样本复杂度方面接近最优。算法创新包括了对加权MLE的精确且紧密的$\ell_\infty$误差分析，并与加权比较图的谱特性相关联。 |
| [^5] | [Gradient descent induces alignment between weights and the empirical NTK for deep non-linear networks](https://arxiv.org/abs/2402.05271) | 了解神经网络从输入-标签对中提取统计信息的机制是监督学习中最重要的未解决问题之一。前人的研究表明，在训练过程中，权重的格拉姆矩阵与模型的平均梯度外积成正比，这被称为神经特征分析（NFA）。本研究解释了这种相关性的出现，并发现NFA等价于权重矩阵的左奇异结构与与这些权重相关的经验神经切线核的显著成分之间的对齐。在早期训练阶段，可以通过解析的方式预测NFA的发展速度。 |
| [^6] | [Anatomically-Controllable Medical Image Generation with Segmentation-Guided Diffusion Models](https://arxiv.org/abs/2402.05210) | 这篇论文提出了一种采用分割引导扩散模型的解剖可控医学图像生成方法，通过随机掩模消融训练算法实现对解剖约束的条件化，同时提高了网络对解剖真实性的学习能力。 |
| [^7] | [Discounted Adaptive Online Prediction](https://arxiv.org/abs/2402.02720) | 本论文提出了一种折扣自适应在线预测算法，该算法适应于复杂的损失序列和比较器，并改进了非自适应算法。算法具有无需结构性假设的理论保证，并且在超参数调整方面具有鲁棒性。通过在线符合预测任务的实验证明了算法的好处。 |
| [^8] | [What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement](https://arxiv.org/abs/2402.01865) | 本文研究了语言模型更新中的遗忘现象，提出了一种预测上游实例遗忘的方法，以改进重播过程的可控性和解释性。根据预训练实例的预-softmax对数几率分数变化与在线学习实例的相似性，提出了一种部分可解释的预测模型，在BART模型上表现良好但在T5模型上失败。此外，还展示了基于内积的黑盒分类器。 |
| [^9] | [Precipitation Downscaling with Spatiotemporal Video Diffusion](https://arxiv.org/abs/2312.06071) | 通过扩展视频扩散模型至降水超分辨率，本研究提出了一种利用确定性降尺度器和暂时条件扩散模型来捕捉噪声特征和高频率模式的方法。 |
| [^10] | [Analyzing Sharpness-aware Minimization under Overparameterization](https://arxiv.org/abs/2311.17539) | 本文分析了在过参数化条件下的锐度感知最小化方法。通过实证和理论结果，发现过参数化对锐度感知最小化具有重要影响，并且在过参数化增加的情况下，锐度感知最小化仍然受益。 |
| [^11] | [TAP: The Attention Patch for Cross-Modal Knowledge Transfer from Unlabeled Modality](https://arxiv.org/abs/2302.02224) | 通过引入The Attention Patch（TAP）神经网络附加组件，本文提出了一种简单且有效的方法，允许从未标记的次要模态实现跨模态的数据级知识传递。 |
| [^12] | [Revisiting the Learnability of Apple Tasting.](http://arxiv.org/abs/2310.19064) | 该论文重新审视了苹果品尝的可学习性，从组合角度研究了在线可学习性。作者通过引入Effective width参数，紧密量化了在可实现设置中的极小期望错误，并在可实现设置中建立了极小期望错误数量的三分法。 |
| [^13] | [The statistical thermodynamics of generative diffusion models.](http://arxiv.org/abs/2310.17467) | 本文通过在生成性扩散模型中应用平衡统计力学的工具，揭示了这些模型中的二阶相变现象，并且认为这种稳定性形式是生成能力的关键。 |
| [^14] | [Pseudo-Bayesian Optimization.](http://arxiv.org/abs/2310.09766) | 本文提出了伪贝叶斯优化，并通过研究最小要求的公理框架，构建了能确保黑盒优化收敛性的算法。 |
| [^15] | [Everything, Everywhere All in One Evaluation: Using Multiverse Analysis to Evaluate the Influence of Model Design Decisions on Algorithmic Fairness.](http://arxiv.org/abs/2308.16681) | 通过多元宇宙分析评估模型设计决策对算法公平性的影响，可以揭示算法决策系统中设计决策的关键作用。 |
| [^16] | [On the use of the Gram matrix for multivariate functional principal components analysis.](http://arxiv.org/abs/2306.12949) | 本文提出使用内积来估计多元和多维函数数据集的特征向量，为函数主成分分析提供了新的有效方法。 |
| [^17] | [Online Learning with Set-Valued Feedback.](http://arxiv.org/abs/2306.06247) | 本文研究了一种在线多类分类的变体，其中使用集合型反馈。通过引入新的组合维度，该论文表明确定性和随机性的在线可学习性在实现设置下不等价，并将在线多标签排名和在线多标签分类等实际学习设置作为其特定实例。 |
| [^18] | [Physics-informed neural networks with unknown measurement noise.](http://arxiv.org/abs/2211.15498) | 这篇论文提出了一种解决物理信息神经网络在存在非高斯噪声情况下失效的问题的方法，即通过同时训练一个能量模型来学习正确的噪声分布。通过多个例子的实验证明了该方法的改进性能。 |
| [^19] | [Imputation of missing values in multi-view data.](http://arxiv.org/abs/2210.14484) | 本文提出了一种基于StaPLR算法的新的多视角数据插补算法，通过在降维空间中执行插补以解决计算挑战，并在模拟数据集中得到了竞争性结果。 |

# 详细

[^1]: 用于处理因变量选择的活跃自适应实验设计的处理效应估计

    Active Adaptive Experimental Design for Treatment Effect Estimation with Covariate Choices

    [https://arxiv.org/abs/2403.03589](https://arxiv.org/abs/2403.03589)

    该研究提出了一种更有效地估计处理效应的活跃自适应实验设计方法，通过优化协变量密度和倾向得分来降低渐近方差。

    

    这项研究设计了一个自适应实验，用于高效地估计平均处理效应（ATEs）。我们考虑了一个自适应实验，其中实验者按顺序从由实验者决定的协变量密度中抽样一个实验单元，并分配一种处理。在分配处理后，实验者立即观察相应的结果。在实验结束时，实验者利用收集的样本估算出一个ATE。实验者的目标是通过较小的渐近方差估计ATE。现有研究已经设计了一些能够自适应优化倾向得分（处理分配概率）的实验。作为这种方法的一个概括，我们提出了一个框架，该框架下实验者优化协变量密度以及倾向得分，并发现优化协变量密度和倾向得分比仅优化倾向得分可以减少渐近方差更多的情况。

    arXiv:2403.03589v1 Announce Type: cross  Abstract: This study designs an adaptive experiment for efficiently estimating average treatment effect (ATEs). We consider an adaptive experiment where an experimenter sequentially samples an experimental unit from a covariate density decided by the experimenter and assigns a treatment. After assigning a treatment, the experimenter observes the corresponding outcome immediately. At the end of the experiment, the experimenter estimates an ATE using gathered samples. The objective of the experimenter is to estimate the ATE with a smaller asymptotic variance. Existing studies have designed experiments that adaptively optimize the propensity score (treatment-assignment probability). As a generalization of such an approach, we propose a framework under which an experimenter optimizes the covariate density, as well as the propensity score, and find that optimizing both covariate density and propensity score reduces the asymptotic variance more than o
    
[^2]: 多目标可微神经架构搜索

    Multi-objective Differentiable Neural Architecture Search

    [https://arxiv.org/abs/2402.18213](https://arxiv.org/abs/2402.18213)

    提出了一种新颖的NAS算法，可以在一个搜索运行中编码用户对性能和硬件指标之间的权衡偏好，生成精心选择的多设备架构。

    

    多目标优化（MOO）中的Pareto前沿轮廓剖析是具有挑战性的，尤其是在像神经网络训练这样的昂贵目标中。 相对于传统的NAS方法，我们提出了一种新颖的NAS算法，该算法在一个搜索运行中编码用户对性能和硬件指标之间的权衡偏好，并生成精心选择的多设备架构。为此，我们通过一个超网络参数化跨多个设备和多个目标的联合架构分布，超网络可以根据硬件特征和偏好向量进行条件化，实现零次搜索。

    arXiv:2402.18213v1 Announce Type: new  Abstract: Pareto front profiling in multi-objective optimization (MOO), i.e. finding a diverse set of Pareto optimal solutions, is challenging, especially with expensive objectives like neural network training. Typically, in MOO neural architecture search (NAS), we aim to balance performance and hardware metrics across devices. Prior NAS approaches simplify this task by incorporating hardware constraints into the objective function, but profiling the Pareto front necessitates a search for each constraint. In this work, we propose a novel NAS algorithm that encodes user preferences for the trade-off between performance and hardware metrics, and yields representative and diverse architectures across multiple devices in just one search run. To this end, we parameterize the joint architectural distribution across devices and multiple objectives via a hypernetwork that can be conditioned on hardware features and preference vectors, enabling zero-shot t
    
[^3]: 镜像影响假设：通过利用前向传递实现高效的数据影响估计

    The Mirrored Influence Hypothesis: Efficient Data Influence Estimation by Harnessing Forward Passes

    [https://arxiv.org/abs/2402.08922](https://arxiv.org/abs/2402.08922)

    本文介绍和探讨了镜像影响假设，突出了训练和测试数据之间影响的相互性。具体而言，它指出，评估训练数据对测试预测的影响可以重新表述为一个等效但相反的问题：评估如果模型在特定的测试样本上进行训练，对训练样本的预测将如何改变。通过实证和理论验证，我们演示了这一假设的正确性。

    

    大规模黑盒模型已经在许多应用中变得无处不在。了解个别训练数据源对这些模型所做预测的影响对于改善其可信性至关重要。当前的影响评估技术涉及计算每个训练点的梯度或在不同子集上重复训练。当扩展到大规模数据集和模型时，这些方法面临明显的计算挑战。

    arXiv:2402.08922v1 Announce Type: new Abstract: Large-scale black-box models have become ubiquitous across numerous applications. Understanding the influence of individual training data sources on predictions made by these models is crucial for improving their trustworthiness. Current influence estimation techniques involve computing gradients for every training point or repeated training on different subsets. These approaches face obvious computational challenges when scaled up to large datasets and models.   In this paper, we introduce and explore the Mirrored Influence Hypothesis, highlighting a reciprocal nature of influence between training and test data. Specifically, it suggests that evaluating the influence of training data on test predictions can be reformulated as an equivalent, yet inverse problem: assessing how the predictions for training samples would be altered if the model were trained on specific test samples. Through both empirical and theoretical validations, we demo
    
[^4]: 具有单调对手的Top-K排名问题

    Top-$K$ ranking with a monotone adversary

    [https://arxiv.org/abs/2402.07445](https://arxiv.org/abs/2402.07445)

    本文针对具有单调对手的Top-K排名问题，提出了一种加权最大似然估计器(MLE)，在样本复杂度方面接近最优。算法创新包括了对加权MLE的精确且紧密的$\ell_\infty$误差分析，并与加权比较图的谱特性相关联。

    

    本文解决了具有单调对手的Top-K排名问题。我们考虑了一个比较图被随机生成且对手可以添加任意边的情况。统计学家的目标是根据从这个半随机比较图导出的两两比较准确地识别出Top-K的首选项。本文的主要贡献是开发出一种加权最大似然估计器(MLE)，它在样本复杂度方面达到了近似最优，最多差一个$log^2(n)$的因子，其中n表示比较项的数量。这得益于分析和算法创新的结合。在分析方面，我们提供了一种更明确、更紧密的加权MLE的$\ell_\infty$误差分析，它与加权比较图的谱特性相关。受此启发，我们的算法创新涉及到了

    In this paper, we address the top-$K$ ranking problem with a monotone adversary. We consider the scenario where a comparison graph is randomly generated and the adversary is allowed to add arbitrary edges. The statistician's goal is then to accurately identify the top-$K$ preferred items based on pairwise comparisons derived from this semi-random comparison graph. The main contribution of this paper is to develop a weighted maximum likelihood estimator (MLE) that achieves near-optimal sample complexity, up to a $\log^2(n)$ factor, where n denotes the number of items under comparison. This is made possible through a combination of analytical and algorithmic innovations. On the analytical front, we provide a refined $\ell_\infty$ error analysis of the weighted MLE that is more explicit and tighter than existing analyses. It relates the $\ell_\infty$ error with the spectral properties of the weighted comparison graph. Motivated by this, our algorithmic innovation involves the development 
    
[^5]: 梯度下降引发了深度非线性网络权重与经验NTK之间的对齐

    Gradient descent induces alignment between weights and the empirical NTK for deep non-linear networks

    [https://arxiv.org/abs/2402.05271](https://arxiv.org/abs/2402.05271)

    了解神经网络从输入-标签对中提取统计信息的机制是监督学习中最重要的未解决问题之一。前人的研究表明，在训练过程中，权重的格拉姆矩阵与模型的平均梯度外积成正比，这被称为神经特征分析（NFA）。本研究解释了这种相关性的出现，并发现NFA等价于权重矩阵的左奇异结构与与这些权重相关的经验神经切线核的显著成分之间的对齐。在早期训练阶段，可以通过解析的方式预测NFA的发展速度。

    

    理解神经网络从输入-标签对中提取统计信息的机制是监督学习中最重要的未解决问题之一。先前的研究已经确定，在一般结构的训练神经网络中，权重的格拉姆矩阵与模型的平均梯度外积成正比，这个说法被称为神经特征分析（NFA）。然而，这些数量在训练过程中如何相关尚不清楚。在这项工作中，我们解释了这种相关性的出现。我们发现NFA等价于权重矩阵的左奇异结构与与这些权重相关的经验神经切线核的显著成分之间的对齐。我们证明了先前研究中引入的NFA是由隔离这种对齐的中心化NFA驱动的。我们还展示了在早期训练阶段，可以通过解析的方式预测NFA的发展速度。

    Understanding the mechanisms through which neural networks extract statistics from input-label pairs is one of the most important unsolved problems in supervised learning. Prior works have identified that the gram matrices of the weights in trained neural networks of general architectures are proportional to the average gradient outer product of the model, in a statement known as the Neural Feature Ansatz (NFA). However, the reason these quantities become correlated during training is poorly understood. In this work, we explain the emergence of this correlation. We identify that the NFA is equivalent to alignment between the left singular structure of the weight matrices and a significant component of the empirical neural tangent kernels associated with those weights. We establish that the NFA introduced in prior works is driven by a centered NFA that isolates this alignment. We show that the speed of NFA development can be predicted analytically at early training times in terms of sim
    
[^6]: 采用分割引导扩散模型的解剖可控医学图像生成

    Anatomically-Controllable Medical Image Generation with Segmentation-Guided Diffusion Models

    [https://arxiv.org/abs/2402.05210](https://arxiv.org/abs/2402.05210)

    这篇论文提出了一种采用分割引导扩散模型的解剖可控医学图像生成方法，通过随机掩模消融训练算法实现对解剖约束的条件化，同时提高了网络对解剖真实性的学习能力。

    

    扩散模型已经实现了非常高质量的医学图像生成，可以通过为小型或不平衡的数据集提供补充，从而帮助减轻获取和注释新图像的费用，同时还可以应用于其他方面。然而，这些模型在生成图像时面临着全局解剖真实性的挑战。因此，我们提出了一种解剖可控的医学图像生成模型。我们的模型在每个采样步骤中遵循多类解剖分割掩模，并采用随机掩模消融训练算法，以实现对所选解剖约束的条件化，同时允许其他解剖区域的灵活性。这也改善了网络在完全无条件（无约束生成）情况下对解剖真实性的学习。通过对乳腺MRI和腹部/颈部到盆腔CT数据集的比较评估，证明了我们模型在解剖真实性和输入掩模保真度方面具有优越性。

    Diffusion models have enabled remarkably high-quality medical image generation, which can help mitigate the expenses of acquiring and annotating new images by supplementing small or imbalanced datasets, along with other applications. However, these are hampered by the challenge of enforcing global anatomical realism in generated images. To this end, we propose a diffusion model for anatomically-controlled medical image generation. Our model follows a multi-class anatomical segmentation mask at each sampling step and incorporates a \textit{random mask ablation} training algorithm, to enable conditioning on a selected combination of anatomical constraints while allowing flexibility in other anatomical areas. This also improves the network's learning of anatomical realism for the completely unconditional (unconstrained generation) case. Comparative evaluation on breast MRI and abdominal/neck-to-pelvis CT datasets demonstrates superior anatomical realism and input mask faithfulness over st
    
[^7]: 折扣自适应在线预测

    Discounted Adaptive Online Prediction

    [https://arxiv.org/abs/2402.02720](https://arxiv.org/abs/2402.02720)

    本论文提出了一种折扣自适应在线预测算法，该算法适应于复杂的损失序列和比较器，并改进了非自适应算法。算法具有无需结构性假设的理论保证，并且在超参数调整方面具有鲁棒性。通过在线符合预测任务的实验证明了算法的好处。

    

    在线学习并不总是要记住一切。由于未来在统计上可能与过去有很大的不同，一个关键的挑战是在新数据到来时优雅地忘记历史。为了形式化这种直觉，我们运用最近发展的自适应在线学习技术重新思考了经典的折扣遗憾概念。我们的主要结果是一个新的算法，它适应于损失序列和比较器的复杂性，改进了广泛使用的非自适应算法-梯度下降算法，且具有恒定的学习率。特别地，我们的理论保证不需要任何结构性假设，只要求凸性，并且该算法经过证明对次优的超参数调整具有鲁棒性。我们进一步通过在线符合预测来展示这些好处，而在线符合预测是一个带有集合成员决策的下游在线学习任务。

    Online learning is not always about memorizing everything. Since the future can be statistically very different from the past, a critical challenge is to gracefully forget the history while new data comes in. To formalize this intuition, we revisit the classical notion of discounted regret using recently developed techniques in adaptive online learning. Our main result is a new algorithm that adapts to the complexity of both the loss sequence and the comparator, improving the widespread non-adaptive algorithm - gradient descent with a constant learning rate. In particular, our theoretical guarantee does not require any structural assumption beyond convexity, and the algorithm is provably robust to suboptimal hyperparameter tuning. We further demonstrate such benefits through online conformal prediction, a downstream online learning task with set-membership decisions.
    
[^8]: 我的模型会忘记什么？语言模型改进中的被遗忘实例预测

    What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement

    [https://arxiv.org/abs/2402.01865](https://arxiv.org/abs/2402.01865)

    本文研究了语言模型更新中的遗忘现象，提出了一种预测上游实例遗忘的方法，以改进重播过程的可控性和解释性。根据预训练实例的预-softmax对数几率分数变化与在线学习实例的相似性，提出了一种部分可解释的预测模型，在BART模型上表现良好但在T5模型上失败。此外，还展示了基于内积的黑盒分类器。

    

    在实际应用中，语言模型会出现错误。然而，仅仅通过将模型更新为纠正错误实例，会导致灾难性的遗忘，更新后的模型在指导微调或上游训练阶段中学到的实例上出现错误。随机重播上游数据的效果不令人满意，往往伴随着较高的方差和较差的可控性。为了改善重播过程的可控性和解释性，我们试图预测由于模型更新而遗忘的上游实例。我们根据一组在线学习的实例和相应被遗忘的上游预训练实例训练预测模型。我们提出了一种部分可解释的预测模型，该模型基于这样的观察结果：预训练实例的预-softmax对数几率分数的变化类似于在线学习实例的变化，这在BART模型上表现出不错的效果，但在T5模型上失败。我们进一步展示了基于内积的黑盒分类器

    Language models deployed in the wild make errors. However, simply updating the model with the corrected error instances causes catastrophic forgetting -- the updated model makes errors on instances learned during the instruction tuning or upstream training phase. Randomly replaying upstream data yields unsatisfactory performance and often comes with high variance and poor controllability. To this end, we try to forecast upstream examples that will be forgotten due to a model update for improved controllability of the replay process and interpretability. We train forecasting models given a collection of online learned examples and corresponding forgotten upstream pre-training examples. We propose a partially interpretable forecasting model based on the observation that changes in pre-softmax logit scores of pretraining examples resemble that of online learned examples, which performs decently on BART but fails on T5 models. We further show a black-box classifier based on inner products 
    
[^9]: 具有时空视频扩散的降水降尺度

    Precipitation Downscaling with Spatiotemporal Video Diffusion

    [https://arxiv.org/abs/2312.06071](https://arxiv.org/abs/2312.06071)

    通过扩展视频扩散模型至降水超分辨率，本研究提出了一种利用确定性降尺度器和暂时条件扩散模型来捕捉噪声特征和高频率模式的方法。

    

    在气候科学和气象学领域，高分辨率的局部降水（雨雪）预测受到基于模拟方法的计算成本限制。统计降尺度，或者称为超分辨率，是一种常见的解决方法，其中低分辨率预测通过统计方法得到改进。与传统计算机视觉任务不同，天气和气候应用需要捕捉给定低分辨率模式的高分辨率的准确条件分布，以确保可靠的集合平均和极端事件（如暴雨）的无偏估计。本研究将最新的视频扩散模型扩展到降水超分辨率，使用确定性降尺度器，然后是暂时条件的扩散模型来捕捉噪声特征和高频率模式。我们在FV3GFS输出上测试了我们的方法，这是一个已建立的大规模全球大气模型，并将其与其他方法进行了比较。

    arXiv:2312.06071v2 Announce Type: replace-cross  Abstract: In climate science and meteorology, high-resolution local precipitation (rain and snowfall) predictions are limited by the computational costs of simulation-based methods. Statistical downscaling, or super-resolution, is a common workaround where a low-resolution prediction is improved using statistical approaches. Unlike traditional computer vision tasks, weather and climate applications require capturing the accurate conditional distribution of high-resolution given low-resolution patterns to assure reliable ensemble averages and unbiased estimates of extreme events, such as heavy rain. This work extends recent video diffusion models to precipitation super-resolution, employing a deterministic downscaler followed by a temporally-conditioned diffusion model to capture noise characteristics and high-frequency patterns. We test our approach on FV3GFS output, an established large-scale global atmosphere model, and compare it agai
    
[^10]: 在过参数化下分析锐度感知最小化

    Analyzing Sharpness-aware Minimization under Overparameterization

    [https://arxiv.org/abs/2311.17539](https://arxiv.org/abs/2311.17539)

    本文分析了在过参数化条件下的锐度感知最小化方法。通过实证和理论结果，发现过参数化对锐度感知最小化具有重要影响，并且在过参数化增加的情况下，锐度感知最小化仍然受益。

    

    在训练过参数化的神经网络时，尽管训练损失相同，但可以得到具有不同泛化能力的极小值。有证据表明，极小值的锐度与其泛化误差之间存在相关性，因此已经做出了更多努力开发一种优化方法，以显式地找到扁平极小值作为更具有泛化能力的解。然而，至今为止，关于过参数化对锐度感知最小化（SAM）策略的影响的研究还不多。在这项工作中，我们分析了在不同程度的过参数化下的SAM，并提出了实证和理论结果，表明过参数化对SAM具有重要影响。具体而言，我们进行了广泛的数值实验，涵盖了各个领域，并表明存在一种一致的趋势，即SAM在过参数化增加的情况下仍然受益。我们还发现了一些令人信服的案例，说明了过参数化的影响。

    Training an overparameterized neural network can yield minimizers of different generalization capabilities despite the same level of training loss. With evidence that suggests a correlation between sharpness of minima and their generalization errors, increasing efforts have been made to develop an optimization method to explicitly find flat minima as more generalizable solutions. However, this sharpness-aware minimization (SAM) strategy has not been studied much yet as to whether and how it is affected by overparameterization.   In this work, we analyze SAM under overparameterization of varying degrees and present both empirical and theoretical results that indicate a critical influence of overparameterization on SAM. Specifically, we conduct extensive numerical experiments across various domains, and show that there exists a consistent trend that SAM continues to benefit from increasing overparameterization. We also discover compelling cases where the effect of overparameterization is
    
[^11]: TAP: 跨模态知识传递中的注意力补丁

    TAP: The Attention Patch for Cross-Modal Knowledge Transfer from Unlabeled Modality

    [https://arxiv.org/abs/2302.02224](https://arxiv.org/abs/2302.02224)

    通过引入The Attention Patch（TAP）神经网络附加组件，本文提出了一种简单且有效的方法，允许从未标记的次要模态实现跨模态的数据级知识传递。

    

    本文解决了跨模态学习框架，其目标是通过未标记、不配对的次要模态，增强主要模态中监督学习的性能。采用概率方法进行缺失信息估计，我们表明次要模态中包含的额外信息可以通过Nadaraya-Watson（NW）核回归进行估计，其可以进一步表示为经过线性变换的核交叉注意力模块。我们的结果为引入The Attention Patch（TAP）奠定了基础，这是一个简单的神经网络附加组件，允许从未标记的模态进行数据级知识传递。我们使用四个真实世界数据集进行了大量数值模拟，结果表明TAP能够显著提高跨不同领域和不同神经网络架构的泛化能力，利用看似无用的未标记信息。

    arXiv:2302.02224v2 Announce Type: replace  Abstract: This paper addresses a cross-modal learning framework, where the objective is to enhance the performance of supervised learning in the primary modality using an unlabeled, unpaired secondary modality. Taking a probabilistic approach for missing information estimation, we show that the extra information contained in the secondary modality can be estimated via Nadaraya-Watson (NW) kernel regression, which can further be expressed as a kernelized cross-attention module (under linear transformation). Our results lay the foundations for introducing The Attention Patch (TAP), a simple neural network add-on that allows data-level knowledge transfer from the unlabeled modality. We provide extensive numerical simulations using four real-world datasets to show that TAP can provide statistically significant improvement in generalization across different domains and different neural network architectures, making use of seemingly unusable unlabel
    
[^12]: 重新审视苹果品尝的可学习性

    Revisiting the Learnability of Apple Tasting. (arXiv:2310.19064v1 [cs.LG])

    [http://arxiv.org/abs/2310.19064](http://arxiv.org/abs/2310.19064)

    该论文重新审视了苹果品尝的可学习性，从组合角度研究了在线可学习性。作者通过引入Effective width参数，紧密量化了在可实现设置中的极小期望错误，并在可实现设置中建立了极小期望错误数量的三分法。

    

    在在线二元分类中，学习者只有在预测为"1"时观察到真实标签。本文重新研究了这种经典的部分反馈设置，并从组合角度研究了在线可学习性。我们证明了在不可知设置下，Littlestone维度仍然是苹果品尝的紧密定量刻画，解决了\cite{helmbold2000apple}提出的一个悬而未决的问题。此外，我们给出了一个新的组合参数，称为有效宽度，紧密量化了在可实现设置中的极小期望错误。作为推论，我们使用有效宽度在可实现设置中建立了极小期望错误数量的三分法。特别地，我们证明了在可实现设置中，任何学习者在苹果品尝反馈下的期望错误数量只能是$\Theta(1), \Theta(\sqrt{T})$, 或 $\Theta(T)$。

    In online binary classification under \textit{apple tasting} feedback, the learner only observes the true label if it predicts "1". First studied by \cite{helmbold2000apple}, we revisit this classical partial-feedback setting and study online learnability from a combinatorial perspective. We show that the Littlestone dimension continues to prove a tight quantitative characterization of apple tasting in the agnostic setting, closing an open question posed by \cite{helmbold2000apple}. In addition, we give a new combinatorial parameter, called the Effective width, that tightly quantifies the minimax expected mistakes in the realizable setting. As a corollary, we use the Effective width to establish a \textit{trichotomy} of the minimax expected number of mistakes in the realizable setting. In particular, we show that in the realizable setting, the expected number of mistakes for any learner under apple tasting feedback can only be $\Theta(1), \Theta(\sqrt{T})$, or $\Theta(T)$.
    
[^13]: 生成性扩散模型的统计热力学

    The statistical thermodynamics of generative diffusion models. (arXiv:2310.17467v1 [stat.ML])

    [http://arxiv.org/abs/2310.17467](http://arxiv.org/abs/2310.17467)

    本文通过在生成性扩散模型中应用平衡统计力学的工具，揭示了这些模型中的二阶相变现象，并且认为这种稳定性形式是生成能力的关键。

    

    生成性扩散模型在生成建模的许多领域取得了惊人的表现。虽然这些模型的基本思想来自非平衡物理学，但本文中我们表明，可以用平衡统计力学的工具来理解这些模型的许多方面。利用这种重构，我们展示了生成性扩散模型经历了与对称性破缺现象相对应的二阶相变。我们认为，这导致了一种稳定性形式，它是生成能力的核心，并可以用一组平均场临界指数来描述。最后，我们根据热力学的公式分析了将扩散模型与关联记忆网络连接的最近研究。

    Generative diffusion models have achieved spectacular performance in many areas of generative modeling. While the fundamental ideas behind these models come from non-equilibrium physics, in this paper we show that many aspects of these models can be understood using the tools of equilibrium statistical mechanics. Using this reformulation, we show that generative diffusion models undergo second-order phase transitions corresponding to symmetry breaking phenomena. We argue that this lead to a form of instability that lies at the heart of their generative capabilities and that can be described by a set of mean field critical exponents. We conclude by analyzing recent work connecting diffusion models and associative memory networks in view of the thermodynamic formulations.
    
[^14]: 伪贝叶斯优化

    Pseudo-Bayesian Optimization. (arXiv:2310.09766v1 [stat.ML])

    [http://arxiv.org/abs/2310.09766](http://arxiv.org/abs/2310.09766)

    本文提出了伪贝叶斯优化，并通过研究最小要求的公理框架，构建了能确保黑盒优化收敛性的算法。

    

    贝叶斯优化是一种优化昂贵黑盒函数的流行方法。其关键思想是使用一个替代模型来近似目标，并且重要的是量化相关的不确定性，从而实现探索和开发之间的平衡的顺序搜索。高斯过程(GP)一直是替代模型的首选，因为它具有贝叶斯的不确定性量化能力和建模灵活性。然而，它的挑战也引发了一系列收敛性更显得不明显的备选方案。在本文中，我们通过研究引出最小要求的公理框架来确保黑盒优化的收敛性，以应用于除了GP相关方法之外的情况。此外，我们利用我们的框架中的设计自由，我们称之为伪贝叶斯优化，来构建经验上更优的算法。特别地，我们展示了如何使用简单的局部回归和一个适应问题特性的代理模型来实现这一目标。

    Bayesian Optimization is a popular approach for optimizing expensive black-box functions. Its key idea is to use a surrogate model to approximate the objective and, importantly, quantify the associated uncertainty that allows a sequential search of query points that balance exploitation-exploration. Gaussian process (GP) has been a primary candidate for the surrogate model, thanks to its Bayesian-principled uncertainty quantification power and modeling flexibility. However, its challenges have also spurred an array of alternatives whose convergence properties could be more opaque. Motivated by these, we study in this paper an axiomatic framework that elicits the minimal requirements to guarantee black-box optimization convergence that could apply beyond GP-related methods. Moreover, we leverage the design freedom in our framework, which we call Pseudo-Bayesian Optimization, to construct empirically superior algorithms. In particular, we show how using simple local regression, and a sui
    
[^15]: 通过多元宇宙分析评估模型设计决策对算法公平性的影响：一切，无处不在，全方位评估

    Everything, Everywhere All in One Evaluation: Using Multiverse Analysis to Evaluate the Influence of Model Design Decisions on Algorithmic Fairness. (arXiv:2308.16681v1 [stat.ML])

    [http://arxiv.org/abs/2308.16681](http://arxiv.org/abs/2308.16681)

    通过多元宇宙分析评估模型设计决策对算法公平性的影响，可以揭示算法决策系统中设计决策的关键作用。

    

    全球范围内的许多系统都利用算法决策来（部分）自动化以前由人类进行的决策。当设计良好时，这些系统承诺更客观的决策，同时节省大量资源，节约人力。然而，当算法决策系统设计不良时，可能会导致对社会群体进行歧视的不公平决策。算法决策系统的下游效应在很大程度上取决于系统设计和实施过程中的决策，因为数据中的偏见可能会在建模过程中缓解或加强。许多这些设计决策是隐含进行的，不知道它们确切地如何影响最终系统。因此，明确算法决策系统设计中的决策并了解这些决策如何影响结果系统的公平性非常重要。为了研究这个问题，我们借鉴了心理学领域的见解，并引入了多元宇宙分析方法。

    A vast number of systems across the world use algorithmic decision making (ADM) to (partially) automate decisions that have previously been made by humans. When designed well, these systems promise more objective decisions while saving large amounts of resources and freeing up human time. However, when ADM systems are not designed well, they can lead to unfair decisions which discriminate against societal groups. The downstream effects of ADMs critically depend on the decisions made during the systems' design and implementation, as biases in data can be mitigated or reinforced along the modeling pipeline. Many of these design decisions are made implicitly, without knowing exactly how they will influence the final system. It is therefore important to make explicit the decisions made during the design of ADM systems and understand how these decisions affect the fairness of the resulting system.  To study this issue, we draw on insights from the field of psychology and introduce the metho
    
[^16]: 关于使用格拉姆矩阵进行多元函数主成分分析的研究

    On the use of the Gram matrix for multivariate functional principal components analysis. (arXiv:2306.12949v1 [stat.ME])

    [http://arxiv.org/abs/2306.12949](http://arxiv.org/abs/2306.12949)

    本文提出使用内积来估计多元和多维函数数据集的特征向量，为函数主成分分析提供了新的有效方法。

    

    在函数数据分析中，降维是至关重要的。降维的关键工具是函数主成分分析。现有的函数主成分分析方法通常涉及协方差矩阵的对角化。随着函数数据集的规模和复杂性增加，协方差矩阵的估计变得更加具有挑战性。因此，需要有效的方法来估计特征向量。基于观测空间和函数特征空间的对偶性，我们提出使用曲线之间的内积来估计多元和多维函数数据集的特征向量。建立了协方差矩阵特征向量和内积矩阵特征向量之间的关系。我们探讨了这些方法在几个函数数据分析设置中的应用，并提供了它们的通用指导。

    Dimension reduction is crucial in functional data analysis (FDA). The key tool to reduce the dimension of the data is functional principal component analysis. Existing approaches for functional principal component analysis usually involve the diagonalization of the covariance operator. With the increasing size and complexity of functional datasets, estimating the covariance operator has become more challenging. Therefore, there is a growing need for efficient methodologies to estimate the eigencomponents. Using the duality of the space of observations and the space of functional features, we propose to use the inner-product between the curves to estimate the eigenelements of multivariate and multidimensional functional datasets. The relationship between the eigenelements of the covariance operator and those of the inner-product matrix is established. We explore the application of these methodologies in several FDA settings and provide general guidance on their usability.
    
[^17]: 使用集合型反馈的在线学习

    Online Learning with Set-Valued Feedback. (arXiv:2306.06247v1 [cs.LG])

    [http://arxiv.org/abs/2306.06247](http://arxiv.org/abs/2306.06247)

    本文研究了一种在线多类分类的变体，其中使用集合型反馈。通过引入新的组合维度，该论文表明确定性和随机性的在线可学习性在实现设置下不等价，并将在线多标签排名和在线多标签分类等实际学习设置作为其特定实例。

    

    本文研究了在线多类分类的一种变体，其中学习器预测单个标签，但接收到一个标签的集合作为反馈。在该模型中，如果学习器没有输出包含在反馈集合中的标签，则会受到惩罚。我们表明，与具有单标签反馈的在线多类学习不同，在实现设置中使用集合型反馈时，确定性和随机化的在线可学习性\textit{不等价}。因此，我们提供了两个新的组合维度，分别命名为集合小石和度量破裂维度，严格描述了确定性和随机化的在线可学习性。此外，我们表明度量破裂维度在悟性设置下严格描述在线可学习性。最后，我们证明了在线多标签排名和在线多标签分类等实际学习设置是我们通用在线学习框架的具体实例。

    We study a variant of online multiclass classification where the learner predicts a single label but receives a \textit{set of labels} as feedback. In this model, the learner is penalized for not outputting a label contained in the revealed set. We show that unlike online multiclass learning with single-label feedback, deterministic and randomized online learnability are \textit{not equivalent} even in the realizable setting with set-valued feedback. Accordingly, we give two new combinatorial dimensions, named the Set Littlestone and Measure Shattering dimension, that tightly characterize deterministic and randomized online learnability respectively in the realizable setting. In addition, we show that the Measure Shattering dimension tightly characterizes online learnability in the agnostic setting. Finally, we show that practical learning settings like online multilabel ranking and online multilabel classification are specific instances of our general online learning framework.
    
[^18]: 具有未知测量噪声的物理信息神经网络

    Physics-informed neural networks with unknown measurement noise. (arXiv:2211.15498v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2211.15498](http://arxiv.org/abs/2211.15498)

    这篇论文提出了一种解决物理信息神经网络在存在非高斯噪声情况下失效的问题的方法，即通过同时训练一个能量模型来学习正确的噪声分布。通过多个例子的实验证明了该方法的改进性能。

    

    物理信息神经网络(PINNs)是一种既能找到解决方案又能识别偏微分方程参数的灵活方法。大多数相关的研究都假设数据是无噪声的，或者是受弱高斯噪声污染的。我们展示了标准PINN框架在非高斯噪声情况下失效的问题，并提出了一种解决这个根本性问题的方法，即同时训练一个能量模型(Energy-Based Model, EBM)来学习正确的噪声分布。我们通过多个例子展示了我们方法的改进性能。

    Physics-informed neural networks (PINNs) constitute a flexible approach to both finding solutions and identifying parameters of partial differential equations. Most works on the topic assume noiseless data, or data contaminated by weak Gaussian noise. We show that the standard PINN framework breaks down in case of non-Gaussian noise. We give a way of resolving this fundamental issue and we propose to jointly train an energy-based model (EBM) to learn the correct noise distribution. We illustrate the improved performance of our approach using multiple examples.
    
[^19]: 多视角数据中缺失值的插补问题解决方法

    Imputation of missing values in multi-view data. (arXiv:2210.14484v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.14484](http://arxiv.org/abs/2210.14484)

    本文提出了一种基于StaPLR算法的新的多视角数据插补算法，通过在降维空间中执行插补以解决计算挑战，并在模拟数据集中得到了竞争性结果。

    

    多视角数据是指由多个不同特征集描述的数据。在处理多视角数据时，若出现缺失值，则一个视角中的所有特征极有可能同时缺失，因而导致非常大量的缺失数据问题。本文提出了一种新的多视角学习算法中的插补方法，它基于堆叠惩罚逻辑回归(StaPLR)算法，在降维空间中执行插补，以解决固有的多视角计算挑战。实验结果表明，该方法在模拟数据集上具有竞争性结果，而且具有更低的计算成本，从而可以使用先进的插补算法，例如missForest。

    Data for which a set of objects is described by multiple distinct feature sets (called views) is known as multi-view data. When missing values occur in multi-view data, all features in a view are likely to be missing simultaneously. This leads to very large quantities of missing data which, especially when combined with high-dimensionality, makes the application of conditional imputation methods computationally infeasible. We introduce a new imputation method based on the existing stacked penalized logistic regression (StaPLR) algorithm for multi-view learning. It performs imputation in a dimension-reduced space to address computational challenges inherent to the multi-view context. We compare the performance of the new imputation method with several existing imputation algorithms in simulated data sets. The results show that the new imputation method leads to competitive results at a much lower computational cost, and makes the use of advanced imputation algorithms such as missForest 
    


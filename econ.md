# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FishNet: Deep Neural Networks for Low-Cost Fish Stock Estimation](https://arxiv.org/abs/2403.10916) | 提出了FishNet，一个自动化计算机视觉系统，利用低成本数码相机图像执行鱼类分类和大小估算，具有高准确性和精度。 |
| [^2] | [Generative Probabilistic Forecasting with Applications in Market Operations](https://arxiv.org/abs/2403.05743) | 提出了一种基于Wiener-Kallianpur创新表示的生成式概率预测方法，包括自编码器和新颖的深度学习算法，具有渐近最优性和结构收敛性质，适用于实时市场运营中的高动态和波动时间序列。 |
| [^3] | [Socioeconomic agents as active matter in nonequilibrium Sakoda-Schelling models.](http://arxiv.org/abs/2307.14270) | 该研究通过考虑Sakoda-Schelling模型中的职业模型，揭示了社会经济代理人模型中的非平衡动力学，并在平均场近似下将其映射为主动物质描述。通过研究非互惠性互动，展示了非稳态的宏观行为。这一研究为地理相关的基于代理人的模型提供了统一的框架，有助于同时考虑人口和价格动态。 |

# 详细

[^1]: FishNet:用于低成本鱼类存栏估算的深度神经网络

    FishNet: Deep Neural Networks for Low-Cost Fish Stock Estimation

    [https://arxiv.org/abs/2403.10916](https://arxiv.org/abs/2403.10916)

    提出了FishNet，一个自动化计算机视觉系统，利用低成本数码相机图像执行鱼类分类和大小估算，具有高准确性和精度。

    

    鱼类库存评估通常需要由分类专家进行手工鱼类计数，这既耗时又昂贵。我们提出了一个自动化的计算机视觉系统，可以从使用低成本数码相机拍摄的图像中执行分类和鱼类大小估算。该系统首先利用Mask R-CNN执行目标检测和分割，以识别包含多条鱼的图像中的单条鱼，这些鱼可能由不同物种组成。然后，每个鱼类被分类并使用单独的机器学习模型预测长度。这些模型训练于包含50,000张手工注释图像的数据集，其中包含163种不同长度从10厘米到250厘米的鱼类。在保留的测试数据上评估，我们的系统在鱼类分割任务上达到了92%的交并比，单一鱼类分类准确率为89%，平均误差为2.3厘米。

    arXiv:2403.10916v1 Announce Type: cross  Abstract: Fish stock assessment often involves manual fish counting by taxonomy specialists, which is both time-consuming and costly. We propose an automated computer vision system that performs both taxonomic classification and fish size estimation from images taken with a low-cost digital camera. The system first performs object detection and segmentation using a Mask R-CNN to identify individual fish from images containing multiple fish, possibly consisting of different species. Then each fish species is classified and the predicted length using separate machine learning models. These models are trained on a dataset of 50,000 hand-annotated images containing 163 different fish species, ranging in length from 10cm to 250cm. Evaluated on held-out test data, our system achieves a $92\%$ intersection over union on the fish segmentation task, a $89\%$ top-1 classification accuracy on single fish species classification, and a $2.3$~cm mean error on
    
[^2]: 具有市场运营应用的生成式概率预测

    Generative Probabilistic Forecasting with Applications in Market Operations

    [https://arxiv.org/abs/2403.05743](https://arxiv.org/abs/2403.05743)

    提出了一种基于Wiener-Kallianpur创新表示的生成式概率预测方法，包括自编码器和新颖的深度学习算法，具有渐近最优性和结构收敛性质，适用于实时市场运营中的高动态和波动时间序列。

    

    本文提出了一种新颖的生成式概率预测方法，该方法源自于非参数时间序列的Wiener-Kallianpur创新表示。在生成人工智能的范式下，所提出的预测架构包括一个自编码器，将非参数多变量随机过程转化为规范的创新序列，从中根据过去样本生成未来时间序列样本，条件是它们的概率分布取决于过去样本。提出了一种新的深度学习算法，将潜在过程限制为具有匹配自编码器输入-输出条件概率分布的独立同分布序列。建立了所提出的生成式预测方法的渐近最优性和结构收敛性质。该方法在实时市场运营中涉及高度动态和波动时间序列的三个应用方面。

    arXiv:2403.05743v1 Announce Type: cross  Abstract: This paper presents a novel generative probabilistic forecasting approach derived from the Wiener-Kallianpur innovation representation of nonparametric time series. Under the paradigm of generative artificial intelligence, the proposed forecasting architecture includes an autoencoder that transforms nonparametric multivariate random processes into canonical innovation sequences, from which future time series samples are generated according to their probability distributions conditioned on past samples. A novel deep-learning algorithm is proposed that constrains the latent process to be an independent and identically distributed sequence with matching autoencoder input-output conditional probability distributions. Asymptotic optimality and structural convergence properties of the proposed generative forecasting approach are established. Three applications involving highly dynamic and volatile time series in real-time market operations a
    
[^3]: 非平衡的Sakoda-Schelling模型中的社会经济代理人作为主动物质

    Socioeconomic agents as active matter in nonequilibrium Sakoda-Schelling models. (arXiv:2307.14270v1 [cond-mat.stat-mech])

    [http://arxiv.org/abs/2307.14270](http://arxiv.org/abs/2307.14270)

    该研究通过考虑Sakoda-Schelling模型中的职业模型，揭示了社会经济代理人模型中的非平衡动力学，并在平均场近似下将其映射为主动物质描述。通过研究非互惠性互动，展示了非稳态的宏观行为。这一研究为地理相关的基于代理人的模型提供了统一的框架，有助于同时考虑人口和价格动态。

    

    代理人的决策规则对于社会经济代理人模型有多么稳健？我们通过考虑一种类似Sakoda-Schelling模型的职业模型来解决这个问题，该模型在历史上被引入以揭示人类群体之间的隔离动力学。对于大类的效用函数和决策规则，我们确定了代理人动力学的非平衡性，同时恢复了类似平衡相分离的现象学。在平均场近似下，我们展示了该模型在一定程度上可以被映射为主动物质场描述（Active Model B）。最后，我们考虑了两个人群之间的非互惠性互动，并展示了它们如何导致非稳态的宏观行为。我们相信我们的方法提供了一个统一的框架，进一步研究地理相关的基于代理人的模型，尤其是在场论方法中同时考虑人口和价格动态的研究。

    How robust are socioeconomic agent-based models with respect to the details of the agents' decision rule? We tackle this question by considering an occupation model in the spirit of the Sakoda-Schelling model, historically introduced to shed light on segregation dynamics among human groups. For a large class of utility functions and decision rules, we pinpoint the nonequilibrium nature of the agent dynamics, while recovering the equilibrium-like phase separation phenomenology. Within the mean field approximation we show how the model can be mapped, to some extent, onto an active matter field description (Active Model B). Finally, we consider non-reciprocal interactions between two populations, and show how they can lead to non-steady macroscopic behavior. We believe our approach provides a unifying framework to further study geography-dependent agent-based models, notably paving the way for joint consideration of population and price dynamics within a field theoretic approach.
    


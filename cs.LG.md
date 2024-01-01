# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Sublinear-Time Spectral Clustering Oracle with Improved Preprocessing Time.](http://arxiv.org/abs/2310.17878) | 本研究提出了一种亚线性时间的谱聚类预测器，用于处理具有强聚类特性的图。该预测器能够在亚线性时间内进行预处理和查询聚类成员，并且与真实聚类接近的k-分区保持一致。此外，该预测器对于少量的随机边删除具有鲁棒性。 |
| [^2] | [Compositional Abilities Emerge Multiplicatively: Exploring Diffusion Models on a Synthetic Task.](http://arxiv.org/abs/2310.09336) | 组合能力以乘法方式出现：研究了条件扩散模型在合成任务中的组合泛化能力，结果显示这种能力受到底层数据生成过程的结构影响，且模型在学习到更高级的组合时存在困难。 |
| [^3] | [Comparing the robustness of modern no-reference image- and video-quality metrics to adversarial attacks.](http://arxiv.org/abs/2310.06958) | 本文比较了现代图像和视频质量评估度量方法对抗攻击的鲁棒性，并发现部分度量方法对对抗攻击表现出较高的抵抗力，为基准测试提供了更安全的选择。 |
| [^4] | [Offline Imitation Learning with Variational Counterfactual Reasoning.](http://arxiv.org/abs/2310.04706) | 该论文提出了一个名为OILCA的框架，利用可识别的变分自动编码器生成"对抗性"样本，以解决离线模仿学习中数据稀缺、环境变化等问题。 |
| [^5] | [Robustness-enhanced Uplift Modeling with Adversarial Feature Desensitization.](http://arxiv.org/abs/2310.04693) | 本文提出了一种增强鲁棒性的提升建模框架RUAD，并通过特征选择和对抗特征抑制两个定制模块更有效地解决了提升模型的特征敏感性问题。 |
| [^6] | [Graph Neural Prompting with Large Language Models.](http://arxiv.org/abs/2309.15427) | 本文提出了一种名为图神经提示（GNP）的方法，可以帮助大型语言模型从知识图中学习有益的知识，以弥补它们在准确捕捉和返回基于知识的信息方面的固有限制。 |
| [^7] | [Maximum Diffusion Reinforcement Learning.](http://arxiv.org/abs/2309.15293) | 最大扩散强化学习是一种克服强化学习中数据相关性问题的方法，通过解耦代理的经验实现持续学习，并在各种测试中表现出色。 |
| [^8] | [Fast Slate Policy Optimization: Going Beyond Plackett-Luce.](http://arxiv.org/abs/2308.01566) | 本文介绍了一种快速Slate策略优化方法，通过提出一种新的策略类，可以在大规模决策系统中有效地优化任意奖励函数，结果表明该方法在百万级别动作空间问题上具有很好的效果。 |
| [^9] | [VillanDiffusion: A Unified Backdoor Attack Framework for Diffusion Models.](http://arxiv.org/abs/2306.06874) | 本文提出VillanDiffusion，一个针对扩散模型的统一后门攻击框架，涵盖主流的无条件和有条件DM，便于对不同DM配置进行后门分析，并为基于字幕的DM后门攻击提供了新的见解。 |
| [^10] | [Differentiable Earth Mover's Distance for Data Compression at the High-Luminosity LHC.](http://arxiv.org/abs/2306.04712) | 本文利用可微分的快速逼近方法，训练了一个编码器神经网络用于高亮LHC数据的压缩，同时保留了数据内与粒子探测器中的能量沉积分布相关的信息。 |
| [^11] | [M3ICRO: Machine Learning-Enabled Compact Photonic Tensor Core based on PRogrammable Multi-Operand Multimode Interference.](http://arxiv.org/abs/2305.19505) | M3ICRO是一种基于定制MOMMI器件的机器学习光子张量核心，具有超高能效、紧凑型设计和ML for optics优化方法，可以用于加速图像识别、自然语言处理等多种ML任务。 |
| [^12] | [Distributional Offline Policy Evaluation with Predictive Error Guarantees.](http://arxiv.org/abs/2302.09456) | 本论文提出了一种名为Fitted Likelihood Estimation (FLE)的算法来解决分布式离线策略评估的问题，该算法能够学习到密切接近真实分布的策略回报分布。 |

# 详细

[^1]: 一种具有改进预处理时间的亚线性时间谱聚类预测器

    A Sublinear-Time Spectral Clustering Oracle with Improved Preprocessing Time. (arXiv:2310.17878v1 [cs.DS])

    [http://arxiv.org/abs/2310.17878](http://arxiv.org/abs/2310.17878)

    本研究提出了一种亚线性时间的谱聚类预测器，用于处理具有强聚类特性的图。该预测器能够在亚线性时间内进行预处理和查询聚类成员，并且与真实聚类接近的k-分区保持一致。此外，该预测器对于少量的随机边删除具有鲁棒性。

    

    我们解决了设计一种适用于具有强聚类特性的图的亚线性时间谱聚类预测器的问题。这样的图包含k个潜在聚类，每个聚类的内导纳较大（至少为φ），外导纳较小（最多为ε）。我们的目标是对图进行预处理，以使得聚类成员查询能够在亚线性时间内进行，并且所得到的分区应与接近真实聚类的k-分区一致。之前的预测器要么依赖于内外导纳之间有一个poly(k)log n的差距，要么需要指数级（在k/ε上）的预处理时间。我们的算法放宽了这些假设，尽管会略微增加错误分类率。我们还展示了我们的聚类预测器对于少量的随机边删除是鲁棒的。为了验证我们的理论界限，我们进行了实验。

    We address the problem of designing a sublinear-time spectral clustering oracle for graphs that exhibit strong clusterability. Such graphs contain $k$ latent clusters, each characterized by a large inner conductance (at least $\varphi$) and a small outer conductance (at most $\varepsilon$). Our aim is to preprocess the graph to enable clustering membership queries, with the key requirement that both preprocessing and query answering should be performed in sublinear time, and the resulting partition should be consistent with a $k$-partition that is close to the ground-truth clustering. Previous oracles have relied on either a $\textrm{poly}(k)\log n$ gap between inner and outer conductances or exponential (in $k/\varepsilon$) preprocessing time. Our algorithm relaxes these assumptions, albeit at the cost of a slightly higher misclassification ratio. We also show that our clustering oracle is robust against a few random edge deletions. To validate our theoretical bounds, we conducted exp
    
[^2]: 组合能力以乘法方式出现：在合成任务中探索扩散模型

    Compositional Abilities Emerge Multiplicatively: Exploring Diffusion Models on a Synthetic Task. (arXiv:2310.09336v1 [cs.LG])

    [http://arxiv.org/abs/2310.09336](http://arxiv.org/abs/2310.09336)

    组合能力以乘法方式出现：研究了条件扩散模型在合成任务中的组合泛化能力，结果显示这种能力受到底层数据生成过程的结构影响，且模型在学习到更高级的组合时存在困难。

    

    现代生成模型展示出了产生极为逼真数据的前所未有的能力。然而，考虑到现实世界的自然组合性，这些模型在实际应用中可靠使用需要展示出能够组合新的概念集合以生成训练数据集中未见的输出的能力。先前的研究表明，最近的扩散模型确实表现出了有趣的组合泛化能力，但它们也会出现无法预测的失败。受此启发，我们在合成环境中进行了有控制性的研究，以了解条件扩散模型的组合泛化能力，我们变化了训练数据的不同属性并测量了模型生成越界样本的能力。我们的结果显示：（i）从一个概念生成样本的能力和将它们组合起来的能力的出现顺序受到了底层数据生成过程的结构的影响；（ii）在组合任务上的表现表明模型在学习到更高级的组合时存在困难。

    Modern generative models exhibit unprecedented capabilities to generate extremely realistic data. However, given the inherent compositionality of the real world, reliable use of these models in practical applications requires that they exhibit the capability to compose a novel set of concepts to generate outputs not seen in the training data set. Prior work demonstrates that recent diffusion models do exhibit intriguing compositional generalization abilities, but also fail unpredictably. Motivated by this, we perform a controlled study for understanding compositional generalization in conditional diffusion models in a synthetic setting, varying different attributes of the training data and measuring the model's ability to generate samples out-of-distribution. Our results show: (i) the order in which the ability to generate samples from a concept and compose them emerges is governed by the structure of the underlying data-generating process; (ii) performance on compositional tasks exhib
    
[^3]: 比较现代无参考图像和视频质量评估度量方法对对抗攻击的鲁棒性

    Comparing the robustness of modern no-reference image- and video-quality metrics to adversarial attacks. (arXiv:2310.06958v1 [cs.CV])

    [http://arxiv.org/abs/2310.06958](http://arxiv.org/abs/2310.06958)

    本文比较了现代图像和视频质量评估度量方法对抗攻击的鲁棒性，并发现部分度量方法对对抗攻击表现出较高的抵抗力，为基准测试提供了更安全的选择。

    

    如今，基于神经网络的图像和视频质量评估度量方法相比传统方法表现更好。然而，它们也变得更容易受到对抗性攻击，这些攻击可以提高度量分数但不改善视觉质量。现有的质量度量基准将其性能与主观质量相关性和计算时间进行比较。然而，图像质量度量的对抗鲁棒性也是一个值得研究的领域。本文分析了现代度量方法对不同对抗攻击的鲁棒性。我们采用了计算机视觉任务中的对抗攻击，并比较了这些攻击对15个无参考图像/视频质量度量方法的效果。一些度量方法对对抗攻击表现出了较高的抵抗力，使它们在基准测试中的使用比容易受攻击的方法更安全。该基准测试接受研究人员提交新的度量方法，以使他们的方法对攻击更加鲁棒，或者为他们寻找符合需求的鲁棒度量方法。

    Nowadays neural-network-based image- and video-quality metrics show better performance compared to traditional methods. However, they also became more vulnerable to adversarial attacks that increase metrics' scores without improving visual quality. The existing benchmarks of quality metrics compare their performance in terms of correlation with subjective quality and calculation time. However, the adversarial robustness of image-quality metrics is also an area worth researching. In this paper, we analyse modern metrics' robustness to different adversarial attacks. We adopted adversarial attacks from computer vision tasks and compared attacks' efficiency against 15 no-reference image/video-quality metrics. Some metrics showed high resistance to adversarial attacks which makes their usage in benchmarks safer than vulnerable metrics. The benchmark accepts new metrics submissions for researchers who want to make their metrics more robust to attacks or to find such metrics for their needs. 
    
[^4]: 离线模仿学习与变分逆向推理

    Offline Imitation Learning with Variational Counterfactual Reasoning. (arXiv:2310.04706v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.04706](http://arxiv.org/abs/2310.04706)

    该论文提出了一个名为OILCA的框架，利用可识别的变分自动编码器生成"对抗性"样本，以解决离线模仿学习中数据稀缺、环境变化等问题。

    

    在离线模仿学习中，智能体旨在学习一种最优的专家行为策略，而不需要额外的在线环境交互。然而，在许多真实场景中，例如机器人操作中，离线数据集是从没有奖励的次优行为中收集来的。由于专家数据稀缺，智能体通常只能简单地记住贫乏的轨迹，并且容易受到环境变化的影响，缺乏对新环境的泛化能力。为了有效地消除会对智能体造成偏差并阻碍泛化的伪特征，我们提出了一个名为OILCA的框架，即离线模仿学习与对抗数据增强。具体来说，我们利用可识别的变分自动编码器生成"对抗性"样本。我们从理论上分析了对抗性识别和泛化的改善。

    In offline Imitation Learning (IL), an agent aims to learn an optimal expert behavior policy without additional online environment interactions. However, in many real-world scenarios, such as robotics manipulation, the offline dataset is collected from suboptimal behaviors without rewards. Due to the scarce expert data, the agents usually suffer from simply memorizing poor trajectories and are vulnerable to the variations in the environments, lacking the capability of generalizing to new environments. To effectively remove spurious features that would otherwise bias the agent and hinder generalization, we propose a framework named \underline{O}ffline \underline{I}mitation \underline{L}earning with \underline{C}ounterfactual data \underline{A}ugmentation (OILCA). In particular, we leverage the identifiable variational autoencoder to generate \textit{counterfactual} samples. We theoretically analyze the counterfactual identification and the improvement of generalization. Moreover, we con
    
[^5]: 增强鲁棒性的带对抗特征抑制的提升建模

    Robustness-enhanced Uplift Modeling with Adversarial Feature Desensitization. (arXiv:2310.04693v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.04693](http://arxiv.org/abs/2310.04693)

    本文提出了一种增强鲁棒性的提升建模框架RUAD，并通过特征选择和对抗特征抑制两个定制模块更有效地解决了提升模型的特征敏感性问题。

    

    提升建模在在线营销中展示了非常有希望的结果。然而，大多数现有的工作在一些实际应用中容易受到鲁棒性挑战的影响。本文首先对上述现象给出了一个可能的解释。我们使用不同的真实世界数据集验证了在线营销中存在特征敏感性问题，一些关键特征的扰动会严重影响提升模型的性能，甚至导致相反的趋势。为了解决上述问题，我们提出了一种新颖的通过对抗特征抑制增强鲁棒性的提升建模框架（RUAD）。具体来说，我们的RUAD通过两个定制模块更有效地减轻提升模型的特征敏感性，包括一个具有联合多标签建模的特征选择模块，以从输入特征中识别一个关键子集，以及一个采用对抗训练和软插值操作的对抗特征抑制模块。

    Uplift modeling has shown very promising results in online marketing. However, most existing works are prone to the robustness challenge in some practical applications. In this paper, we first present a possible explanation for the above phenomenon. We verify that there is a feature sensitivity problem in online marketing using different real-world datasets, where the perturbation of some key features will seriously affect the performance of the uplift model and even cause the opposite trend. To solve the above problem, we propose a novel robustness-enhanced uplift modeling framework with adversarial feature desensitization (RUAD). Specifically, our RUAD can more effectively alleviate the feature sensitivity of the uplift model through two customized modules, including a feature selection module with joint multi-label modeling to identify a key subset from the input features and an adversarial feature desensitization module using adversarial training and soft interpolation operations t
    
[^6]: 使用大型语言模型的图神经提示

    Graph Neural Prompting with Large Language Models. (arXiv:2309.15427v1 [cs.CL])

    [http://arxiv.org/abs/2309.15427](http://arxiv.org/abs/2309.15427)

    本文提出了一种名为图神经提示（GNP）的方法，可以帮助大型语言模型从知识图中学习有益的知识，以弥补它们在准确捕捉和返回基于知识的信息方面的固有限制。

    

    大型语言模型（LLMs）在各种语言建模任务中表现出了卓越的泛化能力和出色的性能，但它们在准确捕捉和返回基于知识的信息方面仍存在固有限制。现有的研究已经探索了利用知识图来通过联合训练和定制模型架构增强语言建模，但是将此应用于LLMs存在参数数量庞大和计算成本高的问题。此外，如何利用预训练的LLMs并避免从头开始训练自定义模型仍然是一个开放的问题。在这项工作中，我们提出了图神经提示（GNP），一种新颖的即插即用方法，可以帮助预训练的LLMs从知识图中学习有益的知识。GNP包括各种设计，包括标准的图神经网络编码器、跨模态汇聚模块、域投影器和自监督链接预测目标。在多个实验中展示了GNP的有效性。

    Large Language Models (LLMs) have shown remarkable generalization capability with exceptional performance in various language modeling tasks. However, they still exhibit inherent limitations in precisely capturing and returning grounded knowledge. While existing work has explored utilizing knowledge graphs to enhance language modeling via joint training and customized model architectures, applying this to LLMs is problematic owing to their large number of parameters and high computational cost. In addition, how to leverage the pre-trained LLMs and avoid training a customized model from scratch remains an open question. In this work, we propose Graph Neural Prompting (GNP), a novel plug-and-play method to assist pre-trained LLMs in learning beneficial knowledge from KGs. GNP encompasses various designs, including a standard graph neural network encoder, a cross-modality pooling module, a domain projector, and a self-supervised link prediction objective. Extensive experiments on multiple
    
[^7]: 最大扩散强化学习

    Maximum Diffusion Reinforcement Learning. (arXiv:2309.15293v1 [cs.LG])

    [http://arxiv.org/abs/2309.15293](http://arxiv.org/abs/2309.15293)

    最大扩散强化学习是一种克服强化学习中数据相关性问题的方法，通过解耦代理的经验实现持续学习，并在各种测试中表现出色。

    

    所有机器学习都建立在数据独立且同分布的假设上。然而，在强化学习中，当数据是依次从代理经验中收集而来时，这一假设通常不成立。因此，我们提出了一种名为最大扩散强化学习的方法，利用统计力学中的遍历过程来克服这些限制。我们的方法通过解耦代理的经验，可证明地使代理在单次部署中能够持续学习，而不受初始化方式的影响。此外，我们证明了我们的方法推广了众所周知的最大熵技术，并且通过在流行的基准测试中稳定超过了最先进的性能水平。我们的研究成果极大地促进了物理学、学习和控制的交叉领域，为强化学习代理（如行走机器人和自动驾驶汽车）的透明可靠决策提供了一条道路。

    The assumption that data are independent and identically distributed underpins all machine learning. When data are collected sequentially from agent experiences this assumption does not generally hold, as in reinforcement learning. Here, we derive a method that overcomes these limitations by exploiting the statistical mechanics of ergodic processes, which we term maximum diffusion reinforcement learning. By decorrelating agent experiences, our approach provably enables agents to learn continually in single-shot deployments regardless of how they are initialized. Moreover, we prove our approach generalizes well-known maximum entropy techniques, and show that it robustly exceeds state-of-the-art performance across popular benchmarks. Our results at the nexus of physics, learning, and control pave the way towards more transparent and reliable decision-making in reinforcement learning agents, such as locomoting robots and self-driving cars.
    
[^8]: 快速Slate策略优化：超越Plackett-Luce

    Fast Slate Policy Optimization: Going Beyond Plackett-Luce. (arXiv:2308.01566v1 [cs.LG])

    [http://arxiv.org/abs/2308.01566](http://arxiv.org/abs/2308.01566)

    本文介绍了一种快速Slate策略优化方法，通过提出一种新的策略类，可以在大规模决策系统中有效地优化任意奖励函数，结果表明该方法在百万级别动作空间问题上具有很好的效果。

    

    大规模机器学习系统中一个越来越重要的构建模块是返回Slate，即给定一个查询返回有序的项目列表。该技术的应用包括搜索、信息检索和推荐系统。当行动空间很大时，决策系统会限制在特定结构中以快速完成在线查询。本文解决了这些大规模决策系统在给定任意奖励函数下的优化问题。我们将这个学习问题转化为策略优化框架，并提出了一种新的策略类，它源于决策函数的一种新颖放松。这导致了一个简单而高效的学习算法，可以扩展到大规模的动作空间。我们将我们的方法与常用的Plackett-Luce策略类进行比较，并展示了我们的方法在动作空间大小达到百万级别的问题上的有效性。

    An increasingly important building block of large scale machine learning systems is based on returning slates; an ordered lists of items given a query. Applications of this technology include: search, information retrieval and recommender systems. When the action space is large, decision systems are restricted to a particular structure to complete online queries quickly. This paper addresses the optimization of these large scale decision systems given an arbitrary reward function. We cast this learning problem in a policy optimization framework and propose a new class of policies, born from a novel relaxation of decision functions. This results in a simple, yet efficient learning algorithm that scales to massive action spaces. We compare our method to the commonly adopted Plackett-Luce policy class and demonstrate the effectiveness of our approach on problems with action space sizes in the order of millions.
    
[^9]: VillanDiffusion: 一种针对扩散模型的统一后门攻击框架。

    VillanDiffusion: A Unified Backdoor Attack Framework for Diffusion Models. (arXiv:2306.06874v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2306.06874](http://arxiv.org/abs/2306.06874)

    本文提出VillanDiffusion，一个针对扩散模型的统一后门攻击框架，涵盖主流的无条件和有条件DM，便于对不同DM配置进行后门分析，并为基于字幕的DM后门攻击提供了新的见解。

    

    扩散模型（DM）是最先进的生成模型之一，它通过迭代添加噪声和去噪学习可逆的损坏过程。它们是许多生成人工智能应用的主干，例如文本到图像有条件生成。但是，最近的研究表明基本无条件DM（例如DDPM和DDIM）易受后门注入攻击，这是一种由于恶意嵌入模型输入的模式而触发的输出操纵攻击。本文提出了一个统一的后门攻击框架（VillanDiffusion），以扩展当前的DM后门分析范围。我们的框架涵盖了主流的无条件和有条件DM（基于去噪和基于评分），以及各种无需训练的采样器进行整体评估。实验证明，我们的统一框架便于对不同DM配置进行后门分析，并为基于字幕的DM后门攻击提供新的见解。

    Diffusion Models (DMs) are state-of-the-art generative models that learn a reversible corruption process from iterative noise addition and denoising. They are the backbone of many generative AI applications, such as text-to-image conditional generation. However, recent studies have shown that basic unconditional DMs (e.g., DDPM and DDIM) are vulnerable to backdoor injection, a type of output manipulation attack triggered by a maliciously embedded pattern at model input. This paper presents a unified backdoor attack framework (VillanDiffusion) to expand the current scope of backdoor analysis for DMs. Our framework covers mainstream unconditional and conditional DMs (denoising-based and score-based) and various training-free samplers for holistic evaluations. Experiments show that our unified framework facilitates the backdoor analysis of different DM configurations and provides new insights into caption-based backdoor attacks on DMs.
    
[^10]: 可微的地球移动距离在高亮LHC数据压缩中的应用

    Differentiable Earth Mover's Distance for Data Compression at the High-Luminosity LHC. (arXiv:2306.04712v1 [hep-ex])

    [http://arxiv.org/abs/2306.04712](http://arxiv.org/abs/2306.04712)

    本文利用可微分的快速逼近方法，训练了一个编码器神经网络用于高亮LHC数据的压缩，同时保留了数据内与粒子探测器中的能量沉积分布相关的信息。

    

    地球移动距离(EMD)是图像识别和分类的有用指标，但其通常实现不可微分或过于缓慢，无法用作通过梯度下降训练其他算法的损失函数。本文训练了一个卷积神经网络(CNN)，学习了可微分的、快速的EMD的逼近方法，并证明它可以用作计算密集的EMD实现的替代品。我们将这种可微分的逼近方法应用于用于数据压缩的类自编码神经网络(encoder NN)的训练，这些数据来自欧洲核子研究组织的高亮LHC。编码器NN的目标是在保留与粒子探测器中的能量沉积分布相关的信息的同时压缩数据。我们证明，使用可微的EMD CNN训练的编码器NN的性能超越基于平均平方误差的损失函数的训练。

    The Earth mover's distance (EMD) is a useful metric for image recognition and classification, but its usual implementations are not differentiable or too slow to be used as a loss function for training other algorithms via gradient descent. In this paper, we train a convolutional neural network (CNN) to learn a differentiable, fast approximation of the EMD and demonstrate that it can be used as a substitute for computing-intensive EMD implementations. We apply this differentiable approximation in the training of an autoencoder-inspired neural network (encoder NN) for data compression at the high-luminosity LHC at CERN. The goal of this encoder NN is to compress the data while preserving the information related to the distribution of energy deposits in particle detectors. We demonstrate that the performance of our encoder NN trained using the differentiable EMD CNN surpasses that of training with loss functions based on mean squared error.
    
[^11]: 基于可编程多操作多模干涉的机器学习光子张量核心——M3ICRO

    M3ICRO: Machine Learning-Enabled Compact Photonic Tensor Core based on PRogrammable Multi-Operand Multimode Interference. (arXiv:2305.19505v1 [cs.ET])

    [http://arxiv.org/abs/2305.19505](http://arxiv.org/abs/2305.19505)

    M3ICRO是一种基于定制MOMMI器件的机器学习光子张量核心，具有超高能效、紧凑型设计和ML for optics优化方法，可以用于加速图像识别、自然语言处理等多种ML任务。

    

    光子计算对于机器学习加速具有重要作用，其具有超快速度、大规模并行和高能效等优势。然而，当前基于标准光学元件的光子张量核心（PTC）设计由于其较大的空间占用面积而限制了其可扩展性和计算密度。为解决这个问题，我们提出了一款利用定制可编程多操作多模干涉（MOMMI）器件的超紧凑型PTC，名为M3ICRO。可编程的MOMMI利用光的固有传播原理，提供单设备可编程矩阵单元，超越了传统计算范式中每个设备一个乘积累加（MAC）操作的限制。为了解决常规优化技术对定制器件的优化困难，通常需要耗费大量时间进行模拟，我们使用ML for optics来预测器件行为并实现可微分的优化流程。我们全面研究了M3ICRO的可重构性和矩阵表现力，展示了它相对于现有设计的能效提升，并展示了其在加速图像识别和自然语言处理等各种机器学习任务中的潜力。

    Photonic computing shows promise for transformative advancements in machine learning (ML) acceleration, offering ultra-fast speed, massive parallelism, and high energy efficiency. However, current photonic tensor core (PTC) designs based on standard optical components hinder scalability and compute density due to their large spatial footprint. To address this, we propose an ultra-compact PTC using customized programmable multi-operand multimode interference (MOMMI) devices, named M3ICRO. The programmable MOMMI leverages the intrinsic light propagation principle, providing a single-device programmable matrix unit beyond the conventional computing paradigm of one multiply-accumulate (MAC) operation per device. To overcome the optimization difficulty of customized devices that often requires time-consuming simulation, we apply ML for optics to predict the device behavior and enable a differentiable optimization flow. We thoroughly investigate the reconfigurability and matrix expressivity 
    
[^12]: 具有预测误差保证的分布式离线策略评估算法

    Distributional Offline Policy Evaluation with Predictive Error Guarantees. (arXiv:2302.09456v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.09456](http://arxiv.org/abs/2302.09456)

    本论文提出了一种名为Fitted Likelihood Estimation (FLE)的算法来解决分布式离线策略评估的问题，该算法能够学习到密切接近真实分布的策略回报分布。

    

    本研究探讨使用非策略生成的离线数据集来估算策略回报分布的问题，即分布式离线策略评估（OPE）。提出了一种名为Fitted Likelihood Estimation（FLE）的算法，它执行了一系列的最大似然估计，具有将任何最先进的概率生成模型集成的灵活性，只要它可以通过最大似然估计进行训练。FLE能够用于有限或无限时间折扣设置，其中奖励可以是多维向量。我们的理论结果表明，无论是在有限时间折扣设置还是无限时间折扣设置下，FLE都可以学习到密切接近真实分布的分布，分别在总变差距离和Wasserstein距离下。在训练MLE过程成功时，我们的理论结果适用于离线数据覆盖测试策略痕迹的条件。在实验上，我们证明了FLE在各种环境中都能取得良好的效果。

    We study the problem of estimating the distribution of the return of a policy using an offline dataset that is not generated from the policy, i.e., distributional offline policy evaluation (OPE). We propose an algorithm called Fitted Likelihood Estimation (FLE), which conducts a sequence of Maximum Likelihood Estimation (MLE) and has the flexibility of integrating any state-of-the-art probabilistic generative models as long as it can be trained via MLE. FLE can be used for both finite-horizon and infinite-horizon discounted settings where rewards can be multi-dimensional vectors. Our theoretical results show that for both finite-horizon and infinite-horizon discounted settings, FLE can learn distributions that are close to the ground truth under total variation distance and Wasserstein distance, respectively. Our theoretical results hold under the conditions that the offline data covers the test policy's traces and that the supervised learning MLE procedures succeed. Experimentally, we
    


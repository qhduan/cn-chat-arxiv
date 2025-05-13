# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Integrating Large Language Models in Causal Discovery: A Statistical Causal Approach](https://rss.arxiv.org/abs/2402.01454) | 本文提出了一种在因果发现中集成大型语言模型的方法，通过将统计因果提示与知识增强相结合，可以使统计因果发现结果接近真实情况并进一步改进结果。 |
| [^2] | [Stochastic Halpern iteration in normed spaces and applications to reinforcement learning](https://arxiv.org/abs/2403.12338) | 该论文分析了随机Halpern迭代在赋范空间中的Oracle复杂度，提出了改进的算法复杂度，进而在强化学习中提出了新的同步算法应用。 |
| [^3] | [Differential Privacy of Noisy (S)GD under Heavy-Tailed Perturbations](https://arxiv.org/abs/2403.02051) | 在重尾扰动下，噪声SGD实现了差分隐私保证，适用于广泛的损失函数类，特别是非凸函数。 |
| [^4] | [Revealing Multimodal Contrastive Representation Learning through Latent Partial Causal Models](https://arxiv.org/abs/2402.06223) | 通过潜在部分因果模型，我们展示了多模式对比表示学习在识别潜在耦合变量方面的优秀能力，并揭示了预训练的多模态模型通过线性独立分量分析学习分离表示的潜力。 |
| [^5] | [Nonlinear functional regression by functional deep neural network with kernel embedding.](http://arxiv.org/abs/2401.02890) | 本文提出了一种函数深度神经网络用于非线性函数回归的方法，通过平滑核积分变换和数据相关的维度缩减方法，取得了良好的预测效果。 |
| [^6] | [Fundamental Limits of Membership Inference Attacks on Machine Learning Models.](http://arxiv.org/abs/2310.13786) | 本文探讨了机器学习模型上成员推断攻击的基本限制，包括推导了效果和成功率的统计量，并提供了几种情况下的界限。这使得我们能够根据样本数量和其他结构参数推断潜在攻击的准确性。 |
| [^7] | [Towards Understanding Sycophancy in Language Models.](http://arxiv.org/abs/2310.13548) | 这项研究探讨了强化学习从人类反馈中训练高质量AI助手的技术，发现这种方法可能导致模型在回答问题时过于谄媚，而不是坦诚，通过分析人类偏好数据得出了这一结论。 |

# 详细

[^1]: 在因果发现中集成大型语言模型: 一种统计因果方法

    Integrating Large Language Models in Causal Discovery: A Statistical Causal Approach

    [https://rss.arxiv.org/abs/2402.01454](https://rss.arxiv.org/abs/2402.01454)

    本文提出了一种在因果发现中集成大型语言模型的方法，通过将统计因果提示与知识增强相结合，可以使统计因果发现结果接近真实情况并进一步改进结果。

    

    在实际的统计因果发现（SCD）中，将领域专家知识作为约束嵌入到算法中被广泛接受，因为这对于创建一致有意义的因果模型是重要的，尽管识别背景知识的挑战被认可。为了克服这些挑战，本文提出了一种新的因果推断方法，即通过将LLM的“统计因果提示（SCP）”与SCD方法和基于知识的因果推断（KBCI）相结合，对SCD进行先验知识增强。实验证明，GPT-4可以使LLM-KBCI的输出与带有LLM-KBCI的先验知识的SCD结果接近真实情况，如果GPT-4经历了SCP，那么SCD的结果还可以进一步改善。而且，即使LLM不含有数据集的信息，LLM仍然可以通过其背景知识来改进SCD。

    In practical statistical causal discovery (SCD), embedding domain expert knowledge as constraints into the algorithm is widely accepted as significant for creating consistent meaningful causal models, despite the recognized challenges in systematic acquisition of the background knowledge. To overcome these challenges, this paper proposes a novel methodology for causal inference, in which SCD methods and knowledge based causal inference (KBCI) with a large language model (LLM) are synthesized through "statistical causal prompting (SCP)" for LLMs and prior knowledge augmentation for SCD. Experiments have revealed that GPT-4 can cause the output of the LLM-KBCI and the SCD result with prior knowledge from LLM-KBCI to approach the ground truth, and that the SCD result can be further improved, if GPT-4 undergoes SCP. Furthermore, it has been clarified that an LLM can improve SCD with its background knowledge, even if the LLM does not contain information on the dataset. The proposed approach
    
[^2]: 随机Halpern迭代在赋范空间中的应用及其在强化学习中的应用

    Stochastic Halpern iteration in normed spaces and applications to reinforcement learning

    [https://arxiv.org/abs/2403.12338](https://arxiv.org/abs/2403.12338)

    该论文分析了随机Halpern迭代在赋范空间中的Oracle复杂度，提出了改进的算法复杂度，进而在强化学习中提出了新的同步算法应用。

    

    我们分析了具有方差减少的随机Halpern迭代的Oracle复杂度，旨在近似有界和收缩算子的不动点在一个有限维赋范空间中。我们表明，如果底层的随机Oracle具有一致有界的方差，则我们的方法展现出总的Oracle复杂度为$ \tilde{O} (\varepsilon^{-5})$，改进了最近为随机Krasnoselskii-Mann迭代建立的速率。此外，我们建立了 $\Omega (\varepsilon^{-3})$的下界，适用于广泛范围的算法，包括所有带有小批处理的平均迭代。通过适当修改我们的方法，我们推导出了在算子为 $\gamma$-收缩的情况下一个 $O(\varepsilon^{-2}(1-\gamma)^{-3})$复杂度上界。作为一个应用，我们提出了新的用于平均奖励和折扣奖励马尔可夫决策过程的同步算法。

    arXiv:2403.12338v1 Announce Type: cross  Abstract: We analyze the oracle complexity of the stochastic Halpern iteration with variance reduction, where we aim to approximate fixed-points of nonexpansive and contractive operators in a normed finite-dimensional space. We show that if the underlying stochastic oracle is with uniformly bounded variance, our method exhibits an overall oracle complexity of $\tilde{O}(\varepsilon^{-5})$, improving recent rates established for the stochastic Krasnoselskii-Mann iteration. Also, we establish a lower bound of $\Omega(\varepsilon^{-3})$, which applies to a wide range of algorithms, including all averaged iterations even with minibatching. Using a suitable modification of our approach, we derive a $O(\varepsilon^{-2}(1-\gamma)^{-3})$ complexity bound in the case in which the operator is a $\gamma$-contraction. As an application, we propose new synchronous algorithms for average reward and discounted reward Markov decision processes. In particular, f
    
[^3]: 在重尾扰动下噪声(S)GD的差分隐私

    Differential Privacy of Noisy (S)GD under Heavy-Tailed Perturbations

    [https://arxiv.org/abs/2403.02051](https://arxiv.org/abs/2403.02051)

    在重尾扰动下，噪声SGD实现了差分隐私保证，适用于广泛的损失函数类，特别是非凸函数。

    

    将重尾噪声注入随机梯度下降(SGD)的迭代中已经引起越来越多的关注。尽管对导致的算法的各种理论性质进行了分析，主要来自学习理论和优化视角，但它们的隐私保护性质尚未建立。为了弥补这一缺口，我们为噪声SGD提供差分隐私(DP)保证，当注入的噪声遵循$\alpha$-稳定分布时，该分布包括一系列重尾分布(具有无限方差)以及高斯分布。考虑$(\epsilon,\delta)$-DP框架，我们表明带有重尾扰动的SGD实现了$(0,\tilde{\mathcal{O}}(1/n))$-DP的广泛损失函数类，这些函数可以是非凸的，这里$n$是数据点的数量。作为一项显着的副产品，与以往的工作相反，该工作要求有界se

    arXiv:2403.02051v1 Announce Type: cross  Abstract: Injecting heavy-tailed noise to the iterates of stochastic gradient descent (SGD) has received increasing attention over the past few years. While various theoretical properties of the resulting algorithm have been analyzed mainly from learning theory and optimization perspectives, their privacy preservation properties have not yet been established. Aiming to bridge this gap, we provide differential privacy (DP) guarantees for noisy SGD, when the injected noise follows an $\alpha$-stable distribution, which includes a spectrum of heavy-tailed distributions (with infinite variance) as well as the Gaussian distribution. Considering the $(\epsilon, \delta)$-DP framework, we show that SGD with heavy-tailed perturbations achieves $(0, \tilde{\mathcal{O}}(1/n))$-DP for a broad class of loss functions which can be non-convex, where $n$ is the number of data points. As a remarkable byproduct, contrary to prior work that necessitates bounded se
    
[^4]: 通过潜在部分因果模型揭示多模式对比表示学习

    Revealing Multimodal Contrastive Representation Learning through Latent Partial Causal Models

    [https://arxiv.org/abs/2402.06223](https://arxiv.org/abs/2402.06223)

    通过潜在部分因果模型，我们展示了多模式对比表示学习在识别潜在耦合变量方面的优秀能力，并揭示了预训练的多模态模型通过线性独立分量分析学习分离表示的潜力。

    

    多模式对比表示学习方法在各个领域取得了成功，部分原因是由于它们能够生成复杂现象的有意义的共享表示。为了增强对这些获得的表示的深度分析和理解，我们引入了一种特别针对多模态数据设计的统一因果模型。通过研究这个模型，我们展示了多模式对比表示学习在识别在提出的统一模型中的潜在耦合变量方面的优秀能力，即使在不同假设下导致的线性或置换变换。我们的发现揭示了预训练的多模态模型（如CLIP）通过线性独立分量分析这一令人惊讶的简单而高效的工具学习分离表示的潜力。实验证明了我们发现的鲁棒性，即使在被违反假设的情况下，也验证了所提出方法在学习疾病方面的有效性。

    Multimodal contrastive representation learning methods have proven successful across a range of domains, partly due to their ability to generate meaningful shared representations of complex phenomena. To enhance the depth of analysis and understanding of these acquired representations, we introduce a unified causal model specifically designed for multimodal data. By examining this model, we show that multimodal contrastive representation learning excels at identifying latent coupled variables within the proposed unified model, up to linear or permutation transformations resulting from different assumptions. Our findings illuminate the potential of pre-trained multimodal models, eg, CLIP, in learning disentangled representations through a surprisingly simple yet highly effective tool: linear independent component analysis. Experiments demonstrate the robustness of our findings, even when the assumptions are violated, and validate the effectiveness of the proposed method in learning dise
    
[^5]: 函数深度神经网络在非线性函数回归中的应用

    Nonlinear functional regression by functional deep neural network with kernel embedding. (arXiv:2401.02890v1 [stat.ML])

    [http://arxiv.org/abs/2401.02890](http://arxiv.org/abs/2401.02890)

    本文提出了一种函数深度神经网络用于非线性函数回归的方法，通过平滑核积分变换和数据相关的维度缩减方法，取得了良好的预测效果。

    

    随着深度学习在语音识别、图像分类和自然语言处理等领域的迅速发展，它也被广泛应用于函数数据分析中。然而，由于无限维的输入，我们需要一个强大的维度缩减方法来处理非线性函数回归任务。在本文中，基于平滑核积分变换的思想，我们提出了一种具有高效且完全数据依赖的维度缩减方法的函数深度神经网络。我们的函数网络由以下步骤组成：核嵌入步骤：利用数据相关的平滑核进行积分变换；投影步骤：通过基于嵌入核的特征函数基底进行维度缩减；最后是一个表达丰富的深度ReLU神经网络进行预测。

    With the rapid development of deep learning in various fields of science and technology, such as speech recognition, image classification, and natural language processing, recently it is also widely applied in the functional data analysis (FDA) with some empirical success. However, due to the infinite dimensional input, we need a powerful dimension reduction method for functional learning tasks, especially for the nonlinear functional regression. In this paper, based on the idea of smooth kernel integral transformation, we propose a functional deep neural network with an efficient and fully data-dependent dimension reduction method. The architecture of our functional net consists of a kernel embedding step: an integral transformation with a data-dependent smooth kernel; a projection step: a dimension reduction by projection with eigenfunction basis based on the embedding kernel; and finally an expressive deep ReLU neural network for the prediction. The utilization of smooth kernel embe
    
[^6]: 机器学习模型的成员推断攻击的基本限制

    Fundamental Limits of Membership Inference Attacks on Machine Learning Models. (arXiv:2310.13786v1 [stat.ML])

    [http://arxiv.org/abs/2310.13786](http://arxiv.org/abs/2310.13786)

    本文探讨了机器学习模型上成员推断攻击的基本限制，包括推导了效果和成功率的统计量，并提供了几种情况下的界限。这使得我们能够根据样本数量和其他结构参数推断潜在攻击的准确性。

    

    成员推断攻击（MIA）可以揭示特定数据点是否是训练数据集的一部分，可能暴露个人的敏感信息。本文探讨了关于机器学习模型上MIA的基本统计限制。具体而言，我们首先推导了统计量，该统计量决定了这种攻击的有效性和成功率。然后，我们研究了几种情况，并对这个感兴趣的统计量提供了界限。这使我们能够根据样本数量和学习模型的其他结构参数推断潜在攻击的准确性，在某些情况下可以直接从数据集中估计。

    Membership inference attacks (MIA) can reveal whether a particular data point was part of the training dataset, potentially exposing sensitive information about individuals. This article explores the fundamental statistical limitations associated with MIAs on machine learning models. More precisely, we first derive the statistical quantity that governs the effectiveness and success of such attacks. Then, we investigate several situations for which we provide bounds on this quantity of interest. This allows us to infer the accuracy of potential attacks as a function of the number of samples and other structural parameters of learning models, which in some cases can be directly estimated from the dataset.
    
[^7]: 探索语言模型中谄媚行为的理解

    Towards Understanding Sycophancy in Language Models. (arXiv:2310.13548v1 [cs.CL])

    [http://arxiv.org/abs/2310.13548](http://arxiv.org/abs/2310.13548)

    这项研究探讨了强化学习从人类反馈中训练高质量AI助手的技术，发现这种方法可能导致模型在回答问题时过于谄媚，而不是坦诚，通过分析人类偏好数据得出了这一结论。

    

    「从人类反馈中进行强化学习（RLHF）」是训练高质量AI助手的一种流行技术。然而，RLHF可能会鼓励模型通过与用户信念相符的回答来代替真实回答，这种行为被称为谄媚行为。我们研究了RLHF训练模型中谄媚行为的普遍性以及人类偏好判断是否起到了作用。首先，我们证明了五个最先进的AI助手在四个不同的自由文本生成任务中一贯表现出谄媚行为。为了理解人类偏好是否驱动了RLHF模型的这种广泛行为，我们分析了现有的人类偏好数据。我们发现，当回答与用户的观点相符时，它更有可能被选中。此外，人类和偏好模型（PMs）将有说服力的谄媚回答与正确回答相比，有时几乎可以忽略不计地选择了谄媚回答。优化模型输出以满足PMs有时也会在真实性和谄媚行为之间做出取舍。

    Reinforcement learning from human feedback (RLHF) is a popular technique for training high-quality AI assistants. However, RLHF may also encourage model responses that match user beliefs over truthful responses, a behavior known as sycophancy. We investigate the prevalence of sycophancy in RLHF-trained models and whether human preference judgements are responsible. We first demonstrate that five state-of-the-art AI assistants consistently exhibit sycophantic behavior across four varied free-form text-generation tasks. To understand if human preferences drive this broadly observed behavior of RLHF models, we analyze existing human preference data. We find that when a response matches a user's views, it is more likely to be preferred. Moreover, both humans and preference models (PMs) prefer convincingly-written sycophantic responses over correct ones a negligible fraction of the time. Optimizing model outputs against PMs also sometimes sacrifices truthfulness in favor of sycophancy. Over
    


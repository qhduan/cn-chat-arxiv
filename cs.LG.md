# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Effectiveness of Distillation in Mitigating Backdoors in Pre-trained Encoder](https://arxiv.org/abs/2403.03846) | 研究了如何利用蒸馏从受污染的预训练编码器中提取良性知识，将其传递给新编码器，成功降低攻击成功率，并探讨了蒸馏的核心组件对性能的影响。 |
| [^2] | [TorchCP: A Library for Conformal Prediction based on PyTorch](https://arxiv.org/abs/2402.12683) | TorchCP是一个基于PyTorch的Python工具包，为深度学习模型上的合拟常规预测研究提供了实现后验和训练方法的多种工具，包括分类和回归任务。En_Tdlr: TorchCP is a Python toolbox built on PyTorch for conformal prediction research on deep learning models, providing various implementations for posthoc and training methods for classification and regression tasks, including multi-dimension output. |
| [^3] | [Generative quantum machine learning via denoising diffusion probabilistic models](https://arxiv.org/abs/2310.05866) | 通过引入量子去噪扩散概率模型（QuDDPM），我们实现了对量子数据的高效可训练的生成学习，该模型采用足够层数的电路以保证表达能力，并引入多个中间训练任务以避免贫瘠平原并保证高效的训练。 |
| [^4] | [Can Continual Learning Improve Long-Tailed Recognition? Toward a Unified Framework.](http://arxiv.org/abs/2306.13275) | 本文针对长尾识别问题，提出一种持续学习方法，通过将头部集和尾部集的学习视为两个独立连续的步骤，并利用定理证明持续学习可以有效地更新学习者的权重以学习尾部，同时不会忘记头部。 |
| [^5] | [A VAE Approach to Sample Multivariate Extremes.](http://arxiv.org/abs/2306.10987) | 本论文提出了一种用于抽样多元重尾分布的VAE方法，该方法可以模拟真实世界的多元极值情况，如河流水位，从而有助于评估未来可能出现的风险。 |
| [^6] | [Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning.](http://arxiv.org/abs/2302.02662) | 本文研究了一种名为GLAM的方法，通过功能基础设施建设，利用在线强化学习提高LLM代理程序的性能来实现LLMs与环境之间的对齐，解决决策问题。 |

# 详细

[^1]: 在缓解预训练编码器中后门问题中蒸馏的有效性研究

    On the Effectiveness of Distillation in Mitigating Backdoors in Pre-trained Encoder

    [https://arxiv.org/abs/2403.03846](https://arxiv.org/abs/2403.03846)

    研究了如何利用蒸馏从受污染的预训练编码器中提取良性知识，将其传递给新编码器，成功降低攻击成功率，并探讨了蒸馏的核心组件对性能的影响。

    

    在本文中，我们研究了一种用于SSL中防御受污染编码器的方法，叫做蒸馏，这个方法最初是用于监督学习中的防御机制。蒸馏旨在从给定模型（称为教师网络）中提炼知识，并将其传递给另一个模型（称为学生网络）。我们现在使用它从受污染的预训练编码器中提炼良性知识，并将其传递给一个新编码器，从而得到一个干净的预训练编码器。具体来说，我们对蒸馏对抗受污染编码器的有效性和性能进行了实证研究。我们使用了两种最先进的针对预训练图像编码器的后门攻击方法和四个常用的图像分类数据集，实验结果表明，蒸馏可以将攻击成功率从80.87%降低到27.51%，而准确率下降了6.35%。此外，我们研究了蒸馏的三个核心组件对性能的影响：教师网络、学生网络和

    arXiv:2403.03846v1 Announce Type: new  Abstract: In this paper, we study a defense against poisoned encoders in SSL called distillation, which is a defense used in supervised learning originally. Distillation aims to distill knowledge from a given model (a.k.a the teacher net) and transfer it to another (a.k.a the student net). Now, we use it to distill benign knowledge from poisoned pre-trained encoders and transfer it to a new encoder, resulting in a clean pre-trained encoder. In particular, we conduct an empirical study on the effectiveness and performance of distillation against poisoned encoders. Using two state-of-the-art backdoor attacks against pre-trained image encoders and four commonly used image classification datasets, our experimental results show that distillation can reduce attack success rate from 80.87% to 27.51% while suffering a 6.35% loss in accuracy. Moreover, we investigate the impact of three core components of distillation on performance: teacher net, student n
    
[^2]: TorchCP：基于PyTorch的一种适用于合拟常规预测的库

    TorchCP: A Library for Conformal Prediction based on PyTorch

    [https://arxiv.org/abs/2402.12683](https://arxiv.org/abs/2402.12683)

    TorchCP是一个基于PyTorch的Python工具包，为深度学习模型上的合拟常规预测研究提供了实现后验和训练方法的多种工具，包括分类和回归任务。En_Tdlr: TorchCP is a Python toolbox built on PyTorch for conformal prediction research on deep learning models, providing various implementations for posthoc and training methods for classification and regression tasks, including multi-dimension output.

    

    TorchCP是一个用于深度学习模型上的合拟常规预测研究的Python工具包。它包含了用于后验和训练方法的各种实现，用于分类和回归任务（包括多维输出）。TorchCP建立在PyTorch之上，并利用矩阵计算的优势，提供简洁高效的推理实现。该代码采用LGPL许可证，并在$\href{https://github.com/ml-stat-Sustech/TorchCP}{\text{this https URL}}$开源。

    arXiv:2402.12683v1 Announce Type: new  Abstract: TorchCP is a Python toolbox for conformal prediction research on deep learning models. It contains various implementations for posthoc and training methods for classification and regression tasks (including multi-dimension output). TorchCP is built on PyTorch (Paszke et al., 2019) and leverages the advantages of matrix computation to provide concise and efficient inference implementations. The code is licensed under the LGPL license and is open-sourced at $\href{https://github.com/ml-stat-Sustech/TorchCP}{\text{this https URL}}$.
    
[^3]: 通过去噪扩散概率模型进行生成性量子机器学习

    Generative quantum machine learning via denoising diffusion probabilistic models

    [https://arxiv.org/abs/2310.05866](https://arxiv.org/abs/2310.05866)

    通过引入量子去噪扩散概率模型（QuDDPM），我们实现了对量子数据的高效可训练的生成学习，该模型采用足够层数的电路以保证表达能力，并引入多个中间训练任务以避免贫瘠平原并保证高效的训练。

    

    深度生成模型是计算机视觉、文本生成和大型语言模型的关键技术。最近，由于其能够生成多样化和高质量的样本，以及结构灵活、训练简单的特点，去噪扩散概率模型（DDPMs）在许多计算机视觉任务中受到了广泛关注。量子生成模型利用纠缠和叠加的能力为学习经典和量子数据带来了新的见解。受经典模型的启发，我们提出了“量子去噪扩散概率模型”（QuDDPM），以实现对量子数据的高效可训练的生成学习。QuDDPM采用足够层数的电路来保证表达能力，同时引入多个中间训练任务，将目标分布与噪声之间的插值作为训练过程，以避免贫瘠平原并保证高效的训练。我们给出了学习误差的上界和...（未完待续）

    Deep generative models are key-enabling technology to computer vision, text generation and large language models. Denoising diffusion probabilistic models (DDPMs) have recently gained much attention due to their ability to generate diverse and high-quality samples in many computer vision tasks, as well as to incorporate flexible model architectures and relatively simple training scheme. Quantum generative models, empowered by entanglement and superposition, have brought new insight to learning classical and quantum data. Inspired by the classical counterpart, we propose the \emph{quantum denoising diffusion probabilistic model} (QuDDPM) to enable efficiently trainable generative learning of quantum data. QuDDPM adopts sufficient layers of circuits to guarantee expressivity, while introduces multiple intermediate training tasks as interpolation between the target distribution and noise to avoid barren plateau and guarantee efficient training. We provide bounds on the learning error and 
    
[^4]: 持续学习能改进长尾识别吗？走向统一框架

    Can Continual Learning Improve Long-Tailed Recognition? Toward a Unified Framework. (arXiv:2306.13275v1 [cs.LG])

    [http://arxiv.org/abs/2306.13275](http://arxiv.org/abs/2306.13275)

    本文针对长尾识别问题，提出一种持续学习方法，通过将头部集和尾部集的学习视为两个独立连续的步骤，并利用定理证明持续学习可以有效地更新学习者的权重以学习尾部，同时不会忘记头部。

    

    在高度不平衡的数据集中，不同类别之间的样本数量极度失衡会出现长尾识别（LTR）问题。LTR方法旨在准确地学习包含一个较大“头”集和一个较小“尾”集的数据集。我们提出了一个定理，假设损失函数是强凸的，那么完整数据集上训练的学习者的权重在同一个学习者严格训练头集时的权重上限之内。接下来，我们声称将头集和尾集的学习视为两个独立的连续步骤，持续学习（CL）方法可以有效地更新学习者的权重以学习尾部，而不会忘记头部。首先，我们使用玩具MNIST-LT数据集验证了我们的理论发现。接着，我们在两个标准LTR基准（CIFAR100-LT和CIFAR10-L）的多个不平衡变体上评估了几种CL策略的有效性。

    The Long-Tailed Recognition (LTR) problem emerges in the context of learning from highly imbalanced datasets, in which the number of samples among different classes is heavily skewed. LTR methods aim to accurately learn a dataset comprising both a larger Head set and a smaller Tail set. We propose a theorem where under the assumption of strong convexity of the loss function, the weights of a learner trained on the full dataset are within an upper bound of the weights of the same learner trained strictly on the Head. Next, we assert that by treating the learning of the Head and Tail as two separate and sequential steps, Continual Learning (CL) methods can effectively update the weights of the learner to learn the Tail without forgetting the Head. First, we validate our theoretical findings with various experiments on the toy MNIST-LT dataset. We then evaluate the efficacy of several CL strategies on multiple imbalanced variations of two standard LTR benchmarks (CIFAR100-LT and CIFAR10-L
    
[^5]: 一种用于生成多元极值的VAE方法

    A VAE Approach to Sample Multivariate Extremes. (arXiv:2306.10987v1 [stat.ML])

    [http://arxiv.org/abs/2306.10987](http://arxiv.org/abs/2306.10987)

    本论文提出了一种用于抽样多元重尾分布的VAE方法，该方法可以模拟真实世界的多元极值情况，如河流水位，从而有助于评估未来可能出现的风险。

    

    当我们需要评估未来可能会出现的比已观察到的极值更大的极端情况的风险时，从观测数据集中准确地生成极值至关重要。 应用范围包括自然灾害和金融崩溃。 机器学习社区的生成方法不适用于极端样本，需要仔细适应。此外，极值理论的渐进结果提供了一个理论框架，尤其是通过多元正则变化的概念来模拟多元极端事件。 连接这两个领域，本文详细介绍了一种用于抽样多元重尾分布的变分自动编码器（VAE）方法，即可能具有特别大强度的极端分布。 我们在合成数据集和沿多瑙河网络的实际数据集上说明了我们方法的相关性。 后者显示了我们的方法通过从已观察到的极值分布中抽样来模拟河流水位的潜力。

    Generating accurate extremes from an observational data set is crucial when seeking to estimate risks associated with the occurrence of future extremes which could be larger than those already observed. Applications range from the occurrence of natural disasters to financial crashes. Generative approaches from the machine learning community do not apply to extreme samples without careful adaptation. Besides, asymptotic results from extreme value theory (EVT) give a theoretical framework to model multivariate extreme events, especially through the notion of multivariate regular variation. Bridging these two fields, this paper details a variational autoencoder (VAE) approach for sampling multivariate heavy-tailed distributions, i.e., distributions likely to have extremes of particularly large intensities. We illustrate the relevance of our approach on a synthetic data set and on a real data set of discharge measurements along the Danube river network. The latter shows the potential of ou
    
[^6]: 在交互环境中使用在线强化学习对大型语言模型进行基础设施建设

    Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning. (arXiv:2302.02662v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.02662](http://arxiv.org/abs/2302.02662)

    本文研究了一种名为GLAM的方法，通过功能基础设施建设，利用在线强化学习提高LLM代理程序的性能来实现LLMs与环境之间的对齐，解决决策问题。

    

    最近的研究成功地利用了大型语言模型（LLM）捕捉世界物理的抽象知识，以解决决策问题。然而，LLMs的知识与环境之间的对齐可能是错误的，并且由于缺乏基础设施建设而限制了其功能能力。在本文中，我们研究了一种通过功能基础设施建设实现这种对齐的方法（称为GLAM）：我们考虑一个使用LLM作为策略的代理程序，随着代理程序与环境进行交互而逐步更新，并利用在线强化学习来提高其解决目标的性能。使用一个交互式的文本环境设计来研究更高级形式的基础设施建设，以及一组空间和导航任务，我们研究了几个科学问题：1）LLMs能否提高各种RL任务的在线学习的样本效率？2）它如何提高不同形式的泛化？3）在线学习的影响是什么？我们通过功能方式研究这些问题。

    Recent works successfully leveraged Large Language Models' (LLM) abilities to capture abstract knowledge about world's physics to solve decision-making problems. Yet, the alignment between LLMs' knowledge and the environment can be wrong and limit functional competence due to lack of grounding. In this paper, we study an approach (named GLAM) to achieve this alignment through functional grounding: we consider an agent using an LLM as a policy that is progressively updated as the agent interacts with the environment, leveraging online Reinforcement Learning to improve its performance to solve goals. Using an interactive textual environment designed to study higher-level forms of functional grounding, and a set of spatial and navigation tasks, we study several scientific questions: 1) Can LLMs boost sample efficiency for online learning of various RL tasks? 2) How can it boost different forms of generalization? 3) What is the impact of online learning? We study these questions by functio
    


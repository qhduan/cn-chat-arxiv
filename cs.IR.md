# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dual-Channel Multiplex Graph Neural Networks for Recommendation](https://arxiv.org/abs/2403.11624) | 该研究提出了一种名为双通道多重图神经网络（DCMGNN）的新型推荐框架，能够有效解决现有推荐方法中存在的多通路关系行为模式建模和对目标关系影响忽略的问题。 |
| [^2] | [Diffusion Recommender Model.](http://arxiv.org/abs/2304.04971) | 本论文提出了一种新颖的扩散推荐模型（DiffRec）来逐步去噪地学习用户交互生成的过程，并针对推荐系统中的冷启动问题和稀疏数据等独特挑战进行了扩展，实验结果显示其在推荐准确性和稳健性方面优于现有方法。 |

# 详细

[^1]: 双通道多重图神经网络用于推荐

    Dual-Channel Multiplex Graph Neural Networks for Recommendation

    [https://arxiv.org/abs/2403.11624](https://arxiv.org/abs/2403.11624)

    该研究提出了一种名为双通道多重图神经网络（DCMGNN）的新型推荐框架，能够有效解决现有推荐方法中存在的多通路关系行为模式建模和对目标关系影响忽略的问题。

    

    高效的推荐系统在准确捕捉反映个人偏好的用户和项目属性方面发挥着至关重要的作用。一些现有的推荐技术已经开始将重点转向在真实世界的推荐场景中对用户和项目之间的各种类型交互关系进行建模，例如在线购物平台上的点击、标记收藏和购买。然而，这些方法仍然面临两个重要的缺点：(1) 不足的建模和利用用户和项目之间多通路关系形成的各种行为模式对表示学习的影响，以及(2) 忽略了行为模式中不同关系对推荐系统场景中目标关系的影响。在本研究中，我们介绍了一种新颖的推荐框架，即双通道多重图神经网络（DCMGNN），该框架解决了上述挑战。

    arXiv:2403.11624v1 Announce Type: cross  Abstract: Efficient recommender systems play a crucial role in accurately capturing user and item attributes that mirror individual preferences. Some existing recommendation techniques have started to shift their focus towards modeling various types of interaction relations between users and items in real-world recommendation scenarios, such as clicks, marking favorites, and purchases on online shopping platforms. Nevertheless, these approaches still grapple with two significant shortcomings: (1) Insufficient modeling and exploitation of the impact of various behavior patterns formed by multiplex relations between users and items on representation learning, and (2) ignoring the effect of different relations in the behavior patterns on the target relation in recommender system scenarios. In this study, we introduce a novel recommendation framework, Dual-Channel Multiplex Graph Neural Network (DCMGNN), which addresses the aforementioned challenges
    
[^2]: 扩散推荐模型

    Diffusion Recommender Model. (arXiv:2304.04971v1 [cs.IR])

    [http://arxiv.org/abs/2304.04971](http://arxiv.org/abs/2304.04971)

    本论文提出了一种新颖的扩散推荐模型（DiffRec）来逐步去噪地学习用户交互生成的过程，并针对推荐系统中的冷启动问题和稀疏数据等独特挑战进行了扩展，实验结果显示其在推荐准确性和稳健性方面优于现有方法。

    

    生成模型（如生成对抗网络（GANs）和变分自动编码器（VAEs））被广泛应用于建模用户交互的生成过程。然而，这些生成模型存在固有的局限性，如GANs的不稳定性和VAEs的受限表征能力。这些限制妨碍了复杂用户交互生成过程的准确建模，例如由各种干扰因素导致的嘈杂交互。考虑到扩散模型（DMs）在图像合成方面相对于传统的生成模型具有显着优势，我们提出了一种新颖的扩散推荐模型（称为DiffRec），以逐步去噪的方式学习生成过程。为了保留用户交互中的个性化信息，DiffRec减少了添加的噪声，并避免将用户交互损坏为像图像合成中的纯噪声。此外，我们扩展了传统的DMs以应对实际推荐系统中的独特挑战，如冷启动问题和稀疏的用户-物品交互数据。在几个真实数据集上的实验结果表明，DiffRec在推荐准确性和稳健性方面优于现有方法。

    Generative models such as Generative Adversarial Networks (GANs) and Variational Auto-Encoders (VAEs) are widely utilized to model the generative process of user interactions. However, these generative models suffer from intrinsic limitations such as the instability of GANs and the restricted representation ability of VAEs. Such limitations hinder the accurate modeling of the complex user interaction generation procedure, such as noisy interactions caused by various interference factors. In light of the impressive advantages of Diffusion Models (DMs) over traditional generative models in image synthesis, we propose a novel Diffusion Recommender Model (named DiffRec) to learn the generative process in a denoising manner. To retain personalized information in user interactions, DiffRec reduces the added noises and avoids corrupting users' interactions into pure noises like in image synthesis. In addition, we extend traditional DMs to tackle the unique challenges in practical recommender 
    


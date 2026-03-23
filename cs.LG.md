# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Multi-Perspective Machine Learning Approach to Evaluate Police-Driver Interaction in Los Angeles](https://arxiv.org/abs/2402.01703) | 该研究提出了一种多角度的机器学习方法，用于分析洛杉矶警察与司机的互动。该方法利用多模态的数据包括音频、视频和文字信息，旨在提供对复杂和有争议的警民互动的分析工具。 |
| [^2] | [Hyper-STTN: Social Group-aware Spatial-Temporal Transformer Network for Human Trajectory Prediction with Hypergraph Reasoning.](http://arxiv.org/abs/2401.06344) | 本论文提出了Hyper-STTN，一种基于超图的时空转换网络，用于人群轨迹预测。通过构建多尺度超图来捕捉拥挤场景中的群体间相互作用，并利用空间-时间转换器来捕捉行人的成对潜在相互作用。这些异构的群体间和成对间相互作用通过一个多模态转换网络进行融合和对准。 |
| [^3] | [torchgfn: A PyTorch GFlowNet library.](http://arxiv.org/abs/2305.14594) | torchgfn是一个基于PyTorch构建的GFlowNet库，提供了简单的API和有用的抽象，解决了不同代码库的约定问题，并且重现了已发表的结果。 |

# 详细

[^1]: 一种多角度的机器学习方法用于评估洛杉矶警察与司机的互动

    A Multi-Perspective Machine Learning Approach to Evaluate Police-Driver Interaction in Los Angeles

    [https://arxiv.org/abs/2402.01703](https://arxiv.org/abs/2402.01703)

    该研究提出了一种多角度的机器学习方法，用于分析洛杉矶警察与司机的互动。该方法利用多模态的数据包括音频、视频和文字信息，旨在提供对复杂和有争议的警民互动的分析工具。

    

    政府官员与市民之间的互动影响公共福祉和民主社会的正当性。警察是国家最显而易见、最接触市民的代理人，在交通站停期间，他们每年与公众互动超过2000万次。如今，这些互动经常被戴在身上的摄像机记录下来，这被视为提高警察问责制和改善警民互动的手段。然而，由于缺乏可靠的自动化工具来分析这些复杂而有争议的警民互动，这些记录的及时分析受到了阻碍。本文提出了一种新的多角度、多模态机器学习（ML）工具的方法，用于分析来自这些身上摄像机记录的音频、视频和文字信息。我们的方法首先确定与不同利益相关者最相关的沟通方面，包括共同感知互动的标志标记以及具有这些标记的符号。

    Interactions between the government officials and civilians affect public wellbeing and the state legitimacy that is necessary for the functioning of democratic society. Police officers, the most visible and contacted agents of the state, interact with the public more than 20 million times a year during traffic stops. Today, these interactions are regularly recorded by body-worn cameras (BWCs), which are lauded as a means to enhance police accountability and improve police-public interactions. However, the timely analysis of these recordings is hampered by a lack of reliable automated tools that can enable the analysis of these complex and contested police-public interactions. This article proposes an approach to developing new multi-perspective, multimodal machine learning (ML) tools to analyze the audio, video, and transcript information from this BWC footage. Our approach begins by identifying the aspects of communication most salient to different stakeholders, including both commun
    
[^2]: 超级-STTN：社交群体感知的时空转换网络用于人体轨迹预测与超图推理

    Hyper-STTN: Social Group-aware Spatial-Temporal Transformer Network for Human Trajectory Prediction with Hypergraph Reasoning. (arXiv:2401.06344v1 [cs.CV] CROSS LISTED)

    [http://arxiv.org/abs/2401.06344](http://arxiv.org/abs/2401.06344)

    本论文提出了Hyper-STTN，一种基于超图的时空转换网络，用于人群轨迹预测。通过构建多尺度超图来捕捉拥挤场景中的群体间相互作用，并利用空间-时间转换器来捕捉行人的成对潜在相互作用。这些异构的群体间和成对间相互作用通过一个多模态转换网络进行融合和对准。

    

    在各种现实世界的应用中，包括服务机器人和自动驾驶汽车，预测拥挤的意图和轨迹是至关重要的。理解环境动态是具有挑战性的，不仅因为对建模成对的空间和时间相互作用的复杂性，还因为群体间相互作用的多样性。为了解码拥挤场景中全面的成对和群体间相互作用，我们引入了Hyper-STTN，这是一种基于超图的时空转换网络，用于人群轨迹预测。在Hyper-STTN中，通过一组多尺度超图构建了拥挤的群体间相关性，这些超图具有不同的群体大小，通过基于随机游走概率的超图谱卷积进行捕捉。此外，还采用了空间-时间转换器来捕捉行人在空间-时间维度上的对照相互作用。然后，这些异构的群体间和成对间相互作用通过一个多模态转换网络进行融合和对准。

    Predicting crowded intents and trajectories is crucial in varouls real-world applications, including service robots and autonomous vehicles. Understanding environmental dynamics is challenging, not only due to the complexities of modeling pair-wise spatial and temporal interactions but also the diverse influence of group-wise interactions. To decode the comprehensive pair-wise and group-wise interactions in crowded scenarios, we introduce Hyper-STTN, a Hypergraph-based Spatial-Temporal Transformer Network for crowd trajectory prediction. In Hyper-STTN, crowded group-wise correlations are constructed using a set of multi-scale hypergraphs with varying group sizes, captured through random-walk robability-based hypergraph spectral convolution. Additionally, a spatial-temporal transformer is adapted to capture pedestrians' pair-wise latent interactions in spatial-temporal dimensions. These heterogeneous group-wise and pair-wise are then fused and aligned though a multimodal transformer net
    
[^3]: torchgfn：一个PyTorch GFlowNet库

    torchgfn: A PyTorch GFlowNet library. (arXiv:2305.14594v1 [cs.LG])

    [http://arxiv.org/abs/2305.14594](http://arxiv.org/abs/2305.14594)

    torchgfn是一个基于PyTorch构建的GFlowNet库，提供了简单的API和有用的抽象，解决了不同代码库的约定问题，并且重现了已发表的结果。

    

    随着生成流网络（GFlowNets或GFNs）的日益流行，代码来源也变得越来越多。这会妨碍实现新功能（例如训练损失），这些功能可以轻松地与现有功能进行比较，并在一组常见环境中进行使用。除了减缓GFlowNets领域的研究外，不同的代码库还使用不同的约定，可能会让新手感到困惑。torchgfn是一个基于PyTorch构建的库，旨在解决这两个问题。它为用户提供了简单的API和有用的抽象，以实现采样器和损失函数。提供了多个示例，重现了已发表的结果。该代码可在https://github.com/saleml/torchgfn中获取。

    The increasing popularity of generative flow networks (GFlowNets or GFNs) is accompanied with a proliferation of code sources. This hinders the implementation of new features, such as training losses, that can readily be compared to existing ones, on a set of common environments. In addition to slowing down research in the field of GFlowNets, different code bases use different conventions, that might be confusing for newcomers. `torchgfn` is a library built on top of PyTorch, that aims at addressing both problems. It provides user with a simple API for environments, and useful abstractions for samplers and losses. Multiple examples are provided, replicating published results. The code is available in https://github.com/saleml/torchgfn.
    


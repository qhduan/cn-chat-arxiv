# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bridging Evolutionary Algorithms and Reinforcement Learning: A Comprehensive Survey](https://arxiv.org/abs/2401.11963) | 通过整合进化算法与强化学习，进化强化学习（ERL）展现出卓越的性能提升，本综述呈现了ERL领域的各个研究分支，突出了EA辅助RL的优化、RL辅助EA的优化以及EA和RL的协同优化这三个主要研究方向。 |
| [^2] | [You Can Ground Earlier than See: An Effective and Efficient Pipeline for Temporal Sentence Grounding in Compressed Videos.](http://arxiv.org/abs/2303.07863) | 本文提出了一种新的压缩域TSG设置，通过直接编码压缩位流来增强视觉特征表示能力和提高时间句子对齐的效率，并且在两个基准数据集的实验中表现优于目前最先进的方法。 |

# 详细

[^1]: 跨越进化算法和强化学习：一项全面调查

    Bridging Evolutionary Algorithms and Reinforcement Learning: A Comprehensive Survey

    [https://arxiv.org/abs/2401.11963](https://arxiv.org/abs/2401.11963)

    通过整合进化算法与强化学习，进化强化学习（ERL）展现出卓越的性能提升，本综述呈现了ERL领域的各个研究分支，突出了EA辅助RL的优化、RL辅助EA的优化以及EA和RL的协同优化这三个主要研究方向。

    

    进化强化学习（ERL）将进化算法（EAs）和强化学习（RL）相结合进行优化，表现出卓越的性能提升。通过融合两种方法的优势，ERL已经成为一个有前景的研究方向。本调查综述了ERL中不同研究分支的全面概述。具体而言，我们系统总结了相关算法的最新进展，并确定了三个主要研究方向：EA辅助RL的优化，RL辅助EA的优化，以及EA和RL的协同优化。随后，我们深入分析了每个研究方向，组织了多个研究分支。我们阐明了每个分支致力于解决的问题，以及EA和RL的整合如何应对这些挑战。最后，我们讨论了潜在的挑战和未来的研究方向。

    arXiv:2401.11963v2 Announce Type: replace-cross  Abstract: Evolutionary Reinforcement Learning (ERL), which integrates Evolutionary Algorithms (EAs) and Reinforcement Learning (RL) for optimization, has demonstrated remarkable performance advancements. By fusing the strengths of both approaches, ERL has emerged as a promising research direction. This survey offers a comprehensive overview of the diverse research branches in ERL. Specifically, we systematically summarize recent advancements in relevant algorithms and identify three primary research directions: EA-assisted optimization of RL, RL-assisted optimization of EA, and synergistic optimization of EA and RL. Following that, we conduct an in-depth analysis of each research direction, organizing multiple research branches. We elucidate the problems that each branch aims to tackle and how the integration of EA and RL addresses these challenges. In conclusion, we discuss potential challenges and prospective future research directions
    
[^2]: 一种针对压缩视频的时间句子对齐的有效和高效管道

    You Can Ground Earlier than See: An Effective and Efficient Pipeline for Temporal Sentence Grounding in Compressed Videos. (arXiv:2303.07863v1 [cs.CV])

    [http://arxiv.org/abs/2303.07863](http://arxiv.org/abs/2303.07863)

    本文提出了一种新的压缩域TSG设置，通过直接编码压缩位流来增强视觉特征表示能力和提高时间句子对齐的效率，并且在两个基准数据集的实验中表现优于目前最先进的方法。

    

    时间句子对齐旨在根据句子查询通过语义定位目标瞬间。在本文中，我们提出了一种新的压缩域TSG（Temporal Sentence Grounding）设置，直接使用压缩视频作为视觉输入。针对原始视频比特流输入，我们提出了一种新型三支路压缩空间时间融合框架（TCSF），用于有效且高效地定位。我们通过利用压缩伪影来增强视觉特征的表示能力，提出了一种直接编码压缩位流的方法，而不是先解码整个帧的方法。在两个基准数据集上的实验结果表明，我们的方法在效果和效率方面优于目前最先进的方法。

    Given an untrimmed video, temporal sentence grounding (TSG) aims to locate a target moment semantically according to a sentence query. Although previous respectable works have made decent success, they only focus on high-level visual features extracted from the consecutive decoded frames and fail to handle the compressed videos for query modelling, suffering from insufficient representation capability and significant computational complexity during training and testing. In this paper, we pose a new setting, compressed-domain TSG, which directly utilizes compressed videos rather than fully-decompressed frames as the visual input. To handle the raw video bit-stream input, we propose a novel Three-branch Compressed-domain Spatial-temporal Fusion (TCSF) framework, which extracts and aggregates three kinds of low-level visual features (I-frame, motion vector and residual features) for effective and efficient grounding. Particularly, instead of encoding the whole decoded frames like previous
    


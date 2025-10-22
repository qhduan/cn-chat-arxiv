# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning by Watching: A Review of Video-based Learning Approaches for Robot Manipulation](https://arxiv.org/abs/2402.07127) |  |
| [^2] | [Graph Neural Networks for Road Safety Modeling: Datasets and Evaluations for Accident Analysis.](http://arxiv.org/abs/2311.00164) | 该论文构建了一个大规模的道路交通事故记录数据集，并使用该数据集评估了现有的深度学习方法在预测事故发生方面的准确性。研究发现，图神经网络GraphSAGE能够准确预测道路上的事故数量，并判断事故是否会发生。 |

# 详细

[^1]: 观察学习：基于视频的机器人操作学习方法综述

    Learning by Watching: A Review of Video-based Learning Approaches for Robot Manipulation

    [https://arxiv.org/abs/2402.07127](https://arxiv.org/abs/2402.07127)

    

    

    机器人学习操作技能受到多样化、无偏的数据集的稀缺性的影响。尽管策划的数据集可以帮助解决问题，但在泛化性和现实世界的转移方面仍然存在挑战。与此同时，“野外”视频数据集的大规模存在通过自监督技术推动了计算机视觉的进展。将这一点应用到机器人领域，最近的研究探索了通过被动观察来学习丰富的在线视频中的操作技能。这种基于视频的学习范式显示出了有希望的结果，它提供了可扩展的监督方法，同时降低了数据集的偏见。本综述回顾了视频特征表示学习技术、物体可行性理解、三维手部/身体建模和大规模机器人资源等基础知识，以及从不受控制的视频演示中获取机器人操作技能的新兴技术。我们讨论了仅从观察大规模人类视频中学习如何增强机器人的泛化性和样本效率。

    Robot learning of manipulation skills is hindered by the scarcity of diverse, unbiased datasets. While curated datasets can help, challenges remain in generalizability and real-world transfer. Meanwhile, large-scale "in-the-wild" video datasets have driven progress in computer vision through self-supervised techniques. Translating this to robotics, recent works have explored learning manipulation skills by passively watching abundant videos sourced online. Showing promising results, such video-based learning paradigms provide scalable supervision while reducing dataset bias. This survey reviews foundations such as video feature representation learning techniques, object affordance understanding, 3D hand/body modeling, and large-scale robot resources, as well as emerging techniques for acquiring robot manipulation skills from uncontrolled video demonstrations. We discuss how learning only from observing large-scale human videos can enhance generalization and sample efficiency for roboti
    
[^2]: 道路安全建模的图神经网络：用于事故分析的数据集和评估

    Graph Neural Networks for Road Safety Modeling: Datasets and Evaluations for Accident Analysis. (arXiv:2311.00164v1 [cs.SI])

    [http://arxiv.org/abs/2311.00164](http://arxiv.org/abs/2311.00164)

    该论文构建了一个大规模的道路交通事故记录数据集，并使用该数据集评估了现有的深度学习方法在预测事故发生方面的准确性。研究发现，图神经网络GraphSAGE能够准确预测道路上的事故数量，并判断事故是否会发生。

    

    我们考虑基于道路网络连接和交通流量的道路网络上的交通事故分析问题。以往的工作使用历史记录设计了各种深度学习方法来预测交通事故的发生。然而，现有方法的准确性缺乏共识，并且一个基本问题是缺乏公共事故数据集进行全面评估。本文构建了一个大规模的、统一的道路交通事故记录数据集，包括来自美国各州官方报告的900万条记录，以及道路网络和交通流量报告。利用这个新数据集，我们评估了现有的深度学习方法来预测道路网络上的事故发生。我们的主要发现是，像GraphSAGE这样的图神经网络可以准确预测道路上的事故数量，平均绝对误差不超过实际数目的22%，并能够判断事故是否会发生。

    We consider the problem of traffic accident analysis on a road network based on road network connections and traffic volume. Previous works have designed various deep-learning methods using historical records to predict traffic accident occurrences. However, there is a lack of consensus on how accurate existing methods are, and a fundamental issue is the lack of public accident datasets for comprehensive evaluations. This paper constructs a large-scale, unified dataset of traffic accident records from official reports of various states in the US, totaling 9 million records, accompanied by road networks and traffic volume reports. Using this new dataset, we evaluate existing deep-learning methods for predicting the occurrence of accidents on road networks. Our main finding is that graph neural networks such as GraphSAGE can accurately predict the number of accidents on roads with less than 22% mean absolute error (relative to the actual count) and whether an accident will occur or not w
    


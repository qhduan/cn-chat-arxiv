# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Train Once, Use Flexibly: A Modular Framework for Multi-Aspect Neural News Recommendation.](http://arxiv.org/abs/2307.16089) | 这项研究提出了一种模块化框架MANNeR，用于灵活的多方面（神经）新闻推荐，支持在推理时对各个方面进行临时定制。通过度量学习和灵活的相似度得分组合，MANNeR实现了更好的多方面推荐效果。 |
| [^2] | [Visualising Personal Data Flows: Insights from a Case Study of Booking.com.](http://arxiv.org/abs/2304.09603) | 本文以Booking.com为基础，以可视化个人数据流为研究，展示公司如何分享消费者个人数据，并讨论使用隐私政策告知客户个人数据流的挑战和限制。本案例研究为未来更以数据流为导向的隐私政策分析和建立更全面的个人数据流本体论的研究提供了参考。 |

# 详细

[^1]: 一次训练，灵活应用：多方面神经新闻推荐的模块化框架

    Train Once, Use Flexibly: A Modular Framework for Multi-Aspect Neural News Recommendation. (arXiv:2307.16089v1 [cs.IR])

    [http://arxiv.org/abs/2307.16089](http://arxiv.org/abs/2307.16089)

    这项研究提出了一种模块化框架MANNeR，用于灵活的多方面（神经）新闻推荐，支持在推理时对各个方面进行临时定制。通过度量学习和灵活的相似度得分组合，MANNeR实现了更好的多方面推荐效果。

    

    最近，神经网络新闻推荐器（NNR）通过（1）将候选新闻与用户历史之间的主题或情感等方面进行对齐，或者（2）在这些方面上推广推荐来扩展基于内容的推荐。这种定制是通过将额外的约束“硬编码”到NNR的架构和/或训练目标中来实现的：因此，任何对期望的推荐行为的更改都需要修改目标重新训练模型，从而阻碍了多方面新闻推荐的广泛应用。在这项工作中，我们引入了MANNeR，这是一个灵活的多方面（神经）新闻推荐的模块化框架，支持在推理时对各个方面进行临时定制。以度量学习为核心，MANNeR获得了专门针对各个方面的新闻编码器，然后灵活地将各个方面的相似度得分组合起来进行最终排序。在两个标准的新闻推荐基准测试上（一个是英文，一个是挪威文），MANNeR的评估结果表明...

    Recent neural news recommenders (NNR) extend content-based recommendation by (1) aligning additional aspects such as topic or sentiment between the candidate news and user history or (2) diversifying recommendations w.r.t. these aspects. This customization is achieved by ``hardcoding'' additional constraints into NNR's architecture and/or training objectives: any change in the desired recommendation behavior thus requires the model to be retrained with a modified objective, impeding wide adoption of multi-aspect news recommenders. In this work, we introduce MANNeR, a modular framework for flexible multi-aspect (neural) news recommendation that supports ad-hoc customization over individual aspects at inference time. With metric-based learning at its core, MANNeR obtains aspect-specialized news encoders and then flexibly combines aspect-specific similarity scores for final ranking. Evaluation on two standard news recommendation benchmarks (one in English, one in Norwegian) shows that MAN
    
[^2]: 可视化个人数据流：以Booking.com为例的案例研究

    Visualising Personal Data Flows: Insights from a Case Study of Booking.com. (arXiv:2304.09603v1 [cs.CR])

    [http://arxiv.org/abs/2304.09603](http://arxiv.org/abs/2304.09603)

    本文以Booking.com为基础，以可视化个人数据流为研究，展示公司如何分享消费者个人数据，并讨论使用隐私政策告知客户个人数据流的挑战和限制。本案例研究为未来更以数据流为导向的隐私政策分析和建立更全面的个人数据流本体论的研究提供了参考。

    

    商业机构持有和处理的个人数据量越来越多。政策和法律不断变化，要求这些公司在收集、存储、处理和共享这些数据方面更加透明。本文报告了我们以Booking.com为案例研究，从他们的隐私政策中提取个人数据流的可视化工作。通过展示该公司如何分享其消费者的个人数据，我们提出了问题，并扩展了有关使用隐私政策告知客户个人数据流范围的挑战和限制的讨论。更重要的是，本案例研究可以为未来更以数据流为导向的隐私政策分析和在复杂商业生态系统中建立更全面的个人数据流本体论的研究提供参考。

    Commercial organisations are holding and processing an ever-increasing amount of personal data. Policies and laws are continually changing to require these companies to be more transparent regarding collection, storage, processing and sharing of this data. This paper reports our work of taking Booking.com as a case study to visualise personal data flows extracted from their privacy policy. By showcasing how the company shares its consumers' personal data, we raise questions and extend discussions on the challenges and limitations of using privacy policy to inform customers the true scale and landscape of personal data flows. More importantly, this case study can inform us about future research on more data flow-oriented privacy policy analysis and on the construction of a more comprehensive ontology on personal data flows in complicated business ecosystems.
    


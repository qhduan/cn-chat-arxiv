# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Empowering Sequential Recommendation from Collaborative Signals and Semantic Relatedness](https://arxiv.org/abs/2403.07623) | 该论文提出了一种端到端两流架构TSSR，旨在通过有效地将协作信号和语义相关性结合起来，增强顺序推荐任务。 |
| [^2] | [Avoiding Catastrophic Forgetting in Visual Classification Using Human Concept Formation](https://arxiv.org/abs/2402.16933) | 提出了一种名为Cobweb4V的新颖视觉分类方法，利用人类类似学习系统，避免了灾难性遗忘效应，与传统方法相比，需要更少的数据来实现有效学习成果，并保持稳定性能。 |
| [^3] | [Visualising Personal Data Flows: Insights from a Case Study of Booking.com.](http://arxiv.org/abs/2304.09603) | 本文以Booking.com为基础，以可视化个人数据流为研究，展示公司如何分享消费者个人数据，并讨论使用隐私政策告知客户个人数据流的挑战和限制。本案例研究为未来更以数据流为导向的隐私政策分析和建立更全面的个人数据流本体论的研究提供了参考。 |

# 详细

[^1]: 从协作信号和语义相关性中增强顺序推荐

    Empowering Sequential Recommendation from Collaborative Signals and Semantic Relatedness

    [https://arxiv.org/abs/2403.07623](https://arxiv.org/abs/2403.07623)

    该论文提出了一种端到端两流架构TSSR，旨在通过有效地将协作信号和语义相关性结合起来，增强顺序推荐任务。

    

    顺序推荐系统(SRS)通过对按时间顺序排列的历史行为进行建模，能够捕捉到动态用户偏好。尽管有效，仅关注行为中的\textit{协作信号}并不能充分把握用户兴趣。在内容特征中反映的\textit{语义相关性}建模也是重要的，例如图片和文本。因此，本文旨在通过有效地将协作信号和语义相关性结合起来，增强SRS任务。值得注意的是，我们实证指出由于语义鸿沟问题，实现这一目标并不容易。因此，我们提出了一种用于顺序推荐的端到端两流架构，命名为TSSR，从基于ID和基于内容的序列中学习用户偏好。具体来说，我们首先提出了新颖的层次对比模块，包括粗用户粒度和细项粒度术语，以对齐表征。

    arXiv:2403.07623v1 Announce Type: new  Abstract: Sequential recommender systems (SRS) could capture dynamic user preferences by modeling historical behaviors ordered in time. Despite effectiveness, focusing only on the \textit{collaborative signals} from behaviors does not fully grasp user interests. It is also significant to model the \textit{semantic relatedness} reflected in content features, e.g., images and text. Towards that end, in this paper, we aim to enhance the SRS tasks by effectively unifying collaborative signals and semantic relatedness together. Notably, we empirically point out that it is nontrivial to achieve such a goal due to semantic gap issues. Thus, we propose an end-to-end two-stream architecture for sequential recommendation, named TSSR, to learn user preferences from ID-based and content-based sequence. Specifically, we first present novel hierarchical contrasting module, including coarse user-grained and fine item-grained terms, to align the representations o
    
[^2]: 使用人类概念形成避免视觉分类中的灾难性遗忘

    Avoiding Catastrophic Forgetting in Visual Classification Using Human Concept Formation

    [https://arxiv.org/abs/2402.16933](https://arxiv.org/abs/2402.16933)

    提出了一种名为Cobweb4V的新颖视觉分类方法，利用人类类似学习系统，避免了灾难性遗忘效应，与传统方法相比，需要更少的数据来实现有效学习成果，并保持稳定性能。

    

    深度神经网络在机器学习中表现出色，特别是在视觉任务中，然而，当按顺序学习新任务时，它们经常面临灾难性遗忘。本研究提出了Cobweb4V，这是一种新颖的视觉分类方法，它基于Cobweb，这是一种人类类似的学习系统，受到人类随时间逐渐学习新概念的启发。我们进行了全面评估，展示了Cobweb4V在学习视觉概念方面的熟练程度，相较于传统方法，需要更少的数据来实现有效的学习成果，随时间保持稳定的性能，并实现了令人称赞的渐近行为，避免了灾难性遗忘效应。这些特征与人类认知中的学习策略一致，将Cobweb4V定位为神经网络方法的一个有前途的替代方案。

    arXiv:2402.16933v1 Announce Type: cross  Abstract: Deep neural networks have excelled in machine learning, particularly in vision tasks, however, they often suffer from catastrophic forgetting when learning new tasks sequentially. In this work, we propose Cobweb4V, a novel visual classification approach that builds on Cobweb, a human like learning system that is inspired by the way humans incrementally learn new concepts over time. In this research, we conduct a comprehensive evaluation, showcasing the proficiency of Cobweb4V in learning visual concepts, requiring less data to achieve effective learning outcomes compared to traditional methods, maintaining stable performance over time, and achieving commendable asymptotic behavior, without catastrophic forgetting effects. These characteristics align with learning strategies in human cognition, positioning Cobweb4V as a promising alternative to neural network approaches.
    
[^3]: 可视化个人数据流：以Booking.com为例的案例研究

    Visualising Personal Data Flows: Insights from a Case Study of Booking.com. (arXiv:2304.09603v1 [cs.CR])

    [http://arxiv.org/abs/2304.09603](http://arxiv.org/abs/2304.09603)

    本文以Booking.com为基础，以可视化个人数据流为研究，展示公司如何分享消费者个人数据，并讨论使用隐私政策告知客户个人数据流的挑战和限制。本案例研究为未来更以数据流为导向的隐私政策分析和建立更全面的个人数据流本体论的研究提供了参考。

    

    商业机构持有和处理的个人数据量越来越多。政策和法律不断变化，要求这些公司在收集、存储、处理和共享这些数据方面更加透明。本文报告了我们以Booking.com为案例研究，从他们的隐私政策中提取个人数据流的可视化工作。通过展示该公司如何分享其消费者的个人数据，我们提出了问题，并扩展了有关使用隐私政策告知客户个人数据流范围的挑战和限制的讨论。更重要的是，本案例研究可以为未来更以数据流为导向的隐私政策分析和在复杂商业生态系统中建立更全面的个人数据流本体论的研究提供参考。

    Commercial organisations are holding and processing an ever-increasing amount of personal data. Policies and laws are continually changing to require these companies to be more transparent regarding collection, storage, processing and sharing of this data. This paper reports our work of taking Booking.com as a case study to visualise personal data flows extracted from their privacy policy. By showcasing how the company shares its consumers' personal data, we raise questions and extend discussions on the challenges and limitations of using privacy policy to inform customers the true scale and landscape of personal data flows. More importantly, this case study can inform us about future research on more data flow-oriented privacy policy analysis and on the construction of a more comprehensive ontology on personal data flows in complicated business ecosystems.
    


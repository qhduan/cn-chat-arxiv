# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Pix2Pix-OnTheFly: Leveraging LLMs for Instruction-Guided Image Editing](https://arxiv.org/abs/2403.08004) | 本文提出了一种新方法，实现了基于自然语言指令的图像编辑，在不需要任何预备工作的情况下，通过图像字幕和DDIM反演，获取编辑方向嵌入，进行指导图像编辑，表现出有效性和竞争力。 |
| [^2] | [Speech emotion recognition from voice messages recorded in the wild](https://arxiv.org/abs/2403.02167) | 使用Emotional Voice Messages数据库，结合eGeMAPS特征和Transformer模型，实现了在野外录制的语音消息中的语音情感识别，取得了较高的准确度，并比基准模型提高了10%。 |
| [^3] | [Facility Location Games with Scaling Effects](https://arxiv.org/abs/2402.18908) | 研究了具有规模效应的设施选址游戏，提供了对于连续比例函数和分段线性比例函数的结果，适用于许多实际情景，同时探讨了近似机制设计设置下代理可能不再单峰偏好的条件与成本近似比率。 |
| [^4] | [Prediction-Powered Ranking of Large Language Models](https://arxiv.org/abs/2402.17826) | 该研究提出了一种统计框架，可以衡量人类与模型偏好之间的不确定性，从而进行大型语言模型的预测排名。 |
| [^5] | [Scaling laws for learning with real and surrogate data](https://arxiv.org/abs/2402.04376) | 本研究探讨了将替代数据与真实数据整合以进行训练的方案，发现整合替代数据能够显著降低测试误差，并提出了一个扩展规律来描述混合模型的测试误差，可以用于预测最优加权和收益。 |
| [^6] | [Exploring Homogeneous and Heterogeneous Consistent Label Associations for Unsupervised Visible-Infrared Person ReID](https://arxiv.org/abs/2402.00672) | 该论文提出了一种同时考虑均质和异质实例级别结构，构建高质量跨模态标签关联的模态统一标签传输方法，用于无监督可见-红外人物重新识别。 |
| [^7] | [Push it to the Demonstrated Limit: Multimodal Visuotactile Imitation Learning with Force Matching.](http://arxiv.org/abs/2311.01248) | 本研究利用视觉触觉传感器和模仿学习相结合，通过配对优化触觉力量曲线和简化传感器应用，对接触丰富的操作任务进行了研究。 |
| [^8] | [FocDepthFormer: Transformer with LSTM for Depth Estimation from Focus.](http://arxiv.org/abs/2310.11178) | FocDepthFormer是一种基于Transformer和LSTM的网络，用于从焦点进行深度估计。通过Transformer的自注意力和LSTM的集成，该方法能够学习更多有信息的特征，并且具有对任意长度堆栈的泛化能力。 |
| [^9] | [Inductive Meta-path Learning for Schema-complex Heterogeneous Information Networks.](http://arxiv.org/abs/2307.03937) | 这项研究提出了一种针对模式复杂的异构信息网络的归纳元路径学习框架SchemaWalk。 |
| [^10] | [Data quality dimensions for fair AI.](http://arxiv.org/abs/2305.06967) | 本文着眼于解决AI系统中的偏见问题，从信息质量维度的角度出发提出了解决偏见的潜在改进，提出了完整性、一致性、及时性和可靠性等数据质量维度。 |
| [^11] | [OpenDriver: an open-road driver state detection dataset.](http://arxiv.org/abs/2304.04203) | OpenDriver是一份旨在解决现有驾驶员生理数据集存在问题的开放路况驾驶员状态检测数据集，包含六轴惯性信号和心电图信号两种模态的数据，可用于驾驶员受损检测和生物识别数据识别。 |

# 详细

[^1]: Pix2Pix-OnTheFly: 利用LLMs进行指导图像编辑

    Pix2Pix-OnTheFly: Leveraging LLMs for Instruction-Guided Image Editing

    [https://arxiv.org/abs/2403.08004](https://arxiv.org/abs/2403.08004)

    本文提出了一种新方法，实现了基于自然语言指令的图像编辑，在不需要任何预备工作的情况下，通过图像字幕和DDIM反演，获取编辑方向嵌入，进行指导图像编辑，表现出有效性和竞争力。

    

    众所周知，最近结合语言处理和图像处理的研究引起了广泛关注，本文提出了一种全新的方法，通过图像字幕和DDIM反演，获取编辑方向嵌入，进行指导图像编辑，而无需预备工作，证明了该方法的有效性和竞争力。

    arXiv:2403.08004v1 Announce Type: cross  Abstract: The combination of language processing and image processing keeps attracting increased interest given recent impressive advances that leverage the combined strengths of both domains of research. Among these advances, the task of editing an image on the basis solely of a natural language instruction stands out as a most challenging endeavour. While recent approaches for this task resort, in one way or other, to some form of preliminary preparation, training or fine-tuning, this paper explores a novel approach: We propose a preparation-free method that permits instruction-guided image editing on the fly. This approach is organized along three steps properly orchestrated that resort to image captioning and DDIM inversion, followed by obtaining the edit direction embedding, followed by image editing proper. While dispensing with preliminary preparation, our approach demonstrates to be effective and competitive, outperforming recent, state 
    
[^2]: 从野外录制的语音消息中识别语音情感

    Speech emotion recognition from voice messages recorded in the wild

    [https://arxiv.org/abs/2403.02167](https://arxiv.org/abs/2403.02167)

    使用Emotional Voice Messages数据库，结合eGeMAPS特征和Transformer模型，实现了在野外录制的语音消息中的语音情感识别，取得了较高的准确度，并比基准模型提高了10%。

    

    用于语音情感识别（SER）的情感数据集通常包含表演或引发的语音，限制了它们在现实场景中的适用性。在这项工作中，我们使用了Emotional Voice Messages（EMOVOME）数据库，其中包括来自100名西班牙语使用者在消息应用中的自发语音消息，由专家和非专家标注者以连续和离散的情感进行标记。我们使用了eGeMAPS特征、基于Transformer的模型以及它们的组合来创建讲话者无关的SER模型。我们将结果与参考数据库进行了比较，并分析了标注者和性别公平性的影响。预训练的Unispeech-L模型及其与eGeMAPS的组合取得了最佳结果，在3类valence和arousal预测中分别获得了61.64%和55.57%的Unweighted Accuracy（UA），比基线模型提高了10%。对于情感类别，获得了42.58%的UA。EMOVOME表现不佳。

    arXiv:2403.02167v1 Announce Type: cross  Abstract: Emotion datasets used for Speech Emotion Recognition (SER) often contain acted or elicited speech, limiting their applicability in real-world scenarios. In this work, we used the Emotional Voice Messages (EMOVOME) database, including spontaneous voice messages from conversations of 100 Spanish speakers on a messaging app, labeled in continuous and discrete emotions by expert and non-expert annotators. We created speaker-independent SER models using the eGeMAPS features, transformer-based models and their combination. We compared the results with reference databases and analyzed the influence of annotators and gender fairness. The pre-trained Unispeech-L model and its combination with eGeMAPS achieved the highest results, with 61.64% and 55.57% Unweighted Accuracy (UA) for 3-class valence and arousal prediction respectively, a 10% improvement over baseline models. For the emotion categories, 42.58% UA was obtained. EMOVOME performed low
    
[^3]: 具有规模效应的设施选址游戏

    Facility Location Games with Scaling Effects

    [https://arxiv.org/abs/2402.18908](https://arxiv.org/abs/2402.18908)

    研究了具有规模效应的设施选址游戏，提供了对于连续比例函数和分段线性比例函数的结果，适用于许多实际情景，同时探讨了近似机制设计设置下代理可能不再单峰偏好的条件与成本近似比率。

    

    我们考虑了经典的设施选址问题的一个变种，其中每个代理的个人成本函数等于他们距离设施的距离乘以一个由设施位置确定的比例因子。除了一般类别的连续比例函数外，我们还提供了适用于许多实际情景的比例函数的分段线性比例函数的结果。我们关注总成本和最大成本的目标，并描述了最优解的计算。然后我们转向近似机制设计设置，观察到代理的偏好可能不再是单峰的。因此，我们表征了确保代理具有单峰偏好的比例函数条件。在这些条件下，我们找到了能够通过strategyproof和anonymous me达到的总成本和最大成本近似比率的结果。

    arXiv:2402.18908v1 Announce Type: cross  Abstract: We take the classic facility location problem and consider a variation, in which each agent's individual cost function is equal to their distance from the facility multiplied by a scaling factor which is determined by the facility placement. In addition to the general class of continuous scaling functions, we also provide results for piecewise linear scaling functions which can effectively approximate or model the scaling of many real world scenarios. We focus on the objectives of total and maximum cost, describing the computation of the optimal solution. We then move to the approximate mechanism design setting, observing that the agents' preferences may no longer be single-peaked. Consequently, we characterize the conditions on scaling functions which ensure that agents have single-peaked preferences. Under these conditions, we find results on the total and maximum cost approximation ratios achievable by strategyproof and anonymous me
    
[^4]: 大型语言模型的预测排名

    Prediction-Powered Ranking of Large Language Models

    [https://arxiv.org/abs/2402.17826](https://arxiv.org/abs/2402.17826)

    该研究提出了一种统计框架，可以衡量人类与模型偏好之间的不确定性，从而进行大型语言模型的预测排名。

    

    大型语言模型通常根据其与人类偏好的一致性水平进行排名--如果一个模型的输出更受人类偏好，那么它就比其他模型更好。本文提出了一种统计框架来弥合人类与模型偏好之间可能引入的不一致性。

    arXiv:2402.17826v1 Announce Type: cross  Abstract: Large language models are often ranked according to their level of alignment with human preferences -- a model is better than other models if its outputs are more frequently preferred by humans. One of the most popular ways to elicit human preferences utilizes pairwise comparisons between the outputs provided by different models to the same inputs. However, since gathering pairwise comparisons by humans is costly and time-consuming, it has become a very common practice to gather pairwise comparisons by a strong large language model -- a model strongly aligned with human preferences. Surprisingly, practitioners cannot currently measure the uncertainty that any mismatch between human and model preferences may introduce in the constructed rankings. In this work, we develop a statistical framework to bridge this gap. Given a small set of pairwise comparisons by humans and a large set of pairwise comparisons by a model, our framework provid
    
[^5]: 使用真实数据和替代数据进行学习的扩展规律

    Scaling laws for learning with real and surrogate data

    [https://arxiv.org/abs/2402.04376](https://arxiv.org/abs/2402.04376)

    本研究探讨了将替代数据与真实数据整合以进行训练的方案，发现整合替代数据能够显著降低测试误差，并提出了一个扩展规律来描述混合模型的测试误差，可以用于预测最优加权和收益。

    

    收集大量高质量的数据通常被限制在成本昂贵或不切实际的范围内, 这是机器学习中的一个关键瓶颈。相反地, 可以将来自目标分布的小规模数据集与来自公共数据集、不同情况下收集的数据或由生成模型合成的数据相结合, 作为替代数据。我们提出了一种简单的方案来将替代数据整合到训练中, 并使用理论模型和实证研究探索其行为。我们的主要发现是：(i) 整合替代数据可以显著降低原始分布的测试误差；(ii) 为了获得这种效益, 使用最优加权经验风险最小化非常关键；(iii) 在混合使用真实数据和替代数据训练的模型的测试误差可以很好地用一个扩展规律来描述。这可以用来预测最优加权和收益。

    Collecting large quantities of high-quality data is often prohibitively expensive or impractical, and a crucial bottleneck in machine learning. One may instead augment a small set of $n$ data points from the target distribution with data from more accessible sources like public datasets, data collected under different circumstances, or synthesized by generative models. Blurring distinctions, we refer to such data as `surrogate data'.   We define a simple scheme for integrating surrogate data into training and use both theoretical models and empirical studies to explore its behavior. Our main findings are: $(i)$ Integrating surrogate data can significantly reduce the test error on the original distribution; $(ii)$ In order to reap this benefit, it is crucial to use optimally weighted empirical risk minimization; $(iii)$ The test error of models trained on mixtures of real and surrogate data is well described by a scaling law. This can be used to predict the optimal weighting and the gai
    
[^6]: 探索用于无监督可见-红外人物重新识别的均质和异质一致标签关联

    Exploring Homogeneous and Heterogeneous Consistent Label Associations for Unsupervised Visible-Infrared Person ReID

    [https://arxiv.org/abs/2402.00672](https://arxiv.org/abs/2402.00672)

    该论文提出了一种同时考虑均质和异质实例级别结构，构建高质量跨模态标签关联的模态统一标签传输方法，用于无监督可见-红外人物重新识别。

    

    无监督可见-红外人物重新识别（USL-VI-ReID）旨在无需注释从不同模态中检索相同身份的行人图像。之前的研究侧重于建立跨模态的伪标签关联以弥合模态间的差异，但忽略了在伪标签空间中保持实例级别的均质和异质一致性，导致关联粗糙。为此，我们引入了一个模态统一标签传输（MULT）模块，同时考虑了均质和异质细粒度实例级结构，生成高质量的跨模态标签关联。它建模了均质和异质的关联性，利用它们定义伪标签的不一致性，然后最小化这种不一致性，从而维持了跨模态的对齐并保持了内部模态结构的一致性。此外，还有一个简单易用的在线交叉记忆标签引用模块。

    Unsupervised visible-infrared person re-identification (USL-VI-ReID) aims to retrieve pedestrian images of the same identity from different modalities without annotations. While prior work focuses on establishing cross-modality pseudo-label associations to bridge the modality-gap, they ignore maintaining the instance-level homogeneous and heterogeneous consistency in pseudo-label space, resulting in coarse associations. In response, we introduce a Modality-Unified Label Transfer (MULT) module that simultaneously accounts for both homogeneous and heterogeneous fine-grained instance-level structures, yielding high-quality cross-modality label associations. It models both homogeneous and heterogeneous affinities, leveraging them to define the inconsistency for the pseudo-labels and then minimize it, leading to pseudo-labels that maintain alignment across modalities and consistency within intra-modality structures. Additionally, a straightforward plug-and-play Online Cross-memory Label Ref
    
[^7]: 将其推向展示极限：多模态视觉触觉模仿学习与力匹配

    Push it to the Demonstrated Limit: Multimodal Visuotactile Imitation Learning with Force Matching. (arXiv:2311.01248v1 [cs.RO])

    [http://arxiv.org/abs/2311.01248](http://arxiv.org/abs/2311.01248)

    本研究利用视觉触觉传感器和模仿学习相结合，通过配对优化触觉力量曲线和简化传感器应用，对接触丰富的操作任务进行了研究。

    

    光学触觉传感器已经成为机器人操作过程中获取密集接触信息的有效手段。最近引入的“透视你的皮肤”（STS）型传感器具有视觉和触觉模式，通过利用半透明表面和可控照明实现。本文研究了视觉触觉传感与模仿学习在富有接触的操作任务中的好处。首先，我们使用触觉力测量和一种新的算法，在运动示范中产生更好匹配人体示范者的力曲线。其次，我们添加了视觉/触觉STS模式切换作为控制策略输出，简化传感器的应用。最后，我们研究了多种观察配置，比较和对比了视觉/触觉数据（包括模式切换和不切换）与手腕挂载的眼在手摄像机的视觉数据的价值。我们在一个广泛的实验系列上进行实验。

    Optical tactile sensors have emerged as an effective means to acquire dense contact information during robotic manipulation. A recently-introduced `see-through-your-skin' (STS) variant of this type of sensor has both visual and tactile modes, enabled by leveraging a semi-transparent surface and controllable lighting. In this work, we investigate the benefits of pairing visuotactile sensing with imitation learning for contact-rich manipulation tasks. First, we use tactile force measurements and a novel algorithm during kinesthetic teaching to yield a force profile that better matches that of the human demonstrator. Second, we add visual/tactile STS mode switching as a control policy output, simplifying the application of the sensor. Finally, we study multiple observation configurations to compare and contrast the value of visual/tactile data (both with and without mode switching) with visual data from a wrist-mounted eye-in-hand camera. We perform an extensive series of experiments on a
    
[^8]: FocDepthFormer: 使用LSTM的Transformer用于从焦点进行深度估计

    FocDepthFormer: Transformer with LSTM for Depth Estimation from Focus. (arXiv:2310.11178v1 [cs.CV])

    [http://arxiv.org/abs/2310.11178](http://arxiv.org/abs/2310.11178)

    FocDepthFormer是一种基于Transformer和LSTM的网络，用于从焦点进行深度估计。通过Transformer的自注意力和LSTM的集成，该方法能够学习更多有信息的特征，并且具有对任意长度堆栈的泛化能力。

    

    从焦点堆栈进行深度估计是一个基本的计算机视觉问题，旨在通过图像堆栈中的焦点/离焦线索推断深度。大多数现有方法通过在一组固定的图像堆栈上应用二维或三维卷积神经网络（CNNs）来处理此问题，以在图像和堆栈之间学习特征。由于CNN的局部性质，它们的性能受到限制，并且它们被限制在处理在训练和推断中一致的固定数量的堆栈上，从而限制了对任意长度堆栈的泛化能力。为了解决上述限制，我们开发了一种新颖的基于Transformer的网络，FocDepthFormer，主要由带有LSTM模块和CNN解码器的Transformer组成。Transformer中的自注意力通过隐含非局部交叉参考能够学习更多有信息的特征。LSTM模块被学习用于将表示集成到具有任意图像的堆栈中。为了直接捕获低级特征

    Depth estimation from focal stacks is a fundamental computer vision problem that aims to infer depth from focus/defocus cues in the image stacks. Most existing methods tackle this problem by applying convolutional neural networks (CNNs) with 2D or 3D convolutions over a set of fixed stack images to learn features across images and stacks. Their performance is restricted due to the local properties of the CNNs, and they are constrained to process a fixed number of stacks consistent in train and inference, limiting the generalization to the arbitrary length of stacks. To handle the above limitations, we develop a novel Transformer-based network, FocDepthFormer, composed mainly of a Transformer with an LSTM module and a CNN decoder. The self-attention in Transformer enables learning more informative features via an implicit non-local cross reference. The LSTM module is learned to integrate the representations across the stack with arbitrary images. To directly capture the low-level featur
    
[^9]: 针对模式复杂的异构信息网络的归纳元路径学习

    Inductive Meta-path Learning for Schema-complex Heterogeneous Information Networks. (arXiv:2307.03937v1 [cs.AI])

    [http://arxiv.org/abs/2307.03937](http://arxiv.org/abs/2307.03937)

    这项研究提出了一种针对模式复杂的异构信息网络的归纳元路径学习框架SchemaWalk。

    

    异构信息网络(HINs)是具有多种节点类型和边类型的信息网络。元路径的概念即一系列连接两个实体的实体类型和关系类型的序列被提出为提供对不同HIN任务的元级可解释语义的一种方法。传统上，元路径主要用于模式简单的HINs，例如只有少量实体类型的文献网络，在这种情况下，元路径通常通过领域知识枚举。然而，元路径在模式复杂的HINs(例如具有数百种实体和关系类型的知识库)中的应用受到了由元路径枚举引起的计算复杂性的限制。此外，有效评估元路径需要枚举相关路径实例，这进一步增加了元路径学习过程的复杂性。为了应对这些挑战，我们提出了一种用于模式复杂的HINs的归纳元路径学习框架SchemaWalk。

    Heterogeneous Information Networks (HINs) are information networks with multiple types of nodes and edges. The concept of meta-path, i.e., a sequence of entity types and relation types connecting two entities, is proposed to provide the meta-level explainable semantics for various HIN tasks. Traditionally, meta-paths are primarily used for schema-simple HINs, e.g., bibliographic networks with only a few entity types, where meta-paths are often enumerated with domain knowledge. However, the adoption of meta-paths for schema-complex HINs, such as knowledge bases (KBs) with hundreds of entity and relation types, has been limited due to the computational complexity associated with meta-path enumeration. Additionally, effectively assessing meta-paths requires enumerating relevant path instances, which adds further complexity to the meta-path learning process. To address these challenges, we propose SchemaWalk, an inductive meta-path learning framework for schema-complex HINs. We represent m
    
[^10]: 面向公正AI的数据质量维度

    Data quality dimensions for fair AI. (arXiv:2305.06967v1 [cs.AI])

    [http://arxiv.org/abs/2305.06967](http://arxiv.org/abs/2305.06967)

    本文着眼于解决AI系统中的偏见问题，从信息质量维度的角度出发提出了解决偏见的潜在改进，提出了完整性、一致性、及时性和可靠性等数据质量维度。

    

    人工智能系统并非本质上具有中立性，因此偏见会渗透到任何类型的技术工具中。特别是在处理人类时，AI算法会反映出源于错标记数据的技术错误。由于它们提供了错误和歧视性的分类，延续了结构性种族主义和边缘化现象，这些系统并未系统地防范偏见。本文从信息质量维度的角度考虑了AI系统偏见问题，以两个通常较为困难的情境，即非二元个体的分类和跨性别个体的分类为例，说明了偏见缓解工具的潜在改进。确定要实施的数据质量维度以实现更公平的目的可能有助于解决这个问题，因此我们提出建议在完整性、一致性、及时性和可靠性等方面考虑这个问题，并提供了一些理论结果。

    AI systems are not intrinsically neutral and biases trickle in any type of technological tool. In particular when dealing with people, AI algorithms reflect technical errors originating with mislabeled data. As they feed wrong and discriminatory classifications, perpetuating structural racism and marginalization, these systems are not systematically guarded against bias. In this article we consider the problem of bias in AI systems from the point of view of Information Quality dimensions. We illustrate potential improvements of a bias mitigation tool in gender classification errors, referring to two typically difficult contexts: the classification of non-binary individuals and the classification of transgender individuals. The identification of data quality dimensions to implement in bias mitigation tool may help achieve more fairness. Hence, we propose to consider this issue in terms of completeness, consistency, timeliness and reliability, and offer some theoretical results.
    
[^11]: OpenDriver: 一份开放路况驾驶员状态检测数据集

    OpenDriver: an open-road driver state detection dataset. (arXiv:2304.04203v1 [cs.AI])

    [http://arxiv.org/abs/2304.04203](http://arxiv.org/abs/2304.04203)

    OpenDriver是一份旨在解决现有驾驶员生理数据集存在问题的开放路况驾驶员状态检测数据集，包含六轴惯性信号和心电图信号两种模态的数据，可用于驾驶员受损检测和生物识别数据识别。

    

    在现代社会中，道路安全严重依赖于驾驶员的心理和生理状态。疲劳、昏昏欲睡和压力等负面因素会影响驾驶员的反应时间和决策能力，导致交通事故的发生率增加。在众多的驾驶员行为监测研究中，可穿戴生理测量是一种实时监测驾驶员状态的方法。然而，目前在开放道路场景下，缺少驾驶员生理数据集，已有的数据集存在信号质量差、样本量小和数据收集时间短等问题。因此，本文设计并描述了一种大规模多模态驾驶数据集，用于驾驶员受损检测和生物识别数据识别。该数据集包含两种驾驶信号模态：六轴惯性信号和心电图（ECG）信号，这些信号是在100多名驾驶员遵循相同路线行驶时记录的。

    In modern society, road safety relies heavily on the psychological and physiological state of drivers. Negative factors such as fatigue, drowsiness, and stress can impair drivers' reaction time and decision making abilities, leading to an increased incidence of traffic accidents. Among the numerous studies for impaired driving detection, wearable physiological measurement is a real-time approach to monitoring a driver's state. However, currently, there are few driver physiological datasets in open road scenarios and the existing datasets suffer from issues such as poor signal quality, small sample sizes, and short data collection periods. Therefore, in this paper, a large-scale multimodal driving dataset for driver impairment detection and biometric data recognition is designed and described. The dataset contains two modalities of driving signals: six-axis inertial signals and electrocardiogram (ECG) signals, which were recorded while over one hundred drivers were following the same ro
    


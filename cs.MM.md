# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Leveraging Pre-trained AudioLDM for Text to Sound Generation: A Benchmark Study.](http://arxiv.org/abs/2303.03857) | 本文研究了使用预训练的AudioLDM作为声音生成的骨干的优势，证明了在数据稀缺情况下使用预训练模型进行文本到声音生成的优势，并在几个常用数据集上使用相同的评估协议评估了各种文本到声音生成系统，为未来的研究提供了基础。 |
| [^2] | [Heterogeneous Graph Learning for Acoustic Event Classification.](http://arxiv.org/abs/2303.02665) | 本文提出了一种新模型，异构图跨模态网络（HGCN），它学习跨模态边缘，可以适应各种空间和时间尺度，有效地连接了跨模态的相关节点，在声音事件分类中表现出最先进的性能。 |
| [^3] | [Vision, Deduction and Alignment: An Empirical Study on Multi-modal Knowledge Graph Alignment.](http://arxiv.org/abs/2302.08774) | 本研究构建了八个大规模的、配备图像的实体对齐基准Multi-OpenEA，并开发了一种新的多模态EA方法LODEME，利用逻辑推理和多模态KG嵌入，实现了最先进的性能。 |
| [^4] | [LDMIC: Learning-based Distributed Multi-view Image Coding.](http://arxiv.org/abs/2301.09799) | LDMIC是一种基于学习的分布式多视图图像编码框架，通过独立编码器和联合上下文传输模块实现了全局视图间的相关性捕捉，对几何关系不敏感。 |
| [^5] | [Temporal Sentence Grounding in Videos: A Survey and Future Directions.](http://arxiv.org/abs/2201.08071) | 本文综述了视频中的时间句子定位（TSGV）的基本概念和当前研究现状，以及未来研究方向。TSGV旨在从未经修剪的视频中检索与语言查询语义对应的时间时刻，连接计算机视觉和自然语言，是两个社区研究人员的重点关注点。 |

# 详细

[^1]: 利用预训练的AudioLDM进行文本到声音生成：基准研究

    Leveraging Pre-trained AudioLDM for Text to Sound Generation: A Benchmark Study. (arXiv:2303.03857v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2303.03857](http://arxiv.org/abs/2303.03857)

    本文研究了使用预训练的AudioLDM作为声音生成的骨干的优势，证明了在数据稀缺情况下使用预训练模型进行文本到声音生成的优势，并在几个常用数据集上使用相同的评估协议评估了各种文本到声音生成系统，为未来的研究提供了基础。

    This paper investigates the advantages of using pre-trained AudioLDM as the backbone for sound generation, demonstrates the benefits of using pre-trained models for text-to-sound generation in data-scarcity scenarios, and evaluates various text-to-sound generation systems on several frequently used datasets under the same evaluation protocols to provide a basis for future research.

    深度神经网络最近在文本提示下实现了声音生成的突破。尽管它们的表现很有前途，但当前的文本到声音生成模型在小规模数据集（例如过度拟合）上面临问题，从而显著限制了它们的性能。在本文中，我们研究了使用预训练的AudioLDM作为声音生成的骨干的优势。我们的研究证明了在数据稀缺情况下使用预训练模型进行文本到声音生成的优势。此外，实验表明，不同的训练策略（例如训练条件）可能会影响AudioLDM在不同规模的数据集上的性能。为了促进未来的研究，我们还在几个常用数据集上使用相同的评估协议评估了各种文本到声音生成系统，这些协议允许在共同基础上公平比较和基准测试这些方法。

    Deep neural networks have recently achieved breakthroughs in sound generation with text prompts. Despite their promising performance, current text-to-sound generation models face issues on small-scale datasets (e.g., overfitting), significantly limiting their performance. In this paper, we investigate the use of pre-trained AudioLDM, the state-of-the-art model for text-to-audio generation, as the backbone for sound generation. Our study demonstrates the advantages of using pre-trained models for text-to-sound generation, especially in data-scarcity scenarios. In addition, experiments show that different training strategies (e.g., training conditions) may affect the performance of AudioLDM on datasets of different scales. To facilitate future studies, we also evaluate various text-to-sound generation systems on several frequently used datasets under the same evaluation protocols, which allow fair comparisons and benchmarking of these methods on the common ground.
    
[^2]: 异构图学习在声音事件分类中的应用

    Heterogeneous Graph Learning for Acoustic Event Classification. (arXiv:2303.02665v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2303.02665](http://arxiv.org/abs/2303.02665)

    本文提出了一种新模型，异构图跨模态网络（HGCN），它学习跨模态边缘，可以适应各种空间和时间尺度，有效地连接了跨模态的相关节点，在声音事件分类中表现出最先进的性能。

    This paper proposes a new model, Heterogeneous Graph Crossmodal Network (HGCN), which learns crossmodal edges and can adapt to various spatial and temporal scales, effectively connecting relevant nodes across modalities. It achieves state-of-the-art performance in acoustic event classification.

    异构图提供了一种紧凑、高效、可扩展的方式来建模涉及多个不同模态的数据。这使得使用异构图来建模音频视觉数据成为一种有吸引力的选择。然而，图结构在音频视觉数据中并不自然。音频视觉数据的图是手动构建的，这既困难又次优。在这项工作中，我们通过（i）提出一种参数化图构建策略来解决这个问题，以及（ii）学习跨模态边缘。为此，我们开发了一种新模型，异构图跨模态网络（HGCN），它学习跨模态边缘。我们提出的模型可以适应各种空间和时间尺度，因为它是参数化构建的，而可学习的跨模态边缘有效地连接了跨模态的相关节点。在一个大型基准数据集（AudioSet）上的实验表明，我们的模型是最先进的（0.53平均精度），优于transfo。

    Heterogeneous graphs provide a compact, efficient, and scalable way to model data involving multiple disparate modalities. This makes modeling audiovisual data using heterogeneous graphs an attractive option. However, graph structure does not appear naturally in audiovisual data. Graphs for audiovisual data are constructed manually which is both difficult and sub-optimal. In this work, we address this problem by (i) proposing a parametric graph construction strategy for the intra-modal edges, and (ii) learning the crossmodal edges. To this end, we develop a new model, heterogeneous graph crossmodal network (HGCN) that learns the crossmodal edges. Our proposed model can adapt to various spatial and temporal scales owing to its parametric construction, while the learnable crossmodal edges effectively connect the relevant nodes across modalities. Experiments on a large benchmark dataset (AudioSet) show that our model is state-of-the-art (0.53 mean average precision), outperforming transfo
    
[^3]: 视觉、推理和对齐：多模态知识图谱对齐的实证研究

    Vision, Deduction and Alignment: An Empirical Study on Multi-modal Knowledge Graph Alignment. (arXiv:2302.08774v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2302.08774](http://arxiv.org/abs/2302.08774)

    本研究构建了八个大规模的、配备图像的实体对齐基准Multi-OpenEA，并开发了一种新的多模态EA方法LODEME，利用逻辑推理和多模态KG嵌入，实现了最先进的性能。

    This study constructed eight large-scale, image-equipped entity alignment benchmarks named Multi-OpenEA, and developed a new multi-modal EA method named LODEME, which utilizes logical deduction and multi-modal KG embedding, achieving state-of-the-art performance on Multi-OpenEA and other existing multi-modal EA benchmarks.

    知识图谱中的实体对齐在知识工程中起着至关重要的作用。现有的实体对齐方法主要集中在利用图形结构和实体属性（包括文字），但忽略了现代多模态知识图谱中常见的图像。在本研究中，我们首先构建了Multi-OpenEA——八个大规模的、配备图像的实体对齐基准，并评估了一些现有的基于嵌入的方法来利用图像。鉴于视觉模态信息和逻辑推理的互补性质，我们进一步开发了一种新的多模态EA方法，名为LODEME，使用逻辑推理和多模态KG嵌入，在Multi-OpenEA和其他现有的多模态EA基准上实现了最先进的性能。

    Entity alignment (EA) for knowledge graphs (KGs) plays a critical role in knowledge engineering. Existing EA methods mostly focus on utilizing the graph structures and entity attributes (including literals), but ignore images that are common in modern multi-modal KGs. In this study we first constructed Multi-OpenEA -- eight large-scale, image-equipped EA benchmarks, and then evaluated some existing embedding-based methods for utilizing images. In view of the complementary nature of visual modal information and logical deduction, we further developed a new multi-modal EA method named LODEME using logical deduction and multi-modal KG embedding, with state-of-the-art performance achieved on Multi-OpenEA and other existing multi-modal EA benchmarks.
    
[^4]: LDMIC：基于学习的分布式多视图图像编码

    LDMIC: Learning-based Distributed Multi-view Image Coding. (arXiv:2301.09799v2 [eess.IV] UPDATED)

    [http://arxiv.org/abs/2301.09799](http://arxiv.org/abs/2301.09799)

    LDMIC是一种基于学习的分布式多视图图像编码框架，通过独立编码器和联合上下文传输模块实现了全局视图间的相关性捕捉，对几何关系不敏感。

    LDMIC is a learning-based distributed multi-view image coding framework that captures global inter-view correlations through independent encoders and a joint context transfer module based on the cross-attention mechanism, which is insensitive to geometric relations.

    多视图图像压缩在3D相关应用中起着至关重要的作用。现有方法采用预测编码架构，需要联合编码压缩相应的视差和残差信息。这要求相机之间进行协作，并强制执行不同视图之间的极线几何约束，这使得在具有随机重叠视野的分布式相机系统中部署这些方法具有挑战性。同时，分布式源编码理论表明，可以通过独立编码和联合解码实现相关源的高效数据压缩，这激发了我们设计基于学习的分布式多视图图像编码（LDMIC）框架的动机。通过独立编码器，LDMIC引入了一个简单而有效的基于交叉注意机制的联合上下文传输模块，以有效捕捉全局视图间的相关性，对几何关系不敏感。

    Multi-view image compression plays a critical role in 3D-related applications. Existing methods adopt a predictive coding architecture, which requires joint encoding to compress the corresponding disparity as well as residual information. This demands collaboration among cameras and enforces the epipolar geometric constraint between different views, which makes it challenging to deploy these methods in distributed camera systems with randomly overlapping fields of view. Meanwhile, distributed source coding theory indicates that efficient data compression of correlated sources can be achieved by independent encoding and joint decoding, which motivates us to design a learning-based distributed multi-view image coding (LDMIC) framework. With independent encoders, LDMIC introduces a simple yet effective joint context transfer module based on the cross-attention mechanism at the decoder to effectively capture the global inter-view correlations, which is insensitive to the geometric relation
    
[^5]: 视频中的时间句子定位：综述与未来方向

    Temporal Sentence Grounding in Videos: A Survey and Future Directions. (arXiv:2201.08071v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2201.08071](http://arxiv.org/abs/2201.08071)

    本文综述了视频中的时间句子定位（TSGV）的基本概念和当前研究现状，以及未来研究方向。TSGV旨在从未经修剪的视频中检索与语言查询语义对应的时间时刻，连接计算机视觉和自然语言，是两个社区研究人员的重点关注点。

    This survey summarizes the fundamental concepts and current research status of temporal sentence grounding in videos (TSGV), also known as natural language video localization (NLVL) or video moment retrieval (VMR), as well as future research directions. TSGV aims to retrieve a temporal moment that semantically corresponds to a language query from an untrimmed video, connecting computer vision and natural language, and has drawn significant attention from researchers in both communities.

    视频中的时间句子定位（TSGV），又称自然语言视频定位（NLVL）或视频时刻检索（VMR），旨在从未经修剪的视频中检索与语言查询语义对应的时间时刻。连接计算机视觉和自然语言，TSGV引起了两个社区研究人员的重视。本综述试图提供TSGV中基本概念和当前研究现状的总结，以及未来研究方向。作为背景，我们以教程的形式介绍了TSGV中功能组件的常见结构：从原始视频和语言查询的特征提取到目标时刻的答案预测。然后，我们回顾了多模态理解和交互的技术，这是TSGV的重点关注点，以实现两种模态之间的有效对齐。我们构建了TSGV技术的分类法，并详细阐述了不同类别的方法及其优缺点。

    Temporal sentence grounding in videos (TSGV), \aka natural language video localization (NLVL) or video moment retrieval (VMR), aims to retrieve a temporal moment that semantically corresponds to a language query from an untrimmed video. Connecting computer vision and natural language, TSGV has drawn significant attention from researchers in both communities. This survey attempts to provide a summary of fundamental concepts in TSGV and current research status, as well as future research directions. As the background, we present a common structure of functional components in TSGV, in a tutorial style: from feature extraction from raw video and language query, to answer prediction of the target moment. Then we review the techniques for multimodal understanding and interaction, which is the key focus of TSGV for effective alignment between the two modalities. We construct a taxonomy of TSGV techniques and elaborate the methods in different categories with their strengths and weaknesses. La
    


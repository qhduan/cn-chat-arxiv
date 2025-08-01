# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FGeo-HyperGNet: Geometry Problem Solving Integrating Formal Symbolic System and Hypergraph Neural Network](https://arxiv.org/abs/2402.11461) | 该论文提出了FGeo-HyperGNet，将形式符号系统和超图神经网络集成，用于解决几何问题，实现自动执行人类化的几何演绎推理。 |
| [^2] | [How AI Ideas Affect the Creativity, Diversity, and Evolution of Human Ideas: Evidence From a Large, Dynamic Experiment.](http://arxiv.org/abs/2401.13481) | AI思想对个体创造力没有影响，但增加了整体思想多样性的数量和变化速率。 |
| [^3] | [An Efficient Intelligent Semi-Automated Warehouse Inventory Stocktaking System.](http://arxiv.org/abs/2309.12365) | 本研究提出了一个智能库存管理系统，通过结合条码和分布式flutter应用技术，以及大数据分析实现数据驱动的决策，解决了库存管理中的准确性、监测延迟和过度依赖主观经验的挑战。 |
| [^4] | [Divided Attention: Unsupervised Multi-Object Discovery with Contextually Separated Slots.](http://arxiv.org/abs/2304.01430) | 该论文提出了一种新的无监督多对象发现方法，通过一种上下文分隔的槽结构来将视觉场分割为独立运动区域，并用对抗性标准来保证解码器无法重构整个光流。 |

# 详细

[^1]: FGeo-HyperGNet: 几何问题求解中集成形式符号系统和超图神经网络

    FGeo-HyperGNet: Geometry Problem Solving Integrating Formal Symbolic System and Hypergraph Neural Network

    [https://arxiv.org/abs/2402.11461](https://arxiv.org/abs/2402.11461)

    该论文提出了FGeo-HyperGNet，将形式符号系统和超图神经网络集成，用于解决几何问题，实现自动执行人类化的几何演绎推理。

    

    几何问题的求解一直是自动推理和人工智能领域中长期存在的挑战。这是我们系列作品中的第五篇文章，我们构建了一个神经符号系统，用于自动执行类似人类的几何演绎推理。符号部分是建立在FormalGeo上的形式系统，可以自动执行几何关系推理和代数计算，并将求解过程组织成一个带有条件作为超节点和定理作为超边的解决方案超树。神经部分称为HyperGNet，是基于注意力机制的超图神经网络，包括一个编码器来有效编码超树的结构和语义信息，以及一个求解器来提供问题求解指导。神经部分根据超树预测定理，而符号部分应用定理并更新超树，从而形成一个预测-

    arXiv:2402.11461v1 Announce Type: new  Abstract: Geometry problem solving has always been a long-standing challenge in the fields of automated reasoning and artificial intelligence. This is the fifth article in a series of our works, we built a neural-symbolic system to automatically perform human-like geometric deductive reasoning. The symbolic part is a formal system built on FormalGeo, which can automatically perform geomertic relational reasoning and algebraic calculations and organize the solving process into a solution hypertree with conditions as hypernodes and theorems as hyperedges. The neural part, called HyperGNet, is a hypergraph neural network based on the attention mechanism, including a encoder to effectively encode the structural and semantic information of the hypertree, and a solver to provide problem-solving guidance. The neural part predicts theorems according to the hypertree, and the symbolic part applies theorems and updates the hypertree, thus forming a Predict-
    
[^2]: AI思想如何影响人类思想的创造力、多样性和进化：来自一个大规模动态实验的证据

    How AI Ideas Affect the Creativity, Diversity, and Evolution of Human Ideas: Evidence From a Large, Dynamic Experiment. (arXiv:2401.13481v1 [cs.CY])

    [http://arxiv.org/abs/2401.13481](http://arxiv.org/abs/2401.13481)

    AI思想对个体创造力没有影响，但增加了整体思想多样性的数量和变化速率。

    

    大规模语言模型输出的接触正在迅速增加。观看到AI生成的思想将如何影响人类思想？我们进行了一个实验（800+参与者，40+个国家），参与者观看了来自ChatGPT或之前实验参与者的创意思想，然后进行了自己的创意思考。我们变化了AI生成示例的数量（无、低、高曝光）以及示例是否标记为“AI”（披露）。我们的动态实验设计 - 在同一实验条件下，使用之前参与者的思想作为未来参与者的刺激 - 模拟了文化创造的相互依赖过程：创造性思想建立在之前的思想基础上。因此，我们捕捉到了LLM“在文化循环中”的复合效应。我们发现高AI曝光（但不是低AI曝光）并没有影响个人思想的创造力，但增加了整体思想多样性的平均数量和变化速率。AI使思想多样性的累积效应增强了。

    Exposure to large language model output is rapidly increasing. How will seeing AI-generated ideas affect human ideas? We conducted an experiment (800+ participants, 40+ countries) where participants viewed creative ideas that were from ChatGPT or prior experimental participants and then brainstormed their own idea. We varied the number of AI-generated examples (none, low, or high exposure) and if the examples were labeled as 'AI' (disclosure). Our dynamic experiment design -- ideas from prior participants in an experimental condition are used as stimuli for future participants in the same experimental condition -- mimics the interdependent process of cultural creation: creative ideas are built upon prior ideas. Hence, we capture the compounding effects of having LLMs 'in the culture loop'. We find that high AI exposure (but not low AI exposure) did not affect the creativity of individual ideas but did increase the average amount and rate of change of collective idea diversity. AI made 
    
[^3]: 一个高效的智能半自动仓库库存盘点系统

    An Efficient Intelligent Semi-Automated Warehouse Inventory Stocktaking System. (arXiv:2309.12365v1 [cs.HC])

    [http://arxiv.org/abs/2309.12365](http://arxiv.org/abs/2309.12365)

    本研究提出了一个智能库存管理系统，通过结合条码和分布式flutter应用技术，以及大数据分析实现数据驱动的决策，解决了库存管理中的准确性、监测延迟和过度依赖主观经验的挑战。

    

    在不断发展的供应链管理背景下，高效的库存管理对于企业变得越来越重要。然而，传统的手工和经验驱动的方法往往难以满足现代市场需求的复杂性。本研究引入了一种智能库存管理系统，以解决与数据不准确、监测延迟和过度依赖主观经验的预测相关的挑战。该系统结合了条码和分布式 flutter 应用技术，用于智能感知，并通过全面的大数据分析实现数据驱动的决策。通过仔细的分析、系统设计、关键技术探索和模拟验证，成功展示了所提出系统的有效性。该智能系统实现了二级监测、高频检查和人工智能驱动的预测，从而提高了自动化程度。

    In the context of evolving supply chain management, the significance of efficient inventory management has grown substantially for businesses. However, conventional manual and experience-based approaches often struggle to meet the complexities of modern market demands. This research introduces an intelligent inventory management system to address challenges related to inaccurate data, delayed monitoring, and overreliance on subjective experience in forecasting. The proposed system integrates bar code and distributed flutter application technologies for intelligent perception, alongside comprehensive big data analytics to enable data-driven decision-making. Through meticulous analysis, system design, critical technology exploration, and simulation validation, the effectiveness of the proposed system is successfully demonstrated. The intelligent system facilitates second-level monitoring, high-frequency checks, and artificial intelligence-driven forecasting, consequently enhancing the au
    
[^4]: 分离的关注力：基于上下文分离槽的无监督多对象发现

    Divided Attention: Unsupervised Multi-Object Discovery with Contextually Separated Slots. (arXiv:2304.01430v1 [cs.CV])

    [http://arxiv.org/abs/2304.01430](http://arxiv.org/abs/2304.01430)

    该论文提出了一种新的无监督多对象发现方法，通过一种上下文分隔的槽结构来将视觉场分割为独立运动区域，并用对抗性标准来保证解码器无法重构整个光流。

    

    我们提出了一种将视觉场分割为独立运动区域的方法，不需要任何基础真值或监督。它由基于槽关注的对抗条件编码器-解码器架构组成，修改为使用图像作为上下文来解码光流，而不是尝试重构图像本身。在结果的多模式表示中，一种模式（流）将馈送给编码器以产生单独的潜在代码（槽），而另一种模式（图像）将决定解码器从槽生成第一个模式（流）。由于惯常的自编码基于最小化重构误差，并不能防止整个流被编码到一个槽中，因此我们将损失修改为基于上下文信息分离的对抗性标准。

    We introduce a method to segment the visual field into independently moving regions, trained with no ground truth or supervision. It consists of an adversarial conditional encoder-decoder architecture based on Slot Attention, modified to use the image as context to decode optical flow without attempting to reconstruct the image itself. In the resulting multi-modal representation, one modality (flow) feeds the encoder to produce separate latent codes (slots), whereas the other modality (image) conditions the decoder to generate the first (flow) from the slots. This design frees the representation from having to encode complex nuisance variability in the image due to, for instance, illumination and reflectance properties of the scene. Since customary autoencoding based on minimizing the reconstruction error does not preclude the entire flow from being encoded into a single slot, we modify the loss to an adversarial criterion based on Contextual Information Separation. The resulting min-m
    


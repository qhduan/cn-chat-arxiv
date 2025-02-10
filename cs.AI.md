# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Is Meta-training Really Necessary for Molecular Few-Shot Learning ?](https://arxiv.org/abs/2404.02314) | 本文重新审视了分子数据微调方法，提出了基于马氏距离的正则化二次探针损失，并设计了块坐标下降优化器，使得在黑匣子设置下，简单微调方法在少样本学习中获得了竞争性表现，同时消除了特定预训练策略的需要。 |
| [^2] | [ADAPT to Robustify Prompt Tuning Vision Transformers](https://arxiv.org/abs/2403.13196) | 本文提出了ADAPT框架，用于在prompt调优范式中进行自适应对抗训练，增强视觉Transformer在下游任务中的稳健性。 |
| [^3] | [A Bayesian Approach to OOD Robustness in Image Classification](https://arxiv.org/abs/2403.07277) | 本文提出了一种基于贝叶斯方法的图像分类中OOD鲁棒性解决方案，利用扩展的组合神经网络和von Mises-Fisher核来处理真实世界的OOD问题。 |
| [^4] | [The DSA Transparency Database: Auditing Self-reported Moderation Actions by Social Media.](http://arxiv.org/abs/2312.10269) | DSA透明数据库对欧盟八大社交媒体平台在前100天提交的审核行动数据进行了全面分析，揭示了这些平台在审核行动方面的部分遵循程度。 |
| [^5] | [Pave the Way to Grasp Anything: Transferring Foundation Models for Universal Pick-Place Robots.](http://arxiv.org/abs/2306.05716) | 本研究提出了一种基于语言分割掩模的新方法，用于解决通用型机器人的泛化能力问题，提高了在开放域场景中新对象的抓取操作的学习效率和推广效果。 |
| [^6] | [A Group Symmetric Stochastic Differential Equation Model for Molecule Multi-modal Pretraining.](http://arxiv.org/abs/2305.18407) | MoleculeSDE是用于分子多模态预训练的群对称随机微分方程模型，通过在输入空间中直接生成3D几何与2D拓扑之间的转换，它能够更有效地保存分子结构信息。 |

# 详细

[^1]: 分子少样本学习是否真的需要元训练？

    Is Meta-training Really Necessary for Molecular Few-Shot Learning ?

    [https://arxiv.org/abs/2404.02314](https://arxiv.org/abs/2404.02314)

    本文重新审视了分子数据微调方法，提出了基于马氏距离的正则化二次探针损失，并设计了块坐标下降优化器，使得在黑匣子设置下，简单微调方法在少样本学习中获得了竞争性表现，同时消除了特定预训练策略的需要。

    

    最近，少样本学习在药物发现领域引起了极大关注，而最近快速增长的文献大多涉及复杂的元学习策略。本文重新审视了更为直接的分子数据微调方法，并提出了基于马氏距离的正则化二次探针损失。我们设计了一个专门的块坐标下降优化器，避免了我们损失函数的退化解。有趣的是，我们的简单微调方法在与最先进方法的比较中获得了极具竞争力的表现，同时适用于黑匣子设置，并消除了特定情节预训练策略的需要。此外，我们引入了一个新的基准来评估竞争方法对领域转移的稳健性。在这个设置下，我们的微调基线始终比元学习方法取得更好的结果。

    arXiv:2404.02314v1 Announce Type: cross  Abstract: Few-shot learning has recently attracted significant interest in drug discovery, with a recent, fast-growing literature mostly involving convoluted meta-learning strategies. We revisit the more straightforward fine-tuning approach for molecular data, and propose a regularized quadratic-probe loss based on the the Mahalanobis distance. We design a dedicated block-coordinate descent optimizer, which avoid the degenerate solutions of our loss. Interestingly, our simple fine-tuning approach achieves highly competitive performances in comparison to state-of-the-art methods, while being applicable to black-box settings and removing the need for specific episodic pre-training strategies. Furthermore, we introduce a new benchmark to assess the robustness of the competing methods to domain shifts. In this setting, our fine-tuning baseline obtains consistently better results than meta-learning methods.
    
[^2]: 使Prompt调优视觉Transformer更为健壮的ADAPT

    ADAPT to Robustify Prompt Tuning Vision Transformers

    [https://arxiv.org/abs/2403.13196](https://arxiv.org/abs/2403.13196)

    本文提出了ADAPT框架，用于在prompt调优范式中进行自适应对抗训练，增强视觉Transformer在下游任务中的稳健性。

    

    深度模型的性能，包括视觉Transformer，已知容易受到对抗性攻击的影响。许多现有对抗性防御方法，如对抗性训练，依赖于对整个模型进行全面微调以增加模型的稳健性。这些防御方法需要为每个任务存储整个模型的副本，而模型可能包含数十亿个参数。与此同时，参数高效的prompt调优被用来适应大型基于Transformer的模型到下游任务，无需保存大型副本。本文从稳健性的角度研究了对视觉Transformer进行下游任务的参数高效prompt调优。我们发现，之前的对抗性防御方法在应用到prompt调优范式时，存在梯度模糊并容易受到自适应攻击的影响。我们引入了ADAPT，一种在prompt调优范式中执行自适应对抗训练的新框架。

    arXiv:2403.13196v1 Announce Type: new  Abstract: The performance of deep models, including Vision Transformers, is known to be vulnerable to adversarial attacks. Many existing defenses against these attacks, such as adversarial training, rely on full-model fine-tuning to induce robustness in the models. These defenses require storing a copy of the entire model, that can have billions of parameters, for each task. At the same time, parameter-efficient prompt tuning is used to adapt large transformer-based models to downstream tasks without the need to save large copies. In this paper, we examine parameter-efficient prompt tuning of Vision Transformers for downstream tasks under the lens of robustness. We show that previous adversarial defense methods, when applied to the prompt tuning paradigm, suffer from gradient obfuscation and are vulnerable to adaptive attacks. We introduce ADAPT, a novel framework for performing adaptive adversarial training in the prompt tuning paradigm. Our meth
    
[^3]: 基于贝叶斯方法的图像分类中OOD鲁棒性解决方案

    A Bayesian Approach to OOD Robustness in Image Classification

    [https://arxiv.org/abs/2403.07277](https://arxiv.org/abs/2403.07277)

    本文提出了一种基于贝叶斯方法的图像分类中OOD鲁棒性解决方案，利用扩展的组合神经网络和von Mises-Fisher核来处理真实世界的OOD问题。

    

    计算机视觉中一个重要且未解决的问题是确保算法对图像领域的变化具有鲁棒性。我们在目标领域中处理此问题的情况下，但没有注释的图像。在面临真实世界的域之外（OOD）干扰和遮挡的OOD-CV基准挑战的激励下，我们引入了一种新颖的贝叶斯方法来实现物体分类的OOD鲁棒性。我们的工作扩展了已被证明在遮挡情况下具有鲁棒性但在OOD数据测试时严重降级的组合神经网络（CompNets）。我们利用了CompNets包含的在von Mises-Fisher（vMF）核表示的特征向量上定义的生成头，这些核大致对应于对象部分，并且可以在无监督的情况下学习。我们观察到不同域之间的某些vMF核是相似的，而另一些则不是。这使我们能够学习一个transiti

    arXiv:2403.07277v1 Announce Type: cross  Abstract: An important and unsolved problem in computer vision is to ensure that the algorithms are robust to changes in image domains. We address this problem in the scenario where we have access to images from the target domains but no annotations. Motivated by the challenges of the OOD-CV benchmark where we encounter real world Out-of-Domain (OOD) nuisances and occlusion, we introduce a novel Bayesian approach to OOD robustness for object classification. Our work extends Compositional Neural Networks (CompNets), which have been shown to be robust to occlusion but degrade badly when tested on OOD data. We exploit the fact that CompNets contain a generative head defined over feature vectors represented by von Mises-Fisher (vMF) kernels, which correspond roughly to object parts, and can be learned without supervision. We obverse that some vMF kernels are similar between different domains, while others are not. This enables us to learn a transiti
    
[^4]: DSA透明数据库：社交媒体自我报告的审核行动

    The DSA Transparency Database: Auditing Self-reported Moderation Actions by Social Media. (arXiv:2312.10269v2 [cs.SI] UPDATED)

    [http://arxiv.org/abs/2312.10269](http://arxiv.org/abs/2312.10269)

    DSA透明数据库对欧盟八大社交媒体平台在前100天提交的审核行动数据进行了全面分析，揭示了这些平台在审核行动方面的部分遵循程度。

    

    从2023年9月开始，数字服务法案(DSA)要求大型在线平台向DSA透明数据库提交关于他们在欧盟内采取的每个审核行动的详细数据。从一开始，这个集中式数据库就引起了学术界的兴趣，因为它是现实世界在线审核数据的一个前所未有的、可能是独特的宝库。在这里，我们深入分析了欧盟八个最大社交媒体平台在数据库的前100天提交的所有3.53亿条记录。具体而言，我们对平台之间进行了比较研究，包括：审核行动的数量、决策依据、应用的限制类型、审核内容类型、审核行动的及时性和提交情况，以及使用的自动化程度。此外，我们系统地与平台自己的透明报告进行了内容交叉检查。我们的分析揭示了以下结果。(i)平台只在一定程度上遵循了审核行动的哲学和方法论。

    Since September 2023, the Digital Services Act (DSA) obliges large online platforms to submit detailed data on each moderation action they take within the European Union (EU) to the DSA Transparency Database. From its inception, this centralized database has sparked scholarly interest as an unprecedented and potentially unique trove of data on real-world online moderation. Here, we thoroughly analyze all 353.12M records submitted by the eight largest social media platforms in the EU during the first 100 days of the database. Specifically, we conduct a platform-wise comparative study of their: volume of moderation actions, grounds for decision, types of applied restrictions, types of moderated content, timeliness in undertaking and submitting moderation actions, and use of automation. Furthermore, we systematically cross-check the contents of the database with the platforms' own transparency reports. Our analyses reveal that (i) the platforms adhered only in part to the philosophy and s
    
[^5]: 为抓住任何物品铺平道路：基于迁移学习的通用抓取放置机器人模型

    Pave the Way to Grasp Anything: Transferring Foundation Models for Universal Pick-Place Robots. (arXiv:2306.05716v1 [cs.RO])

    [http://arxiv.org/abs/2306.05716](http://arxiv.org/abs/2306.05716)

    本研究提出了一种基于语言分割掩模的新方法，用于解决通用型机器人的泛化能力问题，提高了在开放域场景中新对象的抓取操作的学习效率和推广效果。

    

    提高通用型机器人的泛化能力一直是研究社区长期追求的重要挑战。现有的方法通常依赖于收集大规模现实世界机器人数据，如 RT-1 数据集。然而，这些方法通常效率低下，限制了它们在具有新对象和多样背景的开放域场景中的能力。本文提出了一种新的范例，有效地利用最先进的基础模型生成的基于语言的分割掩模，以解决日常场景中广泛的拾放机器人操作任务。通过将掩模传达的精确语义和几何形状集成到我们的多视角策略模型中，我们的方法可以感知准确的物体姿态并实现高效学习，同时也有助于有效的新对象的推广。我们的方法同时可以实现在训练时观察到相似形状的新物体的抓取操作。

    Improving the generalization capabilities of general-purpose robotic agents has long been a significant challenge actively pursued by research communities. Existing approaches often rely on collecting large-scale real-world robotic data, such as the RT-1 dataset. However, these approaches typically suffer from low efficiency, limiting their capability in open-domain scenarios with new objects, and diverse backgrounds. In this paper, we propose a novel paradigm that effectively leverages language-grounded segmentation masks generated by state-of-the-art foundation models, to address a wide range of pick-and-place robot manipulation tasks in everyday scenarios. By integrating precise semantics and geometries conveyed from masks into our multi-view policy model, our approach can perceive accurate object poses and enable sample-efficient learning. Besides, such design facilitates effective generalization for grasping new objects with similar shapes observed during training. Our approach co
    
[^6]: 一种用于分子多模态预训练的群对称随机微分方程模型。

    A Group Symmetric Stochastic Differential Equation Model for Molecule Multi-modal Pretraining. (arXiv:2305.18407v1 [cs.LG])

    [http://arxiv.org/abs/2305.18407](http://arxiv.org/abs/2305.18407)

    MoleculeSDE是用于分子多模态预训练的群对称随机微分方程模型，通过在输入空间中直接生成3D几何与2D拓扑之间的转换，它能够更有效地保存分子结构信息。

    

    分子预训练已经成为提高基于 AI 的药物发现性能的主流方法。然而，大部分现有的方法只关注单一的模态。最近的研究表明，最大化两种模态之间的互信息（MI）可以增强分子表示能力。而现有的分子多模态预训练方法基于从拓扑和几何编码的表示空间来估计 MI，因此丢失了分子的关键结构信息。为解决这一问题，我们提出了 MoleculeSDE。MoleculeSDE利用群对称（如 SE（3）-等变和反射-反对称）随机微分方程模型在输入空间中直接生成 3D 几何形状与 2D 拓扑之间的转换。它不仅获得更紧的MI界限，而且还能够有效地保存分子结构信息。

    Molecule pretraining has quickly become the go-to schema to boost the performance of AI-based drug discovery. Naturally, molecules can be represented as 2D topological graphs or 3D geometric point clouds. Although most existing pertaining methods focus on merely the single modality, recent research has shown that maximizing the mutual information (MI) between such two modalities enhances the molecule representation ability. Meanwhile, existing molecule multi-modal pretraining approaches approximate MI based on the representation space encoded from the topology and geometry, thus resulting in the loss of critical structural information of molecules. To address this issue, we propose MoleculeSDE. MoleculeSDE leverages group symmetric (e.g., SE(3)-equivariant and reflection-antisymmetric) stochastic differential equation models to generate the 3D geometries from 2D topologies, and vice versa, directly in the input space. It not only obtains tighter MI bound but also enables prosperous dow
    


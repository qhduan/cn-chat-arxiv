# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Open Assistant Toolkit -- version 2](https://arxiv.org/abs/2403.00586) | OAT-v2是一个可扩展且灵活的开源助手平台，支持多个领域和用户交互方式，提供模块化系统组件和开放模型与软件，有助于未来多模态虚拟助手的发展。 |
| [^2] | [Generalized User Representations for Transfer Learning](https://arxiv.org/abs/2403.00584) | 提出了一个在大规模推荐系统中有效表示用户口味的通用用户表示框架，结合表示学习和迁移学习，同时提出了管理生产模型中该框架部署的新颖解决方案。 |
| [^3] | [IAI MovieBot 2.0: An Enhanced Research Platform with Trainable Neural Components and Transparent User Modeling](https://arxiv.org/abs/2403.00520) | IAI MovieBot 2.0增强了可训练的神经元件，透明用户建模和用户界面改进，使其成为一个研究平台。 |
| [^4] | [Recommending Target Actions Outside Sessions in the Data-poor Insurance Domain](https://arxiv.org/abs/2403.00368) | 基于循环神经网络的推荐模型应对保险领域数据稀缺性，学习预测会话外的目标动作。 |
| [^5] | [Dual Pose-invariant Embeddings: Learning Category and Object-specific Discriminative Representations for Recognition and Retrieval](https://arxiv.org/abs/2403.00272) | 通过提出一种基于注意力的双编码器架构和特殊设计的损失函数，本文实现了在训练过程中同时学习基于类别和基于对象身份的嵌入，从而显著提高了姿势不变的对象识别和检索性能。 |
| [^6] | [Influencing Bandits: Arm Selection for Preference Shaping](https://arxiv.org/abs/2403.00036) | 该论文考虑了在非静态多臂赌博机中，通过观察奖励来积极和消极地强化人群偏好，并提出了用于最大化支持预定手臂的人口比例的算法。对于不同意见动态，提出了不同的策略并分析了后悔，最后讨论了多个推荐系统共存的情况。 |
| [^7] | [Distributionally Robust Unsupervised Dense Retrieval Training on Web Graphs.](http://arxiv.org/abs/2310.16605) | 本论文提出了一种无监督的密集检索模型Web-DRO，它利用网络结构进行聚类并重新加权，在无监督场景中显著提高了检索效果。群组分布鲁棒优化方法指导模型对高对比损失的群组分配更多权重，在训练过程中更加关注最坏情况。实验结果表明，结合URL信息的网络图训练能达到最佳的聚类性能。 |

# 详细

[^1]: 开放助手工具包--第2版

    Open Assistant Toolkit -- version 2

    [https://arxiv.org/abs/2403.00586](https://arxiv.org/abs/2403.00586)

    OAT-v2是一个可扩展且灵活的开源助手平台，支持多个领域和用户交互方式，提供模块化系统组件和开放模型与软件，有助于未来多模态虚拟助手的发展。

    

    我们介绍了开放助手工具包（OAT-v2）的第二个版本，这是一个开源的面向任务的对话系统，用于构建生成式神经模型。OAT-v2是一个可扩展且灵活的助手平台，支持多个领域和用户交互方式。它将用户话语处理分为模块化系统组件，包括动作代码生成、多模态内容检索和知识增强响应生成等子模块。经过多年的Alexa TaskBot挑战的开发，OAT-v2是一个经过验证的系统，可以在实验和实际部署中进行可扩展和健壮的实验。OAT-v2提供用于研究和商业应用的开放模型和软件，以促进未来多样应用和丰富交互类型的多模态虚拟助手发展。

    arXiv:2403.00586v1 Announce Type: new  Abstract: We present the second version of the Open Assistant Toolkit (OAT-v2), an open-source task-oriented conversational system for composing generative neural models. OAT-v2 is a scalable and flexible assistant platform supporting multiple domains and modalities of user interaction. It splits processing a user utterance into modular system components, including submodules such as action code generation, multimodal content retrieval, and knowledge-augmented response generation. Developed over multiple years of the Alexa TaskBot challenge, OAT-v2 is a proven system that enables scalable and robust experimentation in experimental and real-world deployment. OAT-v2 provides open models and software for research and commercial applications to enable the future of multimodal virtual assistants across diverse applications and types of rich interaction.
    
[^2]: 推荐系统中的通用用户表示的迁移学习

    Generalized User Representations for Transfer Learning

    [https://arxiv.org/abs/2403.00584](https://arxiv.org/abs/2403.00584)

    提出了一个在大规模推荐系统中有效表示用户口味的通用用户表示框架，结合表示学习和迁移学习，同时提出了管理生产模型中该框架部署的新颖解决方案。

    

    我们提出了一个新颖的框架，用于大规模推荐系统中的用户表示，旨在以通用方式有效表示多样化的用户口味。我们的方法采用两阶段方法，结合表示学习和迁移学习。表示学习模型使用自动编码器将各种用户特征压缩成表示空间。在第二阶段，下游任务特定模型通过迁移学习利用用户表示，而不是单独策划用户特征。我们进一步在表示的输入特征上增强这种方法，以增加灵活性，并实现对用户事件（包括新用户体验）的几乎实时反应。此外，我们提出了一个新颖的解决方案，用于管理该框架在生产模型中的部署，允许下游模型独立工作。我们通过严格的线下验证了我们框架的性能。

    arXiv:2403.00584v1 Announce Type: cross  Abstract: We present a novel framework for user representation in large-scale recommender systems, aiming at effectively representing diverse user taste in a generalized manner. Our approach employs a two-stage methodology combining representation learning and transfer learning. The representation learning model uses an autoencoder that compresses various user features into a representation space. In the second stage, downstream task-specific models leverage user representations via transfer learning instead of curating user features individually. We further augment this methodology on the representation's input features to increase flexibility and enable reaction to user events, including new user experiences, in Near-Real Time. Additionally, we propose a novel solution to manage deployment of this framework in production models, allowing downstream models to work independently. We validate the performance of our framework through rigorous offl
    
[^3]: IAI MovieBot 2.0：具有可训练神经元件和透明用户建模的增强研究平台

    IAI MovieBot 2.0: An Enhanced Research Platform with Trainable Neural Components and Transparent User Modeling

    [https://arxiv.org/abs/2403.00520](https://arxiv.org/abs/2403.00520)

    IAI MovieBot 2.0增强了可训练的神经元件，透明用户建模和用户界面改进，使其成为一个研究平台。

    

    虽然对话式推荐系统的兴趣日益增长，但目前缺乏适合作为综合研究平台的运作系统。本文介绍了IAI MovieBot对话式电影推荐系统的增强版本，旨在将其发展为一个稳健且可适应的平台，用于进行面向用户的实验。此次增强的关键亮点包括添加了可训练的神经元件进行自然语言理解和对话策略，透明且可解释的用户偏好建模，以及改进用户界面和研究基础设施。

    arXiv:2403.00520v1 Announce Type: new  Abstract: While interest in conversational recommender systems has been on the rise, operational systems suitable for serving as research platforms for comprehensive studies are currently lacking. This paper introduces an enhanced version of the IAI MovieBot conversational movie recommender system, aiming to evolve it into a robust and adaptable platform for conducting user-facing experiments. The key highlights of this enhancement include the addition of trainable neural components for natural language understanding and dialogue policy, transparent and explainable modeling of user preferences, along with improvements in the user interface and research infrastructure.
    
[^4]: 在数据匮乏的保险领域推荐会话外的目标动作

    Recommending Target Actions Outside Sessions in the Data-poor Insurance Domain

    [https://arxiv.org/abs/2403.00368](https://arxiv.org/abs/2403.00368)

    基于循环神经网络的推荐模型应对保险领域数据稀缺性，学习预测会话外的目标动作。

    

    在保险产品中提供个性化推荐特别具有挑战性，因为保险领域具有固有和独特的特征。本文针对保险领域数据稀缺性提出了不同的推荐模型，利用循环神经网络和三种不同类型的损失函数与架构（交叉熵、截尾威布尔、注意力）。我们的模型通过学习多个会话和不同类型的用户行为来应对数据稀缺性。此外，与以往基于会话的模型不同，我们的模型学会预测会话内未发生的目标动作。

    arXiv:2403.00368v1 Announce Type: new  Abstract: Providing personalized recommendations for insurance products is particularly challenging due to the intrinsic and distinctive features of the insurance domain. First, unlike more traditional domains like retail, movie etc., a large amount of user feedback is not available and the item catalog is smaller. Second, due to the higher complexity of products, the majority of users still prefer to complete their purchases over the phone instead of online. We present different recommender models to address such data scarcity in the insurance domain. We use recurrent neural networks with 3 different types of loss functions and architectures (cross-entropy, censored Weibull, attention). Our models cope with data scarcity by learning from multiple sessions and different types of user actions. Moreover, differently from previous session-based models, our models learn to predict a target action that does not happen within the session. Our models out
    
[^5]: 双重姿势不变嵌入：学习用于识别和检索的类别和对象特定的判别性表示

    Dual Pose-invariant Embeddings: Learning Category and Object-specific Discriminative Representations for Recognition and Retrieval

    [https://arxiv.org/abs/2403.00272](https://arxiv.org/abs/2403.00272)

    通过提出一种基于注意力的双编码器架构和特殊设计的损失函数，本文实现了在训练过程中同时学习基于类别和基于对象身份的嵌入，从而显著提高了姿势不变的对象识别和检索性能。

    

    在姿势不变的对象识别和检索的背景下，我们证明在训练过程中同时学习基于类别和基于对象身份的嵌入是可能的，并且可以显著提高性能。本文提出了一种基于注意力的双编码器架构，配合特别设计的损失函数，同时优化两个不同嵌入空间中的类间和类内距离，一个用于类别嵌入，另一个用于对象级嵌入。

    arXiv:2403.00272v1 Announce Type: cross  Abstract: In the context of pose-invariant object recognition and retrieval, we demonstrate that it is possible to achieve significant improvements in performance if both the category-based and the object-identity-based embeddings are learned simultaneously during training. In hindsight, that sounds intuitive because learning about the categories is more fundamental than learning about the individual objects that correspond to those categories. However, to the best of what we know, no prior work in pose-invariant learning has demonstrated this effect. This paper presents an attention-based dual-encoder architecture with specially designed loss functions that optimize the inter- and intra-class distances simultaneously in two different embedding spaces, one for the category embeddings and the other for the object-level embeddings. The loss functions we have proposed are pose-invariant ranking losses that are designed to minimize the intra-class d
    
[^6]: 影响Bandits：用于形塑偏好的手臂选择

    Influencing Bandits: Arm Selection for Preference Shaping

    [https://arxiv.org/abs/2403.00036](https://arxiv.org/abs/2403.00036)

    该论文考虑了在非静态多臂赌博机中，通过观察奖励来积极和消极地强化人群偏好，并提出了用于最大化支持预定手臂的人口比例的算法。对于不同意见动态，提出了不同的策略并分析了后悔，最后讨论了多个推荐系统共存的情况。

    

    我们考虑一个非静态多臂赌博机，在这其中人群的偏好受到观察到的奖励的积极和消极强化。算法的目标是塑造人群的偏好，以最大化支持预定手臂的人口比例。对于二元意见的情况，考虑了两种意见动态 -- 递减弹性（建模为具有增加球数的Polya采样）和常量弹性（使用投票者模型）。对于第一种情况，我们描述了一种探索-然后-承诺策略和一种Thompson采样策略，并分析了每种策略的后悔。然后，我们展示了这些算法及其分析可推广到常弹性情况。我们还描述了一种基于Thompson采样的算法，用于当存在两种以上类型的意见情况。最后，我们讨论了存在多个推荐系统的情况引发的情况。

    arXiv:2403.00036v1 Announce Type: cross  Abstract: We consider a non stationary multi-armed bandit in which the population preferences are positively and negatively reinforced by the observed rewards. The objective of the algorithm is to shape the population preferences to maximize the fraction of the population favouring a predetermined arm. For the case of binary opinions, two types of opinion dynamics are considered -- decreasing elasticity (modeled as a Polya urn with increasing number of balls) and constant elasticity (using the voter model). For the first case, we describe an Explore-then-commit policy and a Thompson sampling policy and analyse the regret for each of these policies. We then show that these algorithms and their analyses carry over to the constant elasticity case. We also describe a Thompson sampling based algorithm for the case when more than two types of opinions are present. Finally, we discuss the case where presence of multiple recommendation systems gives ris
    
[^7]: 基于网络图的分布鲁棒无监督密集检索训练

    Distributionally Robust Unsupervised Dense Retrieval Training on Web Graphs. (arXiv:2310.16605v1 [cs.IR])

    [http://arxiv.org/abs/2310.16605](http://arxiv.org/abs/2310.16605)

    本论文提出了一种无监督的密集检索模型Web-DRO，它利用网络结构进行聚类并重新加权，在无监督场景中显著提高了检索效果。群组分布鲁棒优化方法指导模型对高对比损失的群组分配更多权重，在训练过程中更加关注最坏情况。实验结果表明，结合URL信息的网络图训练能达到最佳的聚类性能。

    

    本文介绍了Web-DRO，一种基于网络结构进行聚类并在对比训练期间重新加权的无监督密集检索模型。具体而言，我们首先利用网络图链接并对锚点-文档对进行对比训练，训练一个嵌入模型用于聚类。然后，我们使用群组分布鲁棒优化方法来重新加权不同的锚点-文档对群组，这指导模型将更多权重分配给对比损失更高的群组，并在训练过程中更加关注最坏情况。在MS MARCO和BEIR上的实验表明，我们的模型Web-DRO在无监督场景中显著提高了检索效果。对聚类技术的比较表明，结合URL信息的网络图训练能达到最佳的聚类性能。进一步分析证实了群组权重的稳定性和有效性，表明了一致的模型偏好以及对有价值文档的有效加权。

    This paper introduces Web-DRO, an unsupervised dense retrieval model, which clusters documents based on web structures and reweights the groups during contrastive training. Specifically, we first leverage web graph links and contrastively train an embedding model for clustering anchor-document pairs. Then we use Group Distributional Robust Optimization to reweight different clusters of anchor-document pairs, which guides the model to assign more weights to the group with higher contrastive loss and pay more attention to the worst case during training. Our experiments on MS MARCO and BEIR show that our model, Web-DRO, significantly improves the retrieval effectiveness in unsupervised scenarios. A comparison of clustering techniques shows that training on the web graph combining URL information reaches optimal performance on clustering. Further analysis confirms that group weights are stable and valid, indicating consistent model preferences as well as effective up-weighting of valuable 
    


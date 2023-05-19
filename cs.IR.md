# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Preference or Intent? Double Disentangled Collaborative Filtering.](http://arxiv.org/abs/2305.11084) | 本文提出了一种双重解缠协同过滤（DDCF）方法，该方法能够对意图和偏好因素进行分离，并通过解缠表示建立独立的稀疏偏好表示，从而提供更准确和可解释性的个性化推荐。 |
| [^2] | [Contrastive State Augmentations for Reinforcement Learning-Based Recommender Systems.](http://arxiv.org/abs/2305.11081) | 本论文提出了对比状态增强方法来训练基于强化学习的推荐系统，其中包括四种状态增强策略和对比状态表示学习，可以解决现有RL推荐方法的问题。实验结果表明该方法胜过其他最先进的RL推荐器。 |
| [^3] | [BERM: Training the Balanced and Extractable Representation for Matching to Improve Generalization Ability of Dense Retrieval.](http://arxiv.org/abs/2305.11052) | BERM使用平衡的、可提取的特征表示法来捕捉匹配信号，从而提高了密集检索的泛化性能。 |
| [^4] | [Improving Recommendation System Serendipity Through Lexicase Selection.](http://arxiv.org/abs/2305.11044) | 本文提出了一种新的机遇性度量方法来衡量推荐系统中回声室和同质化的存在，并采用词典案例选择来改善推荐技术的多样性，取得了在个性化、覆盖面和机遇性方面的优异表现。 |
| [^5] | [Query Performance Prediction: From Ad-hoc to Conversational Search.](http://arxiv.org/abs/2305.10923) | 本文研究了针对从Ad-hoc到交互式搜索中查询性能预测(QPP)的有效方法，并探索了QPP方法在交互式搜索中是否具有推广应用的能力。 |
| [^6] | [Adaptive Graph Contrastive Learning for Recommendation.](http://arxiv.org/abs/2305.10837) | 本文提出了一种自适应图对比学习的推荐框架，通过对比学习的方式改进用户和物品的表示，关注数据中的难以区分的负面例子的信息。 |
| [^7] | [Integrating Item Relevance in Training Loss for Sequential Recommender Systems.](http://arxiv.org/abs/2305.10824) | 本文提出了一种融合项目相关性的新型训练损失函数，用于提高序列推荐系统对噪声的鲁棒性和性能。 |
| [^8] | [When Search Meets Recommendation: Learning Disentangled Search Representation for Recommendation.](http://arxiv.org/abs/2305.10822) | 论文提出了一种基于搜索增强的顺序推荐（SESRec）框架，它利用用户的搜索兴趣进行推荐，通过区分S＆R行为中的相似和不相似表示，使S＆R特征能够更好地发挥其独特的优势。 |
| [^9] | [ReGen: Zero-Shot Text Classification via Training Data Generation with Progressive Dense Retrieval.](http://arxiv.org/abs/2305.10703) | 本文提出了一种基于检索增强的框架，通过渐进式密集检索从通用领域的无标签语料库中创建训练数据，实现了零样本文本分类，相较于最强的基线模型提高了4.3%的性能，与使用大型NLG模型的基线相比节省了约70％的时间。 |
| [^10] | [BioAug: Conditional Generation based Data Augmentation for Low-Resource Biomedical NER.](http://arxiv.org/abs/2305.10647) | 本文提出了一种基于条件生成的数据增强框架BioAug，用于低资源生物医学命名实体识别。BioAug建立在BART上，通过选择性的屏蔽和知识增强进行训练。实验展示了BioAug在5个基准BioNER数据集上的有效性，且表现优于所有基线。 |
| [^11] | [Iteratively Learning Representations for Unseen Entities with Inter-Rule Correlations.](http://arxiv.org/abs/2305.10531) | 本文提出了一种虚拟邻居网络(VNC)，用于解决知识图谱完成中未知实体表示的问题。该方法通过规则挖掘、规则推理和嵌入三个阶段，实现对规则间相关性进行建模。 |
| [^12] | [Large-Scale Text Analysis Using Generative Language Models: A Case Study in Discovering Public Value Expressions in AI Patents.](http://arxiv.org/abs/2305.10383) | 本文研究使用生成语言模型GPT-4进行大规模文本分析，在US AI专利中发现公共价值表达。采用高级布尔查询收集了154,934个专利文档，并与USPTO的完整专利文本合并。得出5.4百万句子的语料库，使用框架以及GPT-4提示进行标记和理性化。评估结果表明，这种方法很准确。 |
| [^13] | [Unconfounded Propensity Estimation for Unbiased Ranking.](http://arxiv.org/abs/2305.09918) | 该论文提出了一种新的算法PropensityNet，用于在强日志记录策略下进行无偏学习排名（ULTR）的倾向性估计，优于现有的最先进ULTR算法。 |
| [^14] | [Invariant Collaborative Filtering to Popularity Distribution Shift.](http://arxiv.org/abs/2302.05328) | 本文提出了不变的协同过滤(InvCF)学习框架，用于解决协同过滤模型易受流行度分布变化影响的问题，这一方法不需要先了解测试集的流行度分布，能够忠实地揭示用户的偏好和流行度语义。 |
| [^15] | [AdaTask: A Task-aware Adaptive Learning Rate Approach to Multi-task Learning.](http://arxiv.org/abs/2211.15055) | 提出了一种名为AdaTask的任务感知自适应学习率方法，通过自适应地调整不同任务的学习率，以平衡不同任务的重要性，从而在各种基准测试上始终优于现有的MTL方法。 |

# 详细

[^1]: 偏好还是意图？双重解缠协同过滤

    Preference or Intent? Double Disentangled Collaborative Filtering. (arXiv:2305.11084v1 [cs.IR])

    [http://arxiv.org/abs/2305.11084](http://arxiv.org/abs/2305.11084)

    本文提出了一种双重解缠协同过滤（DDCF）方法，该方法能够对意图和偏好因素进行分离，并通过解缠表示建立独立的稀疏偏好表示，从而提供更准确和可解释性的个性化推荐。

    

    人们选择物品时通常有不同的意图，而在相同意图下他们的偏好也可能不同。传统的协同过滤方法通常将意图和偏好因素纠缠在建模过程中，这显著限制了推荐性能的稳健性和可解释性。为了解决这一问题，本文提出了一种名为双重解缠协同过滤（DDCF）的个性化推荐方法。一级解缠是为了将意图和偏好的影响因素分开，而第二级解缠是为了构建独立的稀疏偏好表示。实验结果表明，DDCF方法在推荐精度和可解释性方面均优于现有方法。

    People usually have different intents for choosing items, while their preferences under the same intent may also different. In traditional collaborative filtering approaches, both intent and preference factors are usually entangled in the modeling process, which significantly limits the robustness and interpretability of recommendation performances. For example, the low-rating items are always treated as negative feedback while they actually could provide positive information about user intent. To this end, in this paper, we propose a two-fold representation learning approach, namely Double Disentangled Collaborative Filtering (DDCF), for personalized recommendations. The first-level disentanglement is for separating the influence factors of intent and preference, while the second-level disentanglement is performed to build independent sparse preference representations under individual intent with limited computational complexity. Specifically, we employ two variational autoencoder net
    
[^2]: 基于强化学习的推荐系统的对比状态增强

    Contrastive State Augmentations for Reinforcement Learning-Based Recommender Systems. (arXiv:2305.11081v1 [cs.IR])

    [http://arxiv.org/abs/2305.11081](http://arxiv.org/abs/2305.11081)

    本论文提出了对比状态增强方法来训练基于强化学习的推荐系统，其中包括四种状态增强策略和对比状态表示学习，可以解决现有RL推荐方法的问题。实验结果表明该方法胜过其他最先进的RL推荐器。

    

    从历史用户-项目交互序列中学习基于强化学习（RL）的推荐器对于生成高回报建议和改善长期累积效益至关重要。然而，现有的RL推荐方法遇到以下困难：（i）为不包含在离线训练数据中的状态估计价值函数；以及（ii）由于缺乏对比信号，从用户隐式反馈中学习有效的状态表示。在这项工作中，我们提出了对比状态增强（CSA）来训练基于RL的推荐系统。为了解决第一个问题，我们提出了四种状态增强策略来扩大离线数据的状态空间。所提出的方法通过使RL代理访问本地状态区域并确保原始和增强状态之间学习的价值函数相似，提高了推荐器的泛化能力。为了解决第二个问题，我们提出了引入对比状态表示学习，通过最大化正样本相似性和最小化负样本相似性，使代理人学习到信息丰富的状态表示。实验结果表明，我们提出的方法在两个基准数据集上优于几种最先进的RL推荐器。

    Learning reinforcement learning (RL)-based recommenders from historical user-item interaction sequences is vital to generate high-reward recommendations and improve long-term cumulative benefits. However, existing RL recommendation methods encounter difficulties (i) to estimate the value functions for states which are not contained in the offline training data, and (ii) to learn effective state representations from user implicit feedback due to the lack of contrastive signals. In this work, we propose contrastive state augmentations (CSA) for the training of RL-based recommender systems. To tackle the first issue, we propose four state augmentation strategies to enlarge the state space of the offline data. The proposed method improves the generalization capability of the recommender by making the RL agent visit the local state regions and ensuring the learned value functions are similar between the original and augmented states. For the second issue, we propose introducing contrastive 
    
[^3]: BERM：训练平衡可提取表示以提高密集检索的泛化能力

    BERM: Training the Balanced and Extractable Representation for Matching to Improve Generalization Ability of Dense Retrieval. (arXiv:2305.11052v1 [cs.IR])

    [http://arxiv.org/abs/2305.11052](http://arxiv.org/abs/2305.11052)

    BERM使用平衡的、可提取的特征表示法来捕捉匹配信号，从而提高了密集检索的泛化性能。

    

    密集检索已经表现出在域内标记数据集上训练的情况下在第一阶段检索过程中有所作为。然而，以前的研究发现，由于密集检索对域不变和可解释的特征的建模较弱（即两个文本之间的匹配信号，这是信息检索的本质），因此难以推广到未见过的领域。在本文中，我们通过捕捉匹配信号提出了一种提高密集检索泛化性能的新方法，称为BERM。全面的细粒度表达和查询导向的显着性是匹配信号的两个属性。因此，在BERM中，一个单一的Passage被划分为多个单元，提出了两个单元级要求作为约束进行表示以获得有效的匹配信号。一个是语义单元平衡，另一个是必需的匹配单元可提取性。单元级视图和平衡语义使表示以细粒度的方式表达文本。必需的匹配单元可提取性确保保留信息检索的本质。在各种数据集上的实验表明，BERM在保持域内数据集上有竞争力的性能的同时，提高了算法的泛化能力。

    Dense retrieval has shown promise in the first-stage retrieval process when trained on in-domain labeled datasets. However, previous studies have found that dense retrieval is hard to generalize to unseen domains due to its weak modeling of domain-invariant and interpretable feature (i.e., matching signal between two texts, which is the essence of information retrieval). In this paper, we propose a novel method to improve the generalization of dense retrieval via capturing matching signal called BERM. Fully fine-grained expression and query-oriented saliency are two properties of the matching signal. Thus, in BERM, a single passage is segmented into multiple units and two unit-level requirements are proposed for representation as the constraint in training to obtain the effective matching signal. One is semantic unit balance and the other is essential matching unit extractability. Unit-level view and balanced semantics make representation express the text in a fine-grained manner. Esse
    
[^4]: 通过词典案例选择提高推荐系统的机遇性

    Improving Recommendation System Serendipity Through Lexicase Selection. (arXiv:2305.11044v1 [cs.IR])

    [http://arxiv.org/abs/2305.11044](http://arxiv.org/abs/2305.11044)

    本文提出了一种新的机遇性度量方法来衡量推荐系统中回声室和同质化的存在，并采用词典案例选择来改善推荐技术的多样性，取得了在个性化、覆盖面和机遇性方面的优异表现。

    

    推荐系统影响着我们数字生活的方方面面。不幸的是，在努力满足我们需求的过程中，它们限制了我们的开放度。当前的推荐系统促进了回声室和同质化，使用户只看到他们想要看到的信息和与其背景相似的内容。我们提出了一种新的机遇性度量方法，使用聚类分析来衡量推荐系统中回声室和同质化的存在。然后，我们尝试采用从演化计算文献中知名的家长选择算法，即词典案例选择，来改善保留多样性的推荐技术质量。我们的结果表明，词典案例选择或词典案例选择和排名的混合在个性化、覆盖面和我们专门设计的机遇性基准方面优于仅排名的对手，而在准确性（命中率）方面仅稍有不足。

    Recommender systems influence almost every aspect of our digital lives. Unfortunately, in striving to give us what we want, they end up restricting our open-mindedness. Current recommender systems promote echo chambers, where people only see the information they want to see, and homophily, where users of similar background see similar content. We propose a new serendipity metric to measure the presence of echo chambers and homophily in recommendation systems using cluster analysis. We then attempt to improve the diversity-preservation qualities of well known recommendation techniques by adopting a parent selection algorithm from the evolutionary computation literature known as lexicase selection. Our results show that lexicase selection, or a mixture of lexicase selection and ranking, outperforms its purely ranked counterparts in terms of personalization, coverage and our specifically designed serendipity benchmark, while only slightly under-performing in terms of accuracy (hit rate). 
    
[^5]: 查询性能预测：从Ad-hoc到交互式搜索

    Query Performance Prediction: From Ad-hoc to Conversational Search. (arXiv:2305.10923v1 [cs.IR])

    [http://arxiv.org/abs/2305.10923](http://arxiv.org/abs/2305.10923)

    本文研究了针对从Ad-hoc到交互式搜索中查询性能预测(QPP)的有效方法，并探索了QPP方法在交互式搜索中是否具有推广应用的能力。

    

    查询性能预测(QPP)是信息检索中的一个核心任务。QPP的任务是在没有相关判断的情况下预测查询的检索质量。研究表明，QPP在Ad-hoc搜索中非常有效和有用。近年来，对话式搜索(CS)取得了相当大的进展 。有效的QPP能够帮助CS系统在下一轮决定适当的行动。尽管具有潜力，但CS的QPP研究还很少。本文通过重现和研究现有的QPP方法在CS上的有效性来填补这一研究空白。虽然在两种情况下的通道检索任务相同，但CS中的用户查询取决于对话历史，引入了新的QPP挑战。我们尤其是探讨从Ad-hoc搜索中QPP方法的研究结果在三个CS设置中的推广程度:(i) 评估基于查询重写的检索方法的不同查询的检索质量

    Query performance prediction (QPP) is a core task in information retrieval. The QPP task is to predict the retrieval quality of a search system for a query without relevance judgments. Research has shown the effectiveness and usefulness of QPP for ad-hoc search. Recent years have witnessed considerable progress in conversational search (CS). Effective QPP could help a CS system to decide an appropriate action to be taken at the next turn. Despite its potential, QPP for CS has been little studied. We address this research gap by reproducing and studying the effectiveness of existing QPP methods in the context of CS. While the task of passage retrieval remains the same in the two settings, a user query in CS depends on the conversational history, introducing novel QPP challenges. In particular, we seek to explore to what extent findings from QPP methods for ad-hoc search generalize to three CS settings: (i) estimating the retrieval quality of different query rewriting-based retrieval met
    
[^6]: 自适应图对比学习用于推荐系统

    Adaptive Graph Contrastive Learning for Recommendation. (arXiv:2305.10837v1 [cs.IR])

    [http://arxiv.org/abs/2305.10837](http://arxiv.org/abs/2305.10837)

    本文提出了一种自适应图对比学习的推荐框架，通过对比学习的方式改进用户和物品的表示，关注数据中的难以区分的负面例子的信息。

    

    近年来，图神经网络已成功地应用于推荐系统，成为一种有效的协同过滤方法。基于图神经网络的推荐系统的关键思想是沿着用户-物品交互边递归地执行消息传递，以完善编码嵌入，这依赖于充足和高质量的训练数据。由于实际推荐场景中的用户行为数据通常存在噪声并呈现出倾斜分布，一些推荐方法利用自监督学习来改善用户表示，例如SGL和SimGCL。 然而，尽管它们非常有效，但它们通过创建对比视图进行自监督学习，具有数据增强探索，需要进行繁琐的试错选择增强方法。本文提出了一种新的自适应图对比学习（AdaptiveGCL）框架，通过自适应但关注数据中的难以区分的负面例子的信息，用对比学习的方式改进用户和物品的表示。

    Recently, graph neural networks (GNNs) have been successfully applied to recommender systems as an effective collaborative filtering (CF) approach. The key idea of GNN-based recommender system is to recursively perform the message passing along the user-item interaction edge for refining the encoded embeddings, relying on sufficient and high-quality training data. Since user behavior data in practical recommendation scenarios is often noisy and exhibits skewed distribution, some recommendation approaches, e.g., SGL and SimGCL, leverage self-supervised learning to improve user representations against the above issues. Despite their effectiveness, however, they conduct self-supervised learning through creating contrastvie views, depending on the exploration of data augmentations with the problem of tedious trial-and-error selection of augmentation methods. In this paper, we propose a novel Adaptive Graph Contrastive Learning (AdaptiveGCL) framework which conducts graph contrastive learni
    
[^7]: 融合项目相关性的序列推荐系统训练损失函数

    Integrating Item Relevance in Training Loss for Sequential Recommender Systems. (arXiv:2305.10824v1 [cs.IR])

    [http://arxiv.org/abs/2305.10824](http://arxiv.org/abs/2305.10824)

    本文提出了一种融合项目相关性的新型训练损失函数，用于提高序列推荐系统对噪声的鲁棒性和性能。

    

    序列推荐系统是一种受欢迎的推荐系统，它通过学习用户的历史数据来预测用户下一个可能与之交互的项目。然而，用户的交互可能会受到来自帐户共享、不一致的偏好或意外点击等噪声的影响。为了解决这个问题，我们（i）提出了一个考虑多个未来项目的新的评估协议，（ii）引入了一种新的关注相关性的损失函数，用于训练具有多个未来项目的序列推荐系统，以使其对噪声更加鲁棒。我们的关注相关性模型在传统评估协议中提高了NDCG@10约1.2%和HR约0.88%，而在新评估协议中，改进的NDCG@10约1.63%和HR约1.5%。

    Sequential Recommender Systems (SRSs) are a popular type of recommender system that learns from a user's history to predict the next item they are likely to interact with. However, user interactions can be affected by noise stemming from account sharing, inconsistent preferences, or accidental clicks. To address this issue, we (i) propose a new evaluation protocol that takes multiple future items into account and (ii) introduce a novel relevance-aware loss function to train a SRS with multiple future items to make it more robust to noise. Our relevance-aware models obtain an improvement of ~1.2% of NDCG@10 and 0.88% in the traditional evaluation protocol, while in the new evaluation protocol, the improvement is ~1.63% of NDCG@10 and ~1.5% of HR w.r.t the best performing models.
    
[^8]: 当搜索遇见推荐：学习区分搜索表示以用于推荐

    When Search Meets Recommendation: Learning Disentangled Search Representation for Recommendation. (arXiv:2305.10822v1 [cs.IR])

    [http://arxiv.org/abs/2305.10822](http://arxiv.org/abs/2305.10822)

    论文提出了一种基于搜索增强的顺序推荐（SESRec）框架，它利用用户的搜索兴趣进行推荐，通过区分S＆R行为中的相似和不相似表示，使S＆R特征能够更好地发挥其独特的优势。

    

    现代在线服务提供商，如在线购物平台通常提供搜索和推荐（S＆R）服务以满足不同的用户需求。很少有任何有效的手段将来自S＆R服务的用户行为数据结合起来。大多数现有的方法要么仅将S＆R行为单独处理，要么通过聚合两个服务的数据来联合优化它们，忽略了S＆R中用户意图可以有截然不同的事实。在我们的论文中，我们提出了一种基于搜索增强的顺序推荐（SESRec）框架，通过区分S＆R行为中的相似和不相似表示，利用用户的搜索兴趣进行推荐。具体而言，SESRec首先根据用户的查询-项目交互来对齐查询和项目嵌入以计算它们的相似性。然后，使用两个转换器编码器来独立地学习S＆R行为的上下文表示。最后，设计了对比学习任务以学习搜素特征表示和推荐特征表示的相似度距离，使得S＆R特征能够更好地发挥其独特的优势。

    Modern online service providers such as online shopping platforms often provide both search and recommendation (S&R) services to meet different user needs. Rarely has there been any effective means of incorporating user behavior data from both S&R services. Most existing approaches either simply treat S&R behaviors separately, or jointly optimize them by aggregating data from both services, ignoring the fact that user intents in S&R can be distinctively different. In our paper, we propose a Search-Enhanced framework for the Sequential Recommendation (SESRec) that leverages users' search interests for recommendation, by disentangling similar and dissimilar representations within S&R behaviors. Specifically, SESRec first aligns query and item embeddings based on users' query-item interactions for the computations of their similarities. Two transformer encoders are used to learn the contextual representations of S&R behaviors independently. Then a contrastive learning task is designed to 
    
[^9]: ReGen: 通过渐进式密集检索生成训练数据的零样本文本分类方法

    ReGen: Zero-Shot Text Classification via Training Data Generation with Progressive Dense Retrieval. (arXiv:2305.10703v1 [cs.CL])

    [http://arxiv.org/abs/2305.10703](http://arxiv.org/abs/2305.10703)

    本文提出了一种基于检索增强的框架，通过渐进式密集检索从通用领域的无标签语料库中创建训练数据，实现了零样本文本分类，相较于最强的基线模型提高了4.3%的性能，与使用大型NLG模型的基线相比节省了约70％的时间。

    

    随着大型语言模型（LLM）的发展，零样本学习在各种NLP任务中受到了许多关注。与以往使用数十亿级自然语言生成模型生成训练数据的方法不同，我们提出了一种检索增强的框架，从通用领域的无标签语料库中创建训练数据。为实现这一目标，我们首先进行对比预训练，使用类别描述性话语学习了一个无监督的密集检索器以提取最相关的文档。我们进一步提出了两种简单的策略，即展示增强的话语生成和自一致性引导过滤，以提高数据集的主题覆盖率，同时删除噪声样本。对九个数据集的实验表明，REGEN相较于最强的基线模型提高了4.3%的性能，并且与使用大型NLG模型的基线相比节省了约70％的时间。此外，REGEN可以自然地与最近提出的大型语言模型相结合。

    With the development of large language models (LLMs), zero-shot learning has attracted much attention for various NLP tasks. Different from prior works that generate training data with billion-scale natural language generation (NLG) models, we propose a retrieval-enhanced framework to create training data from a general-domain unlabeled corpus. To realize this, we first conduct contrastive pretraining to learn an unsupervised dense retriever for extracting the most relevant documents using class-descriptive verbalizers. We then further propose two simple strategies, namely Verbalizer Augmentation with Demonstrations and Self-consistency Guided Filtering to improve the topic coverage of the dataset while removing noisy examples. Experiments on nine datasets demonstrate that REGEN achieves 4.3% gain over the strongest baselines and saves around 70% of the time compared to baselines using large NLG models. Besides, REGEN can be naturally integrated with recently proposed large language mo
    
[^10]: BioAug：基于条件生成的数据增强方法用于低资源生物医学命名实体识别

    BioAug: Conditional Generation based Data Augmentation for Low-Resource Biomedical NER. (arXiv:2305.10647v1 [cs.CL])

    [http://arxiv.org/abs/2305.10647](http://arxiv.org/abs/2305.10647)

    本文提出了一种基于条件生成的数据增强框架BioAug，用于低资源生物医学命名实体识别。BioAug建立在BART上，通过选择性的屏蔽和知识增强进行训练。实验展示了BioAug在5个基准BioNER数据集上的有效性，且表现优于所有基线。

    

    生物医学命名实体识别(BioNER)是从生物医学文本中识别命名实体的基本任务。由于注释需要高度专业化和专业知识，BioNER 遭受着严重的数据稀缺和缺乏高质量标记数据的困扰。尽管数据增强在低资源命名实体识别方面已经被证明是高效的，但现有的数据增强技术不能为BioNER生成真实且多样化的增强。本文提出了一种新的数据增强框架BioAug，用于低资源BioNER。BioAug建立在BART上，通过选择性的屏蔽和知识增强进行训练，从而解决了一种新的文本重构任务。在训练后，我们进行有条件的生成并在与训练阶段类似的有选择性地损坏文本的条件下生成多样化的增强。我们在5个基准BioNER数据集上展示了BioAug的有效性，并表明BioAug比所有基线都表现更好。

    Biomedical Named Entity Recognition (BioNER) is the fundamental task of identifying named entities from biomedical text. However, BioNER suffers from severe data scarcity and lacks high-quality labeled data due to the highly specialized and expert knowledge required for annotation. Though data augmentation has shown to be highly effective for low-resource NER in general, existing data augmentation techniques fail to produce factual and diverse augmentations for BioNER. In this paper, we present BioAug, a novel data augmentation framework for low-resource BioNER. BioAug, built on BART, is trained to solve a novel text reconstruction task based on selective masking and knowledge augmentation. Post training, we perform conditional generation and generate diverse augmentations conditioning BioAug on selectively corrupted text similar to the training stage. We demonstrate the effectiveness of BioAug on 5 benchmark BioNER datasets and show that BioAug outperforms all our baselines by a signi
    
[^11]: 迭代学习具有规则间相关性的未知实体表示

    Iteratively Learning Representations for Unseen Entities with Inter-Rule Correlations. (arXiv:2305.10531v1 [cs.IR])

    [http://arxiv.org/abs/2305.10531](http://arxiv.org/abs/2305.10531)

    本文提出了一种虚拟邻居网络(VNC)，用于解决知识图谱完成中未知实体表示的问题。该方法通过规则挖掘、规则推理和嵌入三个阶段，实现对规则间相关性进行建模。

    

    知识图谱完成(KGC)的最新研究侧重于学习知识图谱中实体和关系的嵌入。这些嵌入方法要求所有测试实体在训练时被观察到，导致对超出知识图谱（OOKG）实体的耗时重新训练过程。为解决此问题，当前归纳知识嵌入方法采用图神经网络(GNN)通过聚合已知邻居的信息来表示未知实体。他们面临三个重要挑战:i)数据稀疏性，ii)知识图谱中存在复杂模式(如规则间相关性)，iii)规则挖掘、规则推理和嵌入之间存在交互。在本文中，我们提出了一个包含三个阶段的具有规则间相关性的虚拟邻居网络(VNC):i)规则挖掘，ii)规则推理，和iii)嵌入。

    Recent work on knowledge graph completion (KGC) focused on learning embeddings of entities and relations in knowledge graphs. These embedding methods require that all test entities are observed at training time, resulting in a time-consuming retraining process for out-of-knowledge-graph (OOKG) entities. To address this issue, current inductive knowledge embedding methods employ graph neural networks (GNNs) to represent unseen entities by aggregating information of known neighbors. They face three important challenges: (i) data sparsity, (ii) the presence of complex patterns in knowledge graphs (e.g., inter-rule correlations), and (iii) the presence of interactions among rule mining, rule inference, and embedding. In this paper, we propose a virtual neighbor network with inter-rule correlations (VNC) that consists of three stages: (i) rule mining, (ii) rule inference, and (iii) embedding. In the rule mining process, to identify complex patterns in knowledge graphs, both logic rules and 
    
[^12]: 使用生成语言模型进行大规模文本分析：在AI专利中发现公共价值表达的案例研究

    Large-Scale Text Analysis Using Generative Language Models: A Case Study in Discovering Public Value Expressions in AI Patents. (arXiv:2305.10383v1 [cs.CL])

    [http://arxiv.org/abs/2305.10383](http://arxiv.org/abs/2305.10383)

    本文研究使用生成语言模型GPT-4进行大规模文本分析，在US AI专利中发现公共价值表达。采用高级布尔查询收集了154,934个专利文档，并与USPTO的完整专利文本合并。得出5.4百万句子的语料库，使用框架以及GPT-4提示进行标记和理性化。评估结果表明，这种方法很准确。

    

    标记数据对于训练文本分类器至关重要，但对于复杂和抽象的概念而言，准确标记常常很难实现。本文采用一种新颖方法，使用生成语言模型（GPT-4）进行大规模文本分析的标记和理性化。我们将这种方法应用于在美国AI专利中发现公共价值表达的任务上。我们使用在InnovationQ+上提交的高级布尔查询收集了一个包含154,934个专利文档的数据库，这些结果与来自USPTO的完整专利文本合并，总计5.4百万句子。我们设计了一个框架来识别和标记这些AI专利句子中的公共价值表达。我们开发了GPT-4的提示，其中包括文本分类的定义、指导方针、示例和理性化。我们使用BLEU分数和主题建模评估了GPT-4生成的标签和理性化的质量，并发现它们是准确的。

    Labeling data is essential for training text classifiers but is often difficult to accomplish accurately, especially for complex and abstract concepts. Seeking an improved method, this paper employs a novel approach using a generative language model (GPT-4) to produce labels and rationales for large-scale text analysis. We apply this approach to the task of discovering public value expressions in US AI patents. We collect a database comprising 154,934 patent documents using an advanced Boolean query submitted to InnovationQ+. The results are merged with full patent text from the USPTO, resulting in 5.4 million sentences. We design a framework for identifying and labeling public value expressions in these AI patent sentences. A prompt for GPT-4 is developed which includes definitions, guidelines, examples, and rationales for text classification. We evaluate the quality of the labels and rationales produced by GPT-4 using BLEU scores and topic modeling and find that they are accurate, di
    
[^13]: 无偏倾向估计用于无偏排序

    Unconfounded Propensity Estimation for Unbiased Ranking. (arXiv:2305.09918v1 [cs.IR])

    [http://arxiv.org/abs/2305.09918](http://arxiv.org/abs/2305.09918)

    该论文提出了一种新的算法PropensityNet，用于在强日志记录策略下进行无偏学习排名（ULTR）的倾向性估计，优于现有的最先进ULTR算法。

    

    无偏学习排名（ULTR）的目标是利用隐含的用户反馈来优化学习排序系统。在现有解决方案中，自动ULTR算法在实践中因其卓越的性能和低部署成本而受到关注，该算法同时学习用户偏差模型（即倾向性模型）和无偏排名器。尽管该算法在理论上是可靠的，但其有效性通常在弱日志记录策略下进行验证，其中排名模型几乎无法根据与查询相关性来对文档进行排名。然而，当日志记录策略很强时，例如工业部署的排名策略，所报告的有效性无法再现。在本文中，我们首先从因果角度调查ULTR，并揭示一个负面结果：现有的ULTR算法未能解决由查询-文档相关性混淆导致的倾向性高估问题。然后，我们提出了一种基于反门调整的新的学习目标，并提出了一种名为PropensityNet的算法，用于在强日志记录策略下为ULTR估计无偏的倾向性分数。多个数据集的实证结果表明，PropensityNet在强日志记录策略和弱日志记录策略下均优于现有的最先进的ULTR算法。

    The goal of unbiased learning to rank~(ULTR) is to leverage implicit user feedback for optimizing learning-to-rank systems. Among existing solutions, automatic ULTR algorithms that jointly learn user bias models (\ie propensity models) with unbiased rankers have received a lot of attention due to their superior performance and low deployment cost in practice. Despite their theoretical soundness, the effectiveness is usually justified under a weak logging policy, where the ranking model can barely rank documents according to their relevance to the query. However, when the logging policy is strong, e.g., an industry-deployed ranking policy, the reported effectiveness cannot be reproduced. In this paper, we first investigate ULTR from a causal perspective and uncover a negative result: existing ULTR algorithms fail to address the issue of propensity overestimation caused by the query-document relevance confounder. Then, we propose a new learning objective based on backdoor adjustment and 
    
[^14]: 不变的协同过滤对抗流行度分布的变化

    Invariant Collaborative Filtering to Popularity Distribution Shift. (arXiv:2302.05328v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2302.05328](http://arxiv.org/abs/2302.05328)

    本文提出了不变的协同过滤(InvCF)学习框架，用于解决协同过滤模型易受流行度分布变化影响的问题，这一方法不需要先了解测试集的流行度分布，能够忠实地揭示用户的偏好和流行度语义。

    

    尽管协同过滤(collaborative filtering, CF)模型取得了巨大的成功，但由于流行度分布的变化在现实世界中普遍且不可避免，因此这些模型存在严重的性能下降问题。不幸的是，大多数主流的消除流行度偏见的策略需要事先知道测试分布，以确定偏见程度并进一步学习与流行度纠缠在一起的表示来减轻偏见。因此，这些模型在目标测试集中表现出明显的性能提升，但在不知道流行度分布的情况下，却会大大偏离用户真正的兴趣推荐。在本文中，我们提出了一种新的学习框架，不变的协同过滤(InvCF)，用于发现能够忠实地揭示潜在偏好和流行度语义的解耦表示，而不需要对测试分布做出任何假设。

    Collaborative Filtering (CF) models, despite their great success, suffer from severe performance drops due to popularity distribution shifts, where these changes are ubiquitous and inevitable in real-world scenarios. Unfortunately, most leading popularity debiasing strategies, rather than tackling the vulnerability of CF models to varying popularity distributions, require prior knowledge of the test distribution to identify the degree of bias and further learn the popularity-entangled representations to mitigate the bias. Consequently, these models result in significant performance benefits in the target test set, while dramatically deviating the recommendation from users' true interests without knowing the popularity distribution in advance. In this work, we propose a novel learning framework, Invariant Collaborative Filtering (InvCF), to discover disentangled representations that faithfully reveal the latent preference and popularity semantics without making any assumption about the 
    
[^15]: AdaTask: 一种面向多任务学习的任务感知自适应学习率方法

    AdaTask: A Task-aware Adaptive Learning Rate Approach to Multi-task Learning. (arXiv:2211.15055v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.15055](http://arxiv.org/abs/2211.15055)

    提出了一种名为AdaTask的任务感知自适应学习率方法，通过自适应地调整不同任务的学习率，以平衡不同任务的重要性，从而在各种基准测试上始终优于现有的MTL方法。

    

    多任务学习（MTL）模型已在计算机视觉、自然语言处理和推荐系统等领域展现出令人瞩目的结果。尽管已经提出了许多方法，但这些方法如何在每个参数上平衡不同任务仍然不清楚。在本文中，我们提出通过每个任务对该参数进行的总更新来衡量参数的任务优势度。具体而言，我们通过指数衰减的平均更新（AU）来计算每个任务在该参数上的总更新数。基于这一新颖的度量标准，我们观察到现有MTL方法中的许多参数，尤其是在较高的共享层中的参数，仍然受到一个或几个任务的支配。AU的支配主要是由于一个或几个任务的梯度累积导致的。受此启发，我们提出了一种名为AdaTask的任务感知自适应学习率方法，以分离不同任务之间的累积梯度，从而平衡不同任务的重要性。AdaTask根据AU值自适应地调整不同任务的学习率，以平衡不同任务的重要性。我们在各种基准测试上评估了AdaTask，并证明它始终优于现有的MTL方法。

    Multi-task learning (MTL) models have demonstrated impressive results in computer vision, natural language processing, and recommender systems. Even though many approaches have been proposed, how well these approaches balance different tasks on each parameter still remains unclear. In this paper, we propose to measure the task dominance degree of a parameter by the total updates of each task on this parameter. Specifically, we compute the total updates by the exponentially decaying Average of the squared Updates (AU) on a parameter from the corresponding task.Based on this novel metric, we observe that many parameters in existing MTL methods, especially those in the higher shared layers, are still dominated by one or several tasks. The dominance of AU is mainly due to the dominance of accumulative gradients from one or several tasks. Motivated by this, we propose a Task-wise Adaptive learning rate approach, AdaTask in short, to separate the \emph{accumulative gradients} and hence the l
    


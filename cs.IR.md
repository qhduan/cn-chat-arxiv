# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LLM-based Federated Recommendation](https://arxiv.org/abs/2402.09959) | 这项研究介绍了一种基于LLM的联邦推荐系统，用于提高推荐系统的性能和隐私保护。面临的挑战是客户端性能不平衡和对计算资源的高需求。 |
| [^2] | [Empowering recommender systems using automatically generated Knowledge Graphs and Reinforcement Learning.](http://arxiv.org/abs/2307.04996) | 本文介绍了两种基于知识图谱的方法，一种使用强化学习，另一种使用XGBoost算法，用于个性化文章推荐。这些方法利用自动生成的知识图谱，并在一个大型跨国金融服务公司的客户中进行了实证研究。 |

# 详细

[^1]: 基于LLM的联邦推荐系统

    LLM-based Federated Recommendation

    [https://arxiv.org/abs/2402.09959](https://arxiv.org/abs/2402.09959)

    这项研究介绍了一种基于LLM的联邦推荐系统，用于提高推荐系统的性能和隐私保护。面临的挑战是客户端性能不平衡和对计算资源的高需求。

    

    大规模语言模型（LLM）通过微调方法展示了改进推荐系统的巨大潜力，具备先进的上下文理解能力。然而，微调需要用户行为数据，这会带来隐私风险，因为包含了敏感用户信息。这些数据的意外泄露可能侵犯数据保护法，并引发伦理问题。为了减轻这些隐私问题，联邦学习推荐系统（Fed4Rec）被提出作为一种有前景的方法。然而，将Fed4Rec应用于基于LLM的推荐系统面临两个主要挑战：首先，客户端性能不平衡加剧，影响系统的效率；其次，对于本地训练和推理LLM，对客户端的计算和存储资源需求很高。

    arXiv:2402.09959v1 Announce Type: new  Abstract: Large Language Models (LLMs), with their advanced contextual understanding abilities, have demonstrated considerable potential in enhancing recommendation systems via fine-tuning methods. However, fine-tuning requires users' behavior data, which poses considerable privacy risks due to the incorporation of sensitive user information. The unintended disclosure of such data could infringe upon data protection laws and give rise to ethical issues. To mitigate these privacy issues, Federated Learning for Recommendation (Fed4Rec) has emerged as a promising approach. Nevertheless, applying Fed4Rec to LLM-based recommendation presents two main challenges: first, an increase in the imbalance of performance across clients, affecting the system's efficiency over time, and second, a high demand on clients' computational and storage resources for local training and inference of LLMs.   To address these challenges, we introduce a Privacy-Preserving LL
    
[^2]: 利用自动生成的知识图谱和强化学习增强推荐系统

    Empowering recommender systems using automatically generated Knowledge Graphs and Reinforcement Learning. (arXiv:2307.04996v1 [cs.IR])

    [http://arxiv.org/abs/2307.04996](http://arxiv.org/abs/2307.04996)

    本文介绍了两种基于知识图谱的方法，一种使用强化学习，另一种使用XGBoost算法，用于个性化文章推荐。这些方法利用自动生成的知识图谱，并在一个大型跨国金融服务公司的客户中进行了实证研究。

    

    个性化推荐在直接营销中越来越重要，激发了通过知识图谱（KG）应用来提升客户体验的研究动机。例如，在金融服务领域，公司可以通过向客户提供相关金融文章来培养关系，促进客户参与和促进知情的金融决策。尽管一些方法专注于基于KG的推荐系统以改进内容，但在本研究中，我们专注于可解释的基于KG的推荐系统来进行决策。为此，我们提出了两种基于知识图谱的个性化文章推荐方法，用于一家大型跨国金融服务公司的一组客户。第一种方法使用强化学习，第二种方法使用XGBoost算法来向客户推荐文章。这两种方法都利用从结构化（表格数据）和非结构化数据（大量文本数据）生成的KG。

    Personalized recommendations have a growing importance in direct marketing, which motivates research to enhance customer experiences by knowledge graph (KG) applications. For example, in financial services, companies may benefit from providing relevant financial articles to their customers to cultivate relationships, foster client engagement and promote informed financial decisions. While several approaches center on KG-based recommender systems for improved content, in this study we focus on interpretable KG-based recommender systems for decision making.To this end, we present two knowledge graph-based approaches for personalized article recommendations for a set of customers of a large multinational financial services company. The first approach employs Reinforcement Learning and the second approach uses the XGBoost algorithm for recommending articles to the customers. Both approaches make use of a KG generated from both structured (tabular data) and unstructured data (a large body o
    


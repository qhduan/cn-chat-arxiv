# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On (Normalised) Discounted Cumulative Gain as an Offline Evaluation Metric for Top-$n$ Recommendation.](http://arxiv.org/abs/2307.15053) | 本文批判性审视了(Normalised) Discounted Cumulative Gain作为Top-n推荐离线评估指标的方法，并研究了何时可以期望这些指标逼近在线实验的金标准结果。 |
| [^2] | [The Effect of Third Party Implementations on Reproducibility.](http://arxiv.org/abs/2307.14956) | 本研究通过检查第三方实现如何对可重复性产生影响，为推荐系统研究的可重复性问题增加了一个新的角度。研究结果显示，非官方第三方实现可能会对可重复性产生不利影响，并呼吁研究界对这一被忽视的问题予以关注。 |
| [^3] | [Widespread Flaws in Offline Evaluation of Recommender Systems.](http://arxiv.org/abs/2307.14951) | 本文讨论了广泛存在的推荐系统离线评估中的四个缺陷，并说明了研究人员应该避免这些缺陷，以提高离线评估的质量。 |
| [^4] | [Scaling Session-Based Transformer Recommendations using Optimized Negative Sampling and Loss Functions.](http://arxiv.org/abs/2307.14906) | 本文介绍了一种使用优化的负采样和损失函数扩展基于会话的Transformer推荐系统，该系统在大规模电商数据集上通过集成负采样和列表损失函数实现了较高的推荐准确性，并在实践中表现出潜力。 |
| [^5] | [Integrating Offline Reinforcement Learning with Transformers for Sequential Recommendation.](http://arxiv.org/abs/2307.14450) | 本研究将离线强化学习与Transformer相结合，提出了一种顺序推荐方法。通过使用预训练的Transformer模型和离线RL算法训练，我们的方法能够快速稳定地收敛，并在广泛的推荐场景中表现出较好的推荐性能。 |
| [^6] | [Measuring Americanization: A Global Quantitative Study of Interest in American Topics on Wikipedia.](http://arxiv.org/abs/2307.14401) | 本研究使用大量的Wikidata项目和维基百科文章，全球比较分析了维基百科不同语言版本中对美国主题的覆盖率，研究发现美国化在不同地区和文化中的地位不同，并且对美国主题的兴趣普遍存在。 |
| [^7] | [RRAML: Reinforced Retrieval Augmented Machine Learning.](http://arxiv.org/abs/2307.12798) | RRAML是一种新的机器学习框架，将大型语言模型（LLMs）的推理能力与用户提供的庞大数据库中的支持信息相结合。利用强化学习的进展，该方法成功解决了几个关键挑战。 |
| [^8] | [Information Retrieval Meets Large Language Models: A Strategic Report from Chinese IR Community.](http://arxiv.org/abs/2307.09751) | 本论文总结了中国信息检索界关于信息检索与大型语言模型相结合的战略报告。大型语言模型在文本理解、生成和知识推理方面具有出色能力，为信息检索研究开辟了新的方向。此外，IR模型、LLM和人类之间的协同关系形成了一种更强大的信息寻求技术范式。然而，该领域仍面临计算成本、可信度、领域特定限制和伦理考虑等挑战。 |
| [^9] | [Fast and Examination-agnostic Reciprocal Recommendation in Matching Markets.](http://arxiv.org/abs/2306.09060) | 本研究介绍了匹配市场中基于可转移效用的互惠推荐方法，并提出了一种快速且不依赖具体检视方法的算法。 |
| [^10] | [Boosting Big Brother: Attacking Search Engines with Encodings.](http://arxiv.org/abs/2304.14031) | 通过编码方式攻击搜索引擎，以微不可见的方式扭曲文本，攻击者可以控制搜索结果。该攻击成功地影响了Google、Bing和Elasticsearch等多个搜索引擎。此外，还可以将该攻击针对搜索相关的任务如文本摘要和抄袭检测模型。需要提供一套有效的防御措施来应对这些技术带来的潜在威胁。 |
| [^11] | [In-Context Retrieval-Augmented Language Models.](http://arxiv.org/abs/2302.00083) | 本研究提出了一种上下文检索增强的语言模型（In-Context RALM）方法，通过将相关文件作为输入的一部分，无需对语言模型进行进一步的训练即可显著提高语言建模性能和源归因能力，并且相对于现有的RALM方法，它具有更简单的部署过程。 |
| [^12] | [Decoupling Knowledge from Memorization: Retrieval-augmented Prompt Learning.](http://arxiv.org/abs/2205.14704) | 本论文提出了一种检索增强的提示学习方法，通过将知识从记忆中解耦，帮助模型在泛化和记忆之间取得平衡。 |
| [^13] | [Graph-Based Recommendation System Enhanced with Community Detection.](http://arxiv.org/abs/2201.03622) | 本文提出了一个基于图的推荐系统，利用数学和统计方法确定标签的相似性，包括词汇相似性和共现解决方案，并考虑了标签分配的时间，以提高推荐的准确性。 |

# 详细

[^1]: 关于(Normalised) Discounted Cumulative Gain作为Top-n推荐的离线评估指标的论文翻译

    On (Normalised) Discounted Cumulative Gain as an Offline Evaluation Metric for Top-$n$ Recommendation. (arXiv:2307.15053v1 [cs.IR])

    [http://arxiv.org/abs/2307.15053](http://arxiv.org/abs/2307.15053)

    本文批判性审视了(Normalised) Discounted Cumulative Gain作为Top-n推荐离线评估指标的方法，并研究了何时可以期望这些指标逼近在线实验的金标准结果。

    

    推荐方法通常通过两种方式进行评估：(1) 通过(模拟)在线实验，通常被视为金标准，或者(2) 通过一些离线评估程序，目标是近似在线实验的结果。文献中采用了几种离线评估指标，受信息检索领域中常见的排名指标的启发。(Normalised) Discounted Cumulative Gain (nDCG)是其中一种广泛采用的度量标准，在很多年里，更高的(n)DCG值被用来展示新方法在Top-n推荐中的最新进展。我们的工作对这种方法进行了批判性的审视，并研究了我们何时可以期望这些指标逼近在线实验的金标准结果。我们从第一原理上正式提出了DCG被认为是在线奖励的无偏估计的假设，并给出了这个指标的推导。

    Approaches to recommendation are typically evaluated in one of two ways: (1) via a (simulated) online experiment, often seen as the gold standard, or (2) via some offline evaluation procedure, where the goal is to approximate the outcome of an online experiment. Several offline evaluation metrics have been adopted in the literature, inspired by ranking metrics prevalent in the field of Information Retrieval. (Normalised) Discounted Cumulative Gain (nDCG) is one such metric that has seen widespread adoption in empirical studies, and higher (n)DCG values have been used to present new methods as the state-of-the-art in top-$n$ recommendation for many years.  Our work takes a critical look at this approach, and investigates when we can expect such metrics to approximate the gold standard outcome of an online experiment. We formally present the assumptions that are necessary to consider DCG an unbiased estimator of online reward and provide a derivation for this metric from first principles
    
[^2]: 第三方实现对可重复性的影响

    The Effect of Third Party Implementations on Reproducibility. (arXiv:2307.14956v1 [cs.IR])

    [http://arxiv.org/abs/2307.14956](http://arxiv.org/abs/2307.14956)

    本研究通过检查第三方实现如何对可重复性产生影响，为推荐系统研究的可重复性问题增加了一个新的角度。研究结果显示，非官方第三方实现可能会对可重复性产生不利影响，并呼吁研究界对这一被忽视的问题予以关注。

    

    在过去几年中，推荐系统研究的可重复性受到了关注。除了关注使用特定算法重复实验的工作外，研究界还开始讨论评估的各个方面以及它们如何影响可重复性。我们通过研究非官方第三方实现如何对可重复性产生影响，为这一讨论增加了一个新的角度。除了提供一般概述外，我们还彻底检查了六个流行推荐算法的第三方实现，并将它们与五个公开数据集上的官方版本进行了比较。根据我们令人震惊的发现，我们希望引起研究界对这个被忽视的可重复性方面的关注。

    Reproducibility of recommender systems research has come under scrutiny during recent years. Along with works focusing on repeating experiments with certain algorithms, the research community has also started discussing various aspects of evaluation and how these affect reproducibility. We add a novel angle to this discussion by examining how unofficial third-party implementations could benefit or hinder reproducibility. Besides giving a general overview, we thoroughly examine six third-party implementations of a popular recommender algorithm and compare them to the official version on five public datasets. In the light of our alarming findings we aim to draw the attention of the research community to this neglected aspect of reproducibility.
    
[^3]: 离线评估中推荐系统存在广泛缺陷

    Widespread Flaws in Offline Evaluation of Recommender Systems. (arXiv:2307.14951v1 [cs.IR])

    [http://arxiv.org/abs/2307.14951](http://arxiv.org/abs/2307.14951)

    本文讨论了广泛存在的推荐系统离线评估中的四个缺陷，并说明了研究人员应该避免这些缺陷，以提高离线评估的质量。

    

    尽管离线评估只是在线性能的不完美代理，因为推荐系统的交互性质，但由于生产推荐系统的专有性质阻止了A/B测试设置的独立验证和在线结果的验证，离线评估可能仍然是可见的未来推荐系统研究中的主要评估方式。因此，离线评估设置必须尽可能真实和无瑕疵。不幸的是，由于后期作品复制了前辈们存在缺陷的评估设置而不质疑其有效性，所以在推荐系统研究中评估缺陷相当普遍。为了改善推荐系统的离线评估质量，我们讨论了其中四个广泛存在的缺陷以及研究人员应该避免它们的原因。

    Even though offline evaluation is just an imperfect proxy of online performance -- due to the interactive nature of recommenders -- it will probably remain the primary way of evaluation in recommender systems research for the foreseeable future, since the proprietary nature of production recommenders prevents independent validation of A/B test setups and verification of online results. Therefore, it is imperative that offline evaluation setups are as realistic and as flawless as they can be. Unfortunately, evaluation flaws are quite common in recommender systems research nowadays, due to later works copying flawed evaluation setups from their predecessors without questioning their validity. In the hope of improving the quality of offline evaluation of recommender systems, we discuss four of these widespread flaws and why researchers should avoid them.
    
[^4]: 使用优化的负采样和损失函数扩展基于会话的Transformer推荐系统

    Scaling Session-Based Transformer Recommendations using Optimized Negative Sampling and Loss Functions. (arXiv:2307.14906v1 [cs.IR])

    [http://arxiv.org/abs/2307.14906](http://arxiv.org/abs/2307.14906)

    本文介绍了一种使用优化的负采样和损失函数扩展基于会话的Transformer推荐系统，该系统在大规模电商数据集上通过集成负采样和列表损失函数实现了较高的推荐准确性，并在实践中表现出潜力。

    

    本文介绍了TRON，一种使用优化的负采样的可扩展的基于会话的Transformer推荐系统。受到SASRec和GRU4Rec+等现有模型在可扩展性和性能方面的限制，TRON集成了top-k负采样和列表损失函数，以提高其推荐准确性。在相关的大规模电子商务数据集上的评估结果表明，TRON在保持与SASRec类似的训练速度的同时，改进了当前方法的推荐质量。一项实时的A/B测试显示，相对于SASRec，TRON的点击率增加了18.14%，突显了其在实际环境中的潜力。

    This work introduces TRON, a scalable session-based Transformer Recommender using Optimized Negative-sampling. Motivated by the scalability and performance limitations of prevailing models such as SASRec and GRU4Rec+, TRON integrates top-k negative sampling and listwise loss functions to enhance its recommendation accuracy. Evaluations on relevant large-scale e-commerce datasets show that TRON improves upon the recommendation quality of current methods while maintaining training speeds similar to SASRec. A live A/B test yielded an 18.14% increase in click-through rate over SASRec, highlighting the potential of TRON in practical settings. For further research, we provide access to our source code at https://github.com/otto-de/TRON and an anonymized dataset at https://github.com/otto-de/recsys-dataset.
    
[^5]: 将离线强化学习与Transformer相结合的顺序推荐方法

    Integrating Offline Reinforcement Learning with Transformers for Sequential Recommendation. (arXiv:2307.14450v1 [cs.IR])

    [http://arxiv.org/abs/2307.14450](http://arxiv.org/abs/2307.14450)

    本研究将离线强化学习与Transformer相结合，提出了一种顺序推荐方法。通过使用预训练的Transformer模型和离线RL算法训练，我们的方法能够快速稳定地收敛，并在广泛的推荐场景中表现出较好的推荐性能。

    

    我们考虑了顺序推荐的问题，即根据过去的互动进行当前推荐。这个推荐任务需要对顺序数据进行高效处理，并旨在提供最大化长期奖励的推荐。为此，我们通过使用离线RL算法和我们模型架构中的策略网络进行训练，该模型架构从预训练的Transformer模型初始化。预训练模型利用Transformer处理顺序信息的优秀能力。与依赖在线交互的模拟方法相比，我们专注于实现一种完全离线的RL框架，能够快速稳定地收敛。通过对公共数据集进行大量实验，我们证明了我们的方法在各种推荐场景中（包括电子商务和电影推荐）的稳健性。与最先进的监督学习算法相比，我们的算法产生了更好的推荐性能。

    We consider the problem of sequential recommendation, where the current recommendation is made based on past interactions. This recommendation task requires efficient processing of the sequential data and aims to provide recommendations that maximize the long-term reward. To this end, we train a farsighted recommender by using an offline RL algorithm with the policy network in our model architecture that has been initialized from a pre-trained transformer model. The pre-trained model leverages the superb ability of the transformer to process sequential information. Compared to prior works that rely on online interaction via simulation, we focus on implementing a fully offline RL framework that is able to converge in a fast and stable way. Through extensive experiments on public datasets, we show that our method is robust across various recommendation regimes, including e-commerce and movie suggestions. Compared to state-of-the-art supervised learning algorithms, our algorithm yields re
    
[^6]: 测量美国化：关于维基百科对美国主题的全球定量研究

    Measuring Americanization: A Global Quantitative Study of Interest in American Topics on Wikipedia. (arXiv:2307.14401v1 [cs.IR])

    [http://arxiv.org/abs/2307.14401](http://arxiv.org/abs/2307.14401)

    本研究使用大量的Wikidata项目和维基百科文章，全球比较分析了维基百科不同语言版本中对美国主题的覆盖率，研究发现美国化在不同地区和文化中的地位不同，并且对美国主题的兴趣普遍存在。

    

    我们使用9千万个Wikidata项目和5800万个维基百科文章，对58种语言版本的维基百科中关于美国主题的覆盖率进行了全球比较分析。我们的研究旨在调查美国化在不同地区和文化中是否更加占主导地位，并确定对美国主题的兴趣是否普遍存在。

    We conducted a global comparative analysis of the coverage of American topics in different language versions of Wikipedia, using over 90 million Wikidata items and 40 million Wikipedia articles in 58 languages. Our study aimed to investigate whether Americanization is more or less dominant in different regions and cultures and to determine whether interest in American topics is universal.
    
[^7]: RRAML: 强化检索增强的机器学习

    RRAML: Reinforced Retrieval Augmented Machine Learning. (arXiv:2307.12798v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2307.12798](http://arxiv.org/abs/2307.12798)

    RRAML是一种新的机器学习框架，将大型语言模型（LLMs）的推理能力与用户提供的庞大数据库中的支持信息相结合。利用强化学习的进展，该方法成功解决了几个关键挑战。

    

    大型语言模型（LLMs）的出现彻底改变了机器学习和相关领域，在理解、生成和操作人类语言方面展示了显著的能力。然而，通过基于API的文本提示提交来使用它们会存在一定的限制，包括上下文约束和外部资源的可用性。为了解决这些挑战，我们提出了一种新的框架，称为强化检索增强的机器学习（RRAML）。RRAML将LLMs的推理能力与由专用检索器从用户提供的庞大数据库中检索到的支持信息相结合。通过利用强化学习的最新进展，我们的方法有效地解决了几个关键挑战。首先，它绕过了访问LLM梯度的需求。其次，我们的方法减轻了针对特定任务重新训练LLMs的负担，因为由于对模型和合作的访问受限，这往往是不可行或不可能的。

    The emergence of large language models (LLMs) has revolutionized machine learning and related fields, showcasing remarkable abilities in comprehending, generating, and manipulating human language. However, their conventional usage through API-based text prompt submissions imposes certain limitations in terms of context constraints and external source availability. To address these challenges, we propose a novel framework called Reinforced Retrieval Augmented Machine Learning (RRAML). RRAML integrates the reasoning capabilities of LLMs with supporting information retrieved by a purpose-built retriever from a vast user-provided database. By leveraging recent advancements in reinforcement learning, our method effectively addresses several critical challenges. Firstly, it circumvents the need for accessing LLM gradients. Secondly, our method alleviates the burden of retraining LLMs for specific tasks, as it is often impractical or impossible due to restricted access to the model and the co
    
[^8]: 信息检索遇上大型语言模型：中国信息检索界的战略报告

    Information Retrieval Meets Large Language Models: A Strategic Report from Chinese IR Community. (arXiv:2307.09751v1 [cs.IR])

    [http://arxiv.org/abs/2307.09751](http://arxiv.org/abs/2307.09751)

    本论文总结了中国信息检索界关于信息检索与大型语言模型相结合的战略报告。大型语言模型在文本理解、生成和知识推理方面具有出色能力，为信息检索研究开辟了新的方向。此外，IR模型、LLM和人类之间的协同关系形成了一种更强大的信息寻求技术范式。然而，该领域仍面临计算成本、可信度、领域特定限制和伦理考虑等挑战。

    

    信息检索（IR）领域已经取得了显著的发展，超越了传统搜索，以满足多样化的用户信息需求。最近，大型语言模型（LLM）在文本理解、生成和知识推理方面展示了出色的能力，为IR研究开辟了新的契机。LLM不仅能够促进生成式检索，还提供了改进的用户理解、模型评估和用户系统交互方案。更重要的是，IR模型、LLM和人类之间的协同关系构成了一种更强大的信息寻求技术范式。IR模型提供实时和相关的信息，LLM贡献内部知识，而人类在信息服务的可靠性方面起着需求者和评估者的中心作用。然而，仍然存在着一些重要挑战，包括计算成本、可信度问题、领域特定限制和伦理考虑。

    The research field of Information Retrieval (IR) has evolved significantly, expanding beyond traditional search to meet diverse user information needs. Recently, Large Language Models (LLMs) have demonstrated exceptional capabilities in text understanding, generation, and knowledge inference, opening up exciting avenues for IR research. LLMs not only facilitate generative retrieval but also offer improved solutions for user understanding, model evaluation, and user-system interactions. More importantly, the synergistic relationship among IR models, LLMs, and humans forms a new technical paradigm that is more powerful for information seeking. IR models provide real-time and relevant information, LLMs contribute internal knowledge, and humans play a central role of demanders and evaluators to the reliability of information services. Nevertheless, significant challenges exist, including computational costs, credibility concerns, domain-specific limitations, and ethical considerations. To 
    
[^9]: 在匹配市场中快速且不依赖具体检视方法的互惠推荐算法

    Fast and Examination-agnostic Reciprocal Recommendation in Matching Markets. (arXiv:2306.09060v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2306.09060](http://arxiv.org/abs/2306.09060)

    本研究介绍了匹配市场中基于可转移效用的互惠推荐方法，并提出了一种快速且不依赖具体检视方法的算法。

    

    在职位发布和在线约会平台等匹配市场中，推荐系统在平台的成功中起着关键作用。与向用户推荐物品的标准推荐系统不同，互惠推荐系统需要考虑用户之间的共同兴趣。此外，确保推荐机会不过分偏向热门用户对于匹配数量和用户公平性都至关重要。然而，现有的匹配市场推荐方法在大规模实际应用平台上面临计算挑战，并且依赖于基于职位的模型中特定的检视函数。在本文中，我们介绍了基于可转移效用的匹配模型的互惠推荐方法，并提出了一种更快速且不依赖具体检视方法的算法。此外，我们对我们的方法进行了评估。

    In matching markets such as job posting and online dating platforms, the recommender system plays a critical role in the success of the platform. Unlike standard recommender systems that suggest items to users, reciprocal recommender systems (RRSs) that suggest other users must take into account the mutual interests of users. In addition, ensuring that recommendation opportunities do not disproportionately favor popular users is essential for the total number of matches and for fairness among users. Existing recommendation methods in matching markets, however, face computational challenges on real-world scale platforms and depend on specific examination functions in the position-based model (PBM). In this paper, we introduce the reciprocal recommendation method based on the matching with transferable utility (TU matching) model in the context of ranking recommendations in matching markets, and propose a faster and examination-agnostic algorithm. Furthermore, we evaluate our approach on
    
[^10]: 提升老大哥：采用编码方式攻击搜索引擎

    Boosting Big Brother: Attacking Search Engines with Encodings. (arXiv:2304.14031v1 [cs.CR])

    [http://arxiv.org/abs/2304.14031](http://arxiv.org/abs/2304.14031)

    通过编码方式攻击搜索引擎，以微不可见的方式扭曲文本，攻击者可以控制搜索结果。该攻击成功地影响了Google、Bing和Elasticsearch等多个搜索引擎。此外，还可以将该攻击针对搜索相关的任务如文本摘要和抄袭检测模型。需要提供一套有效的防御措施来应对这些技术带来的潜在威胁。

    

    搜索引擎对于文本编码操纵的索引和搜索存在漏洞。通过以不常见的编码表示形式微不可见地扭曲文本，攻击者可以控制特定搜索查询在多个搜索引擎上的结果。我们演示了这种攻击成功地针对了两个主要的商业搜索引擎——Google和Bing——以及一个开源搜索引擎——Elasticsearch。我们进一步展示了这种攻击成功地针对了包括Bing的GPT-4聊天机器人和Google的Bard聊天机器人在内的LLM聊天搜索。我们还提出了一种变体攻击，针对与搜索密切相关的两个ML任务——文本摘要和抄袭检测模型。我们提供了一套针对这些技术的防御措施，并警告攻击者可以利用这些攻击启动反信息争夺战。这促使搜索引擎维护人员修补已部署的系统。

    Search engines are vulnerable to attacks against indexing and searching via text encoding manipulation. By imperceptibly perturbing text using uncommon encoded representations, adversaries can control results across search engines for specific search queries. We demonstrate that this attack is successful against two major commercial search engines - Google and Bing - and one open source search engine - Elasticsearch. We further demonstrate that this attack is successful against LLM chat search including Bing's GPT-4 chatbot and Google's Bard chatbot. We also present a variant of the attack targeting text summarization and plagiarism detection models, two ML tasks closely tied to search. We provide a set of defenses against these techniques and warn that adversaries can leverage these attacks to launch disinformation campaigns against unsuspecting users, motivating the need for search engine maintainers to patch deployed systems.
    
[^11]: 上下文检索增强的语言模型

    In-Context Retrieval-Augmented Language Models. (arXiv:2302.00083v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.00083](http://arxiv.org/abs/2302.00083)

    本研究提出了一种上下文检索增强的语言模型（In-Context RALM）方法，通过将相关文件作为输入的一部分，无需对语言模型进行进一步的训练即可显著提高语言建模性能和源归因能力，并且相对于现有的RALM方法，它具有更简单的部署过程。

    

    检索增强的语言模型(RALM)方法在生成过程中，通过将相关文件从语料库中检索出来与语言模型(LM)进行协同，已被证明可以显著提高语言建模性能。此外，它们还可以缓解事实不准确的文本生成问题，并提供自然的源归因机制。现有的RALM方法着重于修改LM架构以便于整合外部信息，从而大大增加了部署的复杂性。本文提出了一种简单的替代方法，称为上下文RALM：保持LM架构不变，并在输入中添加检索到的文件，无需对LM进行任何进一步的训练。我们展示了基于现成的通用检索器的上下文RALM在模型大小和不同语料库中能够提供出人意料的大幅度的LM增益。我们还证明，文件检索和排名机制可以针对RALM设置进行专门优化。

    Retrieval-Augmented Language Modeling (RALM) methods, which condition a language model (LM) on relevant documents from a grounding corpus during generation, were shown to significantly improve language modeling performance. In addition, they can mitigate the problem of factually inaccurate text generation and provide natural source attribution mechanism. Existing RALM approaches focus on modifying the LM architecture in order to facilitate the incorporation of external information, significantly complicating deployment. This paper considers a simple alternative, which we dub In-Context RALM: leaving the LM architecture unchanged and prepending grounding documents to the input, without any further training of the LM. We show that In-Context RALM that builds on off-the-shelf general purpose retrievers provides surprisingly large LM gains across model sizes and diverse corpora. We also demonstrate that the document retrieval and ranking mechanism can be specialized to the RALM setting to 
    
[^12]: 将知识从记忆中解耦：检索增强的提示学习

    Decoupling Knowledge from Memorization: Retrieval-augmented Prompt Learning. (arXiv:2205.14704v4 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2205.14704](http://arxiv.org/abs/2205.14704)

    本论文提出了一种检索增强的提示学习方法，通过将知识从记忆中解耦，帮助模型在泛化和记忆之间取得平衡。

    

    提示学习方法在自然语言处理领域取得了显著的突破，提高了少样本学习的性能，但仍然遵循参数化学习范式；在学习过程中，遗忘和机械记忆问题可能导致不稳定的泛化问题。为了缓解这些限制，我们开发了RetroPrompt，旨在从记忆中将知识解耦，帮助模型在泛化和记忆之间取得平衡。与传统的提示学习方法相比，RetroPrompt从训练实例构建了一个开放式知识库，并在输入、训练和推断过程中实施检索机制，使模型具备了从训练语料库中检索相关上下文用于增强的能力。大量实验证明了RetroPrompt的效果。

    Prompt learning approaches have made waves in natural language processing by inducing better few-shot performance while they still follow a parametric-based learning paradigm; the oblivion and rote memorization problems in learning may encounter unstable generalization issues. Specifically, vanilla prompt learning may struggle to utilize atypical instances by rote during fully-supervised training or overfit shallow patterns with low-shot data. To alleviate such limitations, we develop RetroPrompt with the motivation of decoupling knowledge from memorization to help the model strike a balance between generalization and memorization. In contrast with vanilla prompt learning, RetroPrompt constructs an open-book knowledge-store from training instances and implements a retrieval mechanism during the process of input, training and inference, thus equipping the model with the ability to retrieve related contexts from the training corpus as cues for enhancement. Extensive experiments demonstra
    
[^13]: 基于图的推荐系统在社区检测中的增强

    Graph-Based Recommendation System Enhanced with Community Detection. (arXiv:2201.03622v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2201.03622](http://arxiv.org/abs/2201.03622)

    本文提出了一个基于图的推荐系统，利用数学和统计方法确定标签的相似性，包括词汇相似性和共现解决方案，并考虑了标签分配的时间，以提高推荐的准确性。

    

    许多研究者已经利用标签信息来改善推荐系统中推荐技术的性能。通过研究用户的标签，可以了解他们的兴趣，从而提高推荐的准确性。然而，由于用户自定义标签的任意性和缺乏限制，确定其确切含义和标签之间的相似性存在问题。本文利用数学和统计方法确定标签的词汇相似性和共现解决方案，以分配语义相似性。另外，考虑到用户兴趣随时间变化，本文还在共现标签中考虑了标签分配的时间以确定标签的相似性。然后，基于标签的相似性创建图形模型来建模用户的兴趣。

    Many researchers have used tag information to improve the performance of recommendation techniques in recommender systems. Examining the tags of users will help to get their interests and leads to more accuracy in the recommendations. Since user-defined tags are chosen freely and without any restrictions, problems arise in determining their exact meaning and the similarity of tags. However, using thesaurus and ontologies to find the meaning of tags is not very efficient due to their free definition by users and the use of different languages in many data sets. Therefore, this article uses mathematical and statistical methods to determine lexical similarity and co-occurrence tags solution to assign semantic similarity. On the other hand, due to the change of users' interests over time this article has considered the time of tag assignments in co-occurrence tags for determining similarity of tags. Then the graph is created based on similarity of tags. For modeling the interests of the us
    


# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Logical Discrete Graphical Models Must Supplement Large Language Models for Information Synthesis](https://arxiv.org/abs/2403.09599) | 逻辑离散图形模型被认为是大型语言模型无法解决的问题的解决方案，能够解决幻觉、复杂推理、不确定性下的规划和复杂计算等问题。 |
| [^2] | [More than words: Advancements and challenges in speech recognition for singing](https://arxiv.org/abs/2403.09298) | 本文讨论了歌唱语音识别中的挑战和进展，探索了音素识别、歌曲中的语言识别、关键词识别和完整歌词转录等关键领域，并介绍了深度学习和大规模数据集在该领域推动进展的最新发展。 |
| [^3] | [Seed-based information retrieval in networks of research publications: Evaluation of direct citations, bibliographic coupling, co-citations and PubMed related article score](https://arxiv.org/abs/2403.09295) | 论文比较了基于种子的研究出版物网络信息检索中的直接引用、文献耦合、共同引用和PubMed相关文章得分等不同方法的表现，发现组合引文方法优于仅使用共同引用。 |
| [^4] | [Online and Offline Evaluation in Search Clarification](https://arxiv.org/abs/2403.09180) | 研究调查搜索澄清中在线和离线评估之间的不一致情况，以用户参与度作为真实情况，探讨离线排名列表如何与基于在线用户参与度的理想排名列表相似。 |
| [^5] | [Evaluating LLMs for Gender Disparities in Notable Persons](https://arxiv.org/abs/2403.09148) | 评估大型语言模型在显著人物中存在的性别差距，并发现GPT-4在性能上有所改进，但问题尚未完全解决 |
| [^6] | [USimAgent: Large Language Models for Simulating Search Users](https://arxiv.org/abs/2403.09142) | 该论文介绍了一种基于大型语言模型的用户搜索行为模拟器 USimAgent，可以模拟用户在搜索过程中的查询、点击和停止行为，实现生成特定搜索的完整搜索会话。 |
| [^7] | [Projected Gradient Descent for Spectral Compressed Sensing via Symmetric Hankel Factorization](https://arxiv.org/abs/2403.09031) | 提出了一种新的投影梯度下降方法（SHGD），通过对称因子分解进行谱压缩感知，减少了计算和存储成本，引入了新的因子分解歧义。 |
| [^8] | [Domain Adaptation for Dense Retrieval and Conversational Dense Retrieval through Self-Supervision by Meticulous Pseudo-Relevance Labeling](https://arxiv.org/abs/2403.08970) | 本文提出了一种结合查询生成和自我监督方法的域自适应策略，通过在目标域上自动生成伪相关性标签，并应用于密集检索和对话密集检索模型，以实现更好的推广能力。 |
| [^9] | [PAPERCLIP: Associating Astronomical Observations and Natural Language with Multi-Modal Models](https://arxiv.org/abs/2403.08851) | 该研究提出了PAPERCLIP方法，通过将天文观测与自然语言关联起来，利用预训练的CLIP模型进行微调，实现了观测和自然语言之间的有意义的联合表示。 |
| [^10] | [AcademiaOS: Automating Grounded Theory Development in Qualitative Research with Large Language Models](https://arxiv.org/abs/2403.08844) | AcademiaOS利用大型语言模型自动化定性研究中的理论建构，为学术界提供新颖的见解。 |
| [^11] | [End-to-end Graph-Sequential Representation Learning for Accurate Recommendations](https://arxiv.org/abs/2403.00895) | 本文提出了一个新颖的多重表示学习框架，有效地结合了基于序列和基于图的推荐方法，显著改善了推荐性能。 |
| [^12] | [Science Checker Reloaded: A Bidirectional Paradigm for Transparency and Logical Reasoning](https://arxiv.org/abs/2402.13897) | 提出了一个两块式的方法来解决长文档中信息检索领域的挑战，并实现了双向交互 |
| [^13] | [Utilizing Contextual Clues and Role Correlations for Enhancing Document-level Event Argument Extraction](https://arxiv.org/abs/2310.05116) | 本文提出了CARLG模型，通过利用上下文线索和角色相关性，提升了文档级事件论证提取的性能。 |
| [^14] | [Making Language Models Better Tool Learners with Execution Feedback](https://arxiv.org/abs/2305.13068) | 这篇论文提出了一个名为TRICE的框架，通过执行反馈实现语言模型的工具学习，使其能够学会何时以及如何有效地使用工具。 |
| [^15] | [A Survey on Modern Recommendation System based on Big Data.](http://arxiv.org/abs/2206.02631) | 这份综述全面调研了基于大数据的现代推荐系统的发展和挑战，总结了四种主要类型的推荐技术，并指出了未来研究的潜在领域。 |

# 详细

[^1]: 逻辑离散图形模型必须为信息综合补充大语言模型

    Logical Discrete Graphical Models Must Supplement Large Language Models for Information Synthesis

    [https://arxiv.org/abs/2403.09599](https://arxiv.org/abs/2403.09599)

    逻辑离散图形模型被认为是大型语言模型无法解决的问题的解决方案，能够解决幻觉、复杂推理、不确定性下的规划和复杂计算等问题。

    

    鉴于大型语言模型的新兴推理能力，信息检索变得更加复杂。现代信息检索系统宣称他们可以基于潜在的多个不同文档、冲突的数据来源和推理来综合生成答案，而不仅仅是检索文档。我们审查了最近的文献，并认为大型语言模型有关键缺陷，使其不可能独立构成通用智能，或回答一般信息综合请求。这项审查显示大语言模型存在以下问题：幻觉、复杂推理、不确定性下的规划和复杂计算。我们概述了逻辑离散图形模型如何解决所有这些问题，并概述了从无标签文本训练逻辑离散模型的方法。

    arXiv:2403.09599v1 Announce Type: new  Abstract: Given the emergent reasoning abilities of large language models, information retrieval is becoming more complex. Rather than just retrieve a document, modern information retrieval systems advertise that they can synthesize an answer based on potentially many different documents, conflicting data sources, and using reasoning. We review recent literature and argue that the large language model has crucial flaws that prevent it from on its own ever constituting general intelligence, or answering general information synthesis requests. This review shows that the following are problems for large language models: hallucinations, complex reasoning, planning under uncertainty, and complex calculations. We outline how logical discrete graphical models can solve all of these problems, and outline a method of training a logical discrete model from unlabeled text.
    
[^2]: 超越言语：歌唱语音识别中的进展与挑战

    More than words: Advancements and challenges in speech recognition for singing

    [https://arxiv.org/abs/2403.09298](https://arxiv.org/abs/2403.09298)

    本文讨论了歌唱语音识别中的挑战和进展，探索了音素识别、歌曲中的语言识别、关键词识别和完整歌词转录等关键领域，并介绍了深度学习和大规模数据集在该领域推动进展的最新发展。

    

    本文讨论了歌唱语音识别中的挑战和进展，这是一个与标准语音识别完全不同的领域。歌唱包含独特的挑战，包括广泛的音高变化、多样化的声乐风格以及背景音乐干扰。我们探讨了诸如音素识别、歌曲中的语言识别、关键词识别和完整歌词转录等关键领域。我将描述一些我在这些任务上进行研究时的经历，就在它们开始崭露头角的时候，但也会展示深度学习和大规模数据集的最新进展如何推动了这一领域的进步。我的目标是阐明将语音识别应用于歌唱时的复杂性，评估当前的能力，并概述未来的研究方向。

    arXiv:2403.09298v1 Announce Type: cross  Abstract: This paper addresses the challenges and advancements in speech recognition for singing, a domain distinctly different from standard speech recognition. Singing encompasses unique challenges, including extensive pitch variations, diverse vocal styles, and background music interference. We explore key areas such as phoneme recognition, language identification in songs, keyword spotting, and full lyrics transcription. I will describe some of my own experiences when performing research on these tasks just as they were starting to gain traction, but will also show how recent developments in deep learning and large-scale datasets have propelled progress in this field. My goal is to illuminate the complexities of applying speech recognition to singing, evaluate current capabilities, and outline future research directions.
    
[^3]: 基于种子的研究出版物网络信息检索：对直接引用、文献耦合、共同引用和PubMed相关文章得分的评估

    Seed-based information retrieval in networks of research publications: Evaluation of direct citations, bibliographic coupling, co-citations and PubMed related article score

    [https://arxiv.org/abs/2403.09295](https://arxiv.org/abs/2403.09295)

    论文比较了基于种子的研究出版物网络信息检索中的直接引用、文献耦合、共同引用和PubMed相关文章得分等不同方法的表现，发现组合引文方法优于仅使用共同引用。

    

    在这篇论文中，我们探讨了基于种子的研究出版物网络信息检索。使用系统评审作为基准，结合NIH开放引文收集的出版数据，我们比较了三种基于引文的方法——直接引用、共同引用和文献耦合在召回率和精确率方面的表现。此外，我们还将PubMed相关文章得分以及组合方法纳入比较。我们还对先前使用引文关系进行信息检索的早期研究进行了相当全面的回顾。结果显示共同引用优于文献耦合和直接引用。然而，在研究中，将这三种方法组合起来胜过仅使用共同引用。结果进一步表明，与先前研究一致，将基于引文的方法与文本方法相结合

    arXiv:2403.09295v1 Announce Type: new  Abstract: In this contribution, we deal with seed-based information retrieval in networks of research publications. Using systematic reviews as a baseline, and publication data from the NIH Open Citation Collection, we compare the performance of the three citation-based approaches direct citation, co-citation, and bibliographic coupling with respect to recall and precision measures. In addition, we include the PubMed Related Article score as well as combined approaches in the comparison. We also provide a fairly comprehensive review of earlier research in which citation relations have been used for information retrieval purposes. The results show an advantage for co-citation over bibliographic coupling and direct citation. However, combining the three approaches outperforms the exclusive use of co-citation in the study. The results further indicate, in line with previous research, that combining citation-based approaches with textual approaches en
    
[^4]: 搜索澄清中的在线和离线评估

    Online and Offline Evaluation in Search Clarification

    [https://arxiv.org/abs/2403.09180](https://arxiv.org/abs/2403.09180)

    研究调查搜索澄清中在线和离线评估之间的不一致情况，以用户参与度作为真实情况，探讨离线排名列表如何与基于在线用户参与度的理想排名列表相似。

    

    目前，在搜索系统中，澄清问题模型在吸引用户方面的有效性受到限制，对其整体有用性产生了怀疑。要改善这些模型的性能，关键是采用既包括来自用户的实时反馈（在线评估），又通过人工评估评估澄清问题特征的评估方法。然而，在信息检索领域，关于在线和离线评估之间的关系存在争议。本研究旨在调查这种不一致在搜索澄清中的持续情况。我们以用户参与度作为基本事实，并使用多个离线标签来调查离线排名的澄清问题在多大程度上类似于基于在线用户参与度的理想排名列表。

    arXiv:2403.09180v1 Announce Type: new  Abstract: The effectiveness of clarification question models in engaging users within search systems is currently constrained, casting doubt on their overall usefulness. To improve the performance of these models, it is crucial to employ assessment approaches that encompass both real-time feedback from users (online evaluation) and the characteristics of clarification questions evaluated through human assessment (offline evaluation). However, the relationship between online and offline evaluations has been debated in information retrieval. This study aims to investigate how this discordance holds in search clarification. We use user engagement as ground truth and employ several offline labels to investigate to what extent the offline ranked lists of clarification resemble the ideal ranked lists based on online user engagement.
    
[^5]: 评估用于显著人物的LLMs中的性别差距

    Evaluating LLMs for Gender Disparities in Notable Persons

    [https://arxiv.org/abs/2403.09148](https://arxiv.org/abs/2403.09148)

    评估大型语言模型在显著人物中存在的性别差距，并发现GPT-4在性能上有所改进，但问题尚未完全解决

    

    本研究探讨了大型语言模型（LLMs）用于检索事实信息的使用，解决了它们产生事实不准确的“幻觉”回复或完全拒绝甚至回答提示的倾向。具体来说，它调查了LLMs对事实查询的回应中存在的基于性别的偏见。这篇论文通过评估GPT模型在召回、幻觉和拒绝等多个维度上的公平性来采用多管齐下的方法。我们的研究发现GPT-3.5生成的回应中存在明显的性别差距。虽然GPT-4的进展提升了性能，但在回应被拒绝的情况下，这些性别差距并未完全消除。研究进一步探讨了这些差距的起源，通过检查提示中的性别关联和回应中的同质性。

    arXiv:2403.09148v1 Announce Type: new  Abstract: This study examines the use of Large Language Models (LLMs) for retrieving factual information, addressing concerns over their propensity to produce factually incorrect "hallucinated" responses or to altogether decline to even answer prompt at all. Specifically, it investigates the presence of gender-based biases in LLMs' responses to factual inquiries. This paper takes a multi-pronged approach to evaluating GPT models by evaluating fairness across multiple dimensions of recall, hallucinations and declinations. Our findings reveal discernible gender disparities in the responses generated by GPT-3.5. While advancements in GPT-4 have led to improvements in performance, they have not fully eradicated these gender disparities, notably in instances where responses are declined. The study further explores the origins of these disparities by examining the influence of gender associations in prompts and the homogeneity in the responses.
    
[^6]: USimAgent：用于模拟搜索用户的大型语言模型

    USimAgent: Large Language Models for Simulating Search Users

    [https://arxiv.org/abs/2403.09142](https://arxiv.org/abs/2403.09142)

    该论文介绍了一种基于大型语言模型的用户搜索行为模拟器 USimAgent，可以模拟用户在搜索过程中的查询、点击和停止行为，实现生成特定搜索的完整搜索会话。

    

    由于成本效益和可重现性方面的优势，用户模拟已成为信息检索系统用户为中心评估的有前途的解决方案。然而，准确模拟用户搜索行为一直是一项挑战，因为用户在搜索中的行为非常复杂，受到学习、推理和规划等错综复杂认知过程的驱动。最近，大型语言模型（LLMs）已经展示出在模拟人类级智能方面的潜力，并已被用于构建各种任务的自主代理。然而，尚未充分探索使用LLM模拟搜索行为的潜力。在本文中，我们介绍了一种基于LLM的用户搜索行为模拟器，即USimAgent。提出的模拟器可以模拟用户在搜索过程中的查询、点击和停止行为，因此能够生成特定搜索的完整搜索会话。

    arXiv:2403.09142v1 Announce Type: cross  Abstract: Due to the advantages in the cost-efficiency and reproducibility, user simulation has become a promising solution to the user-centric evaluation of information retrieval systems. Nonetheless, accurately simulating user search behaviors has long been a challenge, because users' actions in search are highly complex and driven by intricate cognitive processes such as learning, reasoning, and planning. Recently, Large Language Models (LLMs) have demonstrated remarked potential in simulating human-level intelligence and have been used in building autonomous agents for various tasks. However, the potential of using LLMs in simulating search behaviors has not yet been fully explored. In this paper, we introduce a LLM-based user search behavior simulator, USimAgent. The proposed simulator can simulate users' querying, clicking, and stopping behaviors during search, and thus, is capable of generating complete search sessions for specific search
    
[^7]: 基于对称 Hankel 因子分解的谱压缩感知的投影梯度下降

    Projected Gradient Descent for Spectral Compressed Sensing via Symmetric Hankel Factorization

    [https://arxiv.org/abs/2403.09031](https://arxiv.org/abs/2403.09031)

    提出了一种新的投影梯度下降方法（SHGD），通过对称因子分解进行谱压缩感知，减少了计算和存储成本，引入了新的因子分解歧义。

    

    当前谱压缩感知方法通过 Hankel 矩阵完成采用对称因子分解来展示 Hankel 矩阵的低秩性质。然而，先前的非凸梯度方法只利用不对称因子分解来实现谱压缩感知。在本文中，我们提出了一种新颖的投影梯度下降方法，通过对称因子分解进行谱压缩感知，名为对称 Hankel 投影梯度下降（SHGD），它仅更新一个矩阵并避免了平衡正则化项。与基于不对称因子分解的先前梯度方法相比，SHGD减少了大约一半的计算和存储成本。此外，我们工作中使用的对称因子分解与先前的低秩分解模型完全不同，引入了在复正交变换下的新因子分解歧义。我们为我们的分解设计了新颖的距离度量。

    arXiv:2403.09031v1 Announce Type: new  Abstract: Current spectral compressed sensing methods via Hankel matrix completion employ symmetric factorization to demonstrate the low-rank property of the Hankel matrix. However, previous non-convex gradient methods only utilize asymmetric factorization to achieve spectral compressed sensing. In this paper, we propose a novel nonconvex projected gradient descent method for spectral compressed sensing via symmetric factorization named Symmetric Hankel Projected Gradient Descent (SHGD), which updates only one matrix and avoids a balancing regularization term. SHGD reduces about half of the computation and storage costs compared to the prior gradient method based on asymmetric factorization. {Besides, the symmetric factorization employed in our work is completely novel to the prior low-rank factorization model, introducing a new factorization ambiguity under complex orthogonal transformation}. Novel distance metrics are designed for our factorizat
    
[^8]: 通过细致的伪相关性标记实现自我监督的密集检索和对话密集检索的域自适应

    Domain Adaptation for Dense Retrieval and Conversational Dense Retrieval through Self-Supervision by Meticulous Pseudo-Relevance Labeling

    [https://arxiv.org/abs/2403.08970](https://arxiv.org/abs/2403.08970)

    本文提出了一种结合查询生成和自我监督方法的域自适应策略，通过在目标域上自动生成伪相关性标签，并应用于密集检索和对话密集检索模型，以实现更好的推广能力。

    

    最近的研究表明，密集检索模型在将不同分布的目标域推广方面的能力是有限的，这与基于交互的模型的结果形成了对比。以前为了减轻这一挑战而尝试的方法涉及利用对抗性学习和查询生成方法，但这两种方法仍然带来了有限的改进。在本文中，我们提出将查询生成方法与自我监督方法相结合，其中在目标域上自动生成伪相关性标签。为了实现这一点，我们利用了一个T5-3B模型进行伪正标记，并选择了细致的硬负例。我们还将这种策略应用于用于对话搜索的对话密集检索模型。使用类似的伪标记方法，但增加了一个查询重写模块来重写对话查询以便后续标记。

    arXiv:2403.08970v1 Announce Type: new  Abstract: Recent studies have demonstrated that the ability of dense retrieval models to generalize to target domains with different distributions is limited, which contrasts with the results obtained with interaction-based models. Prior attempts to mitigate this challenge involved leveraging adversarial learning and query generation approaches, but both approaches nevertheless resulted in limited improvements. In this paper, we propose to combine the query-generation approach with a self-supervision approach in which pseudo-relevance labels are automatically generated on the target domain. To accomplish this, a T5-3B model is utilized for pseudo-positive labeling, and meticulous hard negatives are chosen. We also apply this strategy on conversational dense retrieval model for conversational search. A similar pseudo-labeling approach is used, but with the addition of a query-rewriting module to rewrite conversational queries for subsequent labelin
    
[^9]: PAPERCLIP：使用多模态模型将天文观测和自然语言关联起来

    PAPERCLIP: Associating Astronomical Observations and Natural Language with Multi-Modal Models

    [https://arxiv.org/abs/2403.08851](https://arxiv.org/abs/2403.08851)

    该研究提出了PAPERCLIP方法，通过将天文观测与自然语言关联起来，利用预训练的CLIP模型进行微调，实现了观测和自然语言之间的有意义的联合表示。

    

    我们提出了PAPERCLIP（Proposal Abstracts Provide an Effective Representation for Contrastive Language-Image Pre-training），一种使用神经网络模型将由望远镜成像的天文观测与自然语言关联起来的方法。该模型是从经过预训练的对比语言-图像预训练（CLIP）模型微调而来，使用成功的观测提案摘要和相应的下游观测，其中摘要可选择通过使用大型语言模型（LLMs）进行引导生成来进行总结。以哈勃空间望远镜（HST）的观测为例，我们展示了微调的模型通过针对图像检索（即使用自然语言查询找到最相关的观测）和描述检索（即查询与天文物体类别和用例最相关的内容）的测试，体现了观测和自然语言之间的有意义的联合表示。

    arXiv:2403.08851v1 Announce Type: cross  Abstract: We present PAPERCLIP (Proposal Abstracts Provide an Effective Representation for Contrastive Language-Image Pre-training), a method which associates astronomical observations imaged by telescopes with natural language using a neural network model. The model is fine-tuned from a pre-trained Contrastive Language-Image Pre-training (CLIP) model using successful observing proposal abstracts and corresponding downstream observations, with the abstracts optionally summarized via guided generation using large language models (LLMs). Using observations from the Hubble Space Telescope (HST) as an example, we show that the fine-tuned model embodies a meaningful joint representation between observations and natural language through tests targeting image retrieval (i.e., finding the most relevant observations using natural language queries) and description retrieval (i.e., querying for astrophysical object classes and use cases most relevant to a 
    
[^10]: AcademiaOS：利用大型语言模型自动化定性研究中的理论建构

    AcademiaOS: Automating Grounded Theory Development in Qualitative Research with Large Language Models

    [https://arxiv.org/abs/2403.08844](https://arxiv.org/abs/2403.08844)

    AcademiaOS利用大型语言模型自动化定性研究中的理论建构，为学术界提供新颖的见解。

    

    AcademiaOS是第一个尝试利用大型语言模型自动化定性研究中的理论建构的系统。利用最近大型语言模型的语言理解、生成和推理能力，AcademiaOS对筛选过的定性原始数据（如访谈文本）进行编码，并发展主题和维度以进一步构建一个理论模型，提供新颖的见解。一项用户研究（n=19）表明，该系统在学术界中得到认可，并具有增强人类进行定性研究的潜力。AcademiaOS已经开源供他人进行构建并适应其用例。

    arXiv:2403.08844v1 Announce Type: cross  Abstract: AcademiaOS is a first attempt to automate grounded theory development in qualitative research with large language models. Using recent large language models' language understanding, generation, and reasoning capabilities, AcademiaOS codes curated qualitative raw data such as interview transcripts and develops themes and dimensions to further develop a grounded theoretical model, affording novel insights. A user study (n=19) suggests that the system finds acceptance in the academic community and exhibits the potential to augment humans in qualitative research. AcademiaOS has been made open-source for others to build upon and adapt to their use cases.
    
[^11]: 精确推荐的端到端图-序列表示学习

    End-to-end Graph-Sequential Representation Learning for Accurate Recommendations

    [https://arxiv.org/abs/2403.00895](https://arxiv.org/abs/2403.00895)

    本文提出了一个新颖的多重表示学习框架，有效地结合了基于序列和基于图的推荐方法，显著改善了推荐性能。

    

    近年来推荐系统的许多新进展集中在开发基于序列和基于图的方法上。这两种方法在建模行为数据中的复杂关系方面都证明了其有效性，从而在个性化排名和下一个推荐任务中取得了有益的成果，同时保持了良好的可扩展性。然而，它们从数据中捕捉到的信号截然不同。前者直接通过与最近物品的有序交互来表示用户，而后者旨在捕捉交互图中的间接依赖关系。本文提出了一个新颖的多重表示学习框架，利用这两种范式之间的协同作用。我们在几个数据集上的实证评估表明，利用所提出的框架相互训练序列和图组件显著改善了推荐性能。

    arXiv:2403.00895v1 Announce Type: cross  Abstract: Many recent advancements in recommender systems have focused on developing sequence-based and graph-based approaches. Both approaches proved useful in modeling intricate relationships within behavioral data, leading to promising outcomes in personalized ranking and next-item recommendation tasks while maintaining good scalability. However, they capture very different signals from data. While the former approach represents users directly through ordered interactions with recent items, the latter one aims to capture indirect dependencies across the interactions graph. This paper presents a novel multi-representational learning framework that exploits the synergies between these two paradigms. Our empirical evaluation on several datasets demonstrates that mutual training of sequential and graph components with the proposed framework significantly improves recommendations performance.
    
[^12]: 科学检查者再度升级：透明度和逻辑推理的双向范式

    Science Checker Reloaded: A Bidirectional Paradigm for Transparency and Logical Reasoning

    [https://arxiv.org/abs/2402.13897](https://arxiv.org/abs/2402.13897)

    提出了一个两块式的方法来解决长文档中信息检索领域的挑战，并实现了双向交互

    

    信息检索是一个快速发展的领域。然而，它仍然面临着在科学和工业的海量信息中的诸多限制，比如语义分歧和检索中的词汇差距、语义搜索中的低精度和缺乏可解释性，或者生成模型中的幻觉和过时信息。在本文中，我们提出了一个两块式的方法来解决长文档的这些障碍。第一个模块通过查询扩展增强了在稀疏检索中的语言理解，以检索相关文档。第二个模块通过只使用长文档中传播的信息，为复杂问题提供全面和信息丰富的答案来加深结果，实现双向交互。在管道的各个阶段，向用户呈现中间结果以促进对系统推理的理解。我们相信这种双向方法带来了

    arXiv:2402.13897v1 Announce Type: cross  Abstract: Information retrieval is a rapidly evolving field. However it still faces significant limitations in the scientific and industrial vast amounts of information, such as semantic divergence and vocabulary gaps in sparse retrieval, low precision and lack of interpretability in semantic search, or hallucination and outdated information in generative models. In this paper, we introduce a two-block approach to tackle these hurdles for long documents. The first block enhances language understanding in sparse retrieval by query expansion to retrieve relevant documents. The second block deepens the result by providing comprehensive and informative answers to the complex question using only the information spread in the long document, enabling bidirectional engagement. At various stages of the pipeline, intermediate results are presented to users to facilitate understanding of the system's reasoning. We believe this bidirectional approach brings
    
[^13]: 利用上下文线索和角色相关性提升文档级事件论证提取

    Utilizing Contextual Clues and Role Correlations for Enhancing Document-level Event Argument Extraction

    [https://arxiv.org/abs/2310.05116](https://arxiv.org/abs/2310.05116)

    本文提出了CARLG模型，通过利用上下文线索和角色相关性，提升了文档级事件论证提取的性能。

    

    文档级事件论证提取（EAE）是信息提取中至关重要但具有挑战性的子任务之一。现有方法大多关注论证和事件触发器之间的交互，忽视了两个关键点：上下文线索的信息和论证角色之间的语义相关性。本文提出了CARLG模型，包括两个模块：上下文线索聚合（CCA）和基于角色的潜在信息引导（RLIG），通过有效利用上下文线索和角色相关性来提高文档级EAE。CCA模块通过利用来自预训练编码器的上下文注意权重，自适应地捕捉和整合上下文线索。RLIG模块通过角色交互编码捕捉语义相关性，并通过潜在角色表示提供宝贵的信息引导。值得注意的是，我们的CCA和RLIG模块紧凑、可移植且高效，引入的新参数不超过1%，且易于实现。

    Document-level event argument extraction (EAE) is a vital but challenging subtask in information extraction. Most existing approaches focus on the interaction between arguments and event triggers, ignoring two critical points: the information of contextual clues and the semantic correlations among argument roles. In this paper, we propose the CARLG model, which consists of two modules: Contextual Clues Aggregation (CCA) and Role-based Latent Information Guidance (RLIG), effectively leveraging contextual clues and role correlations for improving document-level EAE. The CCA module adaptively captures and integrates contextual clues by utilizing context attention weights from a pre-trained encoder. The RLIG module captures semantic correlations through role-interactive encoding and provides valuable information guidance with latent role representation. Notably, our CCA and RLIG modules are compact, transplantable and efficient, which introduce no more than 1% new parameters and can be eas
    
[^14]: 通过执行反馈使语言模型成为更好的工具学习者

    Making Language Models Better Tool Learners with Execution Feedback

    [https://arxiv.org/abs/2305.13068](https://arxiv.org/abs/2305.13068)

    这篇论文提出了一个名为TRICE的框架，通过执行反馈实现语言模型的工具学习，使其能够学会何时以及如何有效地使用工具。

    

    工具作为关键的界面，使人类能够理解和改变环境。随着基础模型的出现，AI系统可以利用工具扩展其能力并与真实世界互动。现有的工具学习方法包括监督微调和提示工程方法，通常使大型语言模型不加选择地利用工具，因为复杂任务往往超出了它们自身的能力。然而，为简单任务引入工具（模型本身可以轻松解决的任务），可能会无意间传播错误而不是提高性能。因此，研究问题是：我们能否教会语言模型何时以及如何使用工具？为满足这个需求，我们提出了Tool leaRning wIth exeCution fEedback (TRICE)，这是一个两阶段的端到端框架，使模型能够通过从工具执行中得到的反馈不断学习，从而学会何时以及如何有效地使用工具。

    Tools serve as pivotal interfaces that enable humans to understand and reshape the environment. With the advent of foundation models, AI systems can utilize tools to expand their capabilities and interact with the real world. Existing tool learning methodologies, encompassing supervised fine-tuning and prompt engineering approaches, often induce large language models to utilize tools indiscriminately, as complex tasks often exceed their own competencies. However, introducing tools for simple tasks, which the models themselves can readily resolve, can inadvertently propagate errors rather than enhance performance. This leads to the research question: can we teach language models when and how to use tools? To meet this need, we propose Tool leaRning wIth exeCution fEedback (TRICE), a two-stage end-to-end framework that enables the model to continually learn through feedback derived from tool execution, thereby learning when and how to use tools effectively. Experimental results, backed b
    
[^15]: 基于大数据的现代推荐系统综述

    A Survey on Modern Recommendation System based on Big Data. (arXiv:2206.02631v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2206.02631](http://arxiv.org/abs/2206.02631)

    这份综述全面调研了基于大数据的现代推荐系统的发展和挑战，总结了四种主要类型的推荐技术，并指出了未来研究的潜在领域。

    

    本综述全面探索了推荐系统的发展和当前状态，这些系统已广泛整合到各种网络应用中。它重点关注个性化推荐策略在在线产品或服务中的进展。我们将推荐技术分为四种主要类型：基于内容的、协同过滤的、基于知识的和混合的，每种类型都解决了独特的情景。本综述详细审视了推荐系统的历史背景和最新的创新方法，特别是那些使用大数据的方法。此外，本综述还确定并讨论了现代推荐系统面临的关键挑战，如数据稀疏性、可扩展性问题以及对推荐的多样性需求。综述最后强调了这些挑战作为未来研究的潜在领域。

    This survey provides an exhaustive exploration of the evolution and current state of recommendation systems, which have seen widespread integration in various web applications. It focuses on the advancement of personalized recommendation strategies for online products or services. We categorize recommendation techniques into four primary types: content-based, collaborative filtering-based, knowledge-based, and hybrid-based, each addressing unique scenarios. The survey offers a detailed examination of the historical context and the latest innovative approaches in recommendation systems, particularly those employing big data. Additionally, it identifies and discusses key challenges faced by modern recommendation systems, such as data sparsity, scalability issues, and the need for diversity in recommendations. The survey concludes by highlighting these challenges as potential areas for fruitful future research in the field.
    


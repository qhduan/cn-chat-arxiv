# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NeuralMind-UNICAMP at 2022 TREC NeuCLIR: Large Boring Rerankers for Cross-lingual Retrieval.](http://arxiv.org/abs/2303.16145) | 本研究发现尽管mT5模型仅在相同语言的查询-文档对上进行微调，但在不同语言的查询-文档对存在的情况下也是可行的。研究结果表明，在所有任务和语言上都表现出色，获得了很高的获胜位置，强调了其作为一种跨语言检索的可行解决方案的潜力。 |
| [^2] | [Causal Disentangled Recommendation Against User Preference Shifts.](http://arxiv.org/abs/2303.16068) | 本文提出因果分离推荐系统解决用户偏好变化问题，通过抽象因果图发现未观察到的因素的变化导致偏好移位，并关注精细偏好影响与不同项目的交互。 |
| [^3] | [A comment to "A General Theory of IR Evaluation Measures".](http://arxiv.org/abs/2303.16061) | 本文是一篇对于《关于IR评估方法的一般理论》的评论，指出了它的结论存在一些限制。 |
| [^4] | [Item Graph Convolution Collaborative Filtering for Inductive Recommendations.](http://arxiv.org/abs/2303.15946) | 该论文提出了一种基于物品图卷积的归纳式协同过滤推荐算法，通过加权投影构建物品-物品图，并采用卷积将高阶关联注入物品嵌入，同时将用户表示形成加权的加权和。 |
| [^5] | [A Multi-Granularity Matching Attention Network for Query Intent Classification in E-commerce Retrieval.](http://arxiv.org/abs/2303.15870) | 本文提出了一种名为 MMAN 的多粒度匹配注意力网络，可以全面提取查询和查询类别交互矩阵的特征，从而消除查询和类别之间表达差异的差距，用于查询意图分类。 |
| [^6] | [Genetic Analysis of Prostate Cancer with Computer Science Methods.](http://arxiv.org/abs/2303.15851) | 本文应用数据科学、机器学习和拓扑网络分析方法对不同转移部位的前列腺癌肿瘤进行了基因分析，筛选出了与前列腺癌转移相关的13个基因，准确率达到了92%。 |
| [^7] | [Multi-Behavior Recommendation with Cascading Graph Convolution Networks.](http://arxiv.org/abs/2303.15720) | 本文提出了一种基于级联图卷积网络的多行为推荐模型，能够明确地利用行为链中的依赖关系，以缓解推荐系统数据稀疏或冷启动问题。 |
| [^8] | [Model Cascades for Efficient Image Search.](http://arxiv.org/abs/2303.15595) | 该论文提出了一种新颖的图像排名算法，使用级联的神经编码器来逐步过滤图像，从而减少了3倍以上的TIR生命周期成本。 |
| [^9] | [GETT-QA: Graph Embedding based T2T Transformer for Knowledge Graph Question Answering.](http://arxiv.org/abs/2303.13284) | 本论文提出了GETT-QA系统，该系统使用T5对自然语言问题生成简化的SPARQL查询，并使用截断的KG嵌入提高了知识图谱问答的性能。 |
| [^10] | [Optimizing generalized Gini indices for fairness in rankings.](http://arxiv.org/abs/2204.06521) | 本文探讨了使用广义基尼福利函数（GGF）作为规范性准则来指定推荐系统应优化的方法，以此实现排名公平性。 |

# 详细

[^1]: 《神经网络-巴西坎普斯大学》在2022年TREC NeuCLIR中的大型无聊重排器实现跨语言检索

    NeuralMind-UNICAMP at 2022 TREC NeuCLIR: Large Boring Rerankers for Cross-lingual Retrieval. (arXiv:2303.16145v1 [cs.IR])

    [http://arxiv.org/abs/2303.16145](http://arxiv.org/abs/2303.16145)

    本研究发现尽管mT5模型仅在相同语言的查询-文档对上进行微调，但在不同语言的查询-文档对存在的情况下也是可行的。研究结果表明，在所有任务和语言上都表现出色，获得了很高的获胜位置，强调了其作为一种跨语言检索的可行解决方案的潜力。

    

    本文报道了使用mT5-XXL重排器在TREC 2022 NeuCLIR赛道上进行跨语言信息检索（CLIR）的研究。该研究最大的贡献也许是发现尽管mT5模型仅在相同语言的查询-文档对上进行微调，但在不同语言的查询-文档对存在的情况下，它证明了在第一阶段检索表现亚优的情况下是可行的。研究结果表明，在所有任务和语言上都表现出色，获得了很高的获胜位置。最后，本研究为在CLIR任务中使用mT5提供了有价值的见解，并强调了其作为一种可行解决方案的潜力。如需复制，请参阅https://github.com/unicamp-dl/NeuCLIR22-mT5。

    This paper reports on a study of cross-lingual information retrieval (CLIR) using the mT5-XXL reranker on the NeuCLIR track of TREC 2022. Perhaps the biggest contribution of this study is the finding that despite the mT5 model being fine-tuned only on query-document pairs of the same language it proved to be viable for CLIR tasks, where query-document pairs are in different languages, even in the presence of suboptimal first-stage retrieval performance. The results of the study show outstanding performance across all tasks and languages, leading to a high number of winning positions. Finally, this study provides valuable insights into the use of mT5 in CLIR tasks and highlights its potential as a viable solution. For reproduction refer to https://github.com/unicamp-dl/NeuCLIR22-mT5
    
[^2]: 因果分离推荐：应对用户偏好变化的问题

    Causal Disentangled Recommendation Against User Preference Shifts. (arXiv:2303.16068v1 [cs.IR])

    [http://arxiv.org/abs/2303.16068](http://arxiv.org/abs/2303.16068)

    本文提出因果分离推荐系统解决用户偏好变化问题，通过抽象因果图发现未观察到的因素的变化导致偏好移位，并关注精细偏好影响与不同项目的交互。

    

    推荐系统很容易面临用户偏好移位的问题。如果用户偏好随着时间的推移而发生了变化，用户表示将变得过时，从而导致不适当的推荐。为了解决这个问题，现有的工作专注于学习稳健的表示或预测变化模式，缺乏发现用户偏好移位的潜在原因的全面视角。为了理解偏好移位，我们抽象了一个因果图，描述了用户交互序列的生成过程。假设用户偏好在短时间内是稳定的，我们将交互序列抽象为一组时间顺序的环境。从因果图中，我们发现一些未观察到的因素的变化（例如怀孕）导致了环境之间的偏好移位。此外，用户对不同类别的精细偏好稀疏地影响与不同项目的交互。受到因果图的启示，我们关注处理偏好移位问题的关键考虑。

    Recommender systems easily face the issue of user preference shifts. User representations will become out-of-date and lead to inappropriate recommendations if user preference has shifted over time. To solve the issue, existing work focuses on learning robust representations or predicting the shifting pattern. There lacks a comprehensive view to discover the underlying reasons for user preference shifts. To understand the preference shift, we abstract a causal graph to describe the generation procedure of user interaction sequences. Assuming user preference is stable within a short period, we abstract the interaction sequence as a set of chronological environments. From the causal graph, we find that the changes of some unobserved factors (e.g., becoming pregnant) cause preference shifts between environments. Besides, the fine-grained user preference over categories sparsely affects the interactions with different items. Inspired by the causal graph, our key considerations to handle pre
    
[^3]: 一篇《关于IR评估方法的一般理论》的评论

    A comment to "A General Theory of IR Evaluation Measures". (arXiv:2303.16061v1 [cs.IR])

    [http://arxiv.org/abs/2303.16061](http://arxiv.org/abs/2303.16061)

    本文是一篇对于《关于IR评估方法的一般理论》的评论，指出了它的结论存在一些限制。

    

    本文“一般的IR评估方法理论”开发了一个形式化的框架，以确定IR评估方法是否为区间刻度。本评论显示了一些关于其结论的限制。

    The paper "A General Theory of IR Evaluation Measures" develops a formal framework to determine whether IR evaluation measures are interval scales. This comment shows some limitations about its conclusions.
    
[^4]: 基于物品图卷积的归纳式协同过滤推荐算法

    Item Graph Convolution Collaborative Filtering for Inductive Recommendations. (arXiv:2303.15946v1 [cs.IR])

    [http://arxiv.org/abs/2303.15946](http://arxiv.org/abs/2303.15946)

    该论文提出了一种基于物品图卷积的归纳式协同过滤推荐算法，通过加权投影构建物品-物品图，并采用卷积将高阶关联注入物品嵌入，同时将用户表示形成加权的加权和。

    

    最近，GCN被用作推荐系统算法的核心组件，将用户-项目交互作为二分图的边解释。然而，在缺乏附加信息的情况下，大多数现有模型采用随机初始化用户嵌入并在训练过程中优化它们的方法。这种策略使得这些算法本质上是转换型的，从而限制了它们为训练时未见过的用户生成预测的能力。为了解决这个问题，我们提出了一种基于卷积的算法，从用户的角度是归纳式的，同时仅依赖于隐式用户-项目交互数据。我们提出通过二分图交互网络的加权投影构建物品-物品图并采用卷积将高阶关联注入物品嵌入，同时将用户表示形成加权的加权和。

    Graph Convolutional Networks (GCN) have been recently employed as core component in the construction of recommender system algorithms, interpreting user-item interactions as the edges of a bipartite graph. However, in the absence of side information, the majority of existing models adopt an approach of randomly initialising the user embeddings and optimising them throughout the training process. This strategy makes these algorithms inherently transductive, curtailing their ability to generate predictions for users that were unseen at training time. To address this issue, we propose a convolution-based algorithm, which is inductive from the user perspective, while at the same time, depending only on implicit user-item interaction data. We propose the construction of an item-item graph through a weighted projection of the bipartite interaction network and to employ convolution to inject higher order associations into item embeddings, while constructing user representations as weighted su
    
[^5]: 电商检索中用于查询意图分类的多粒度匹配注意力网络

    A Multi-Granularity Matching Attention Network for Query Intent Classification in E-commerce Retrieval. (arXiv:2303.15870v1 [cs.IR])

    [http://arxiv.org/abs/2303.15870](http://arxiv.org/abs/2303.15870)

    本文提出了一种名为 MMAN 的多粒度匹配注意力网络，可以全面提取查询和查询类别交互矩阵的特征，从而消除查询和类别之间表达差异的差距，用于查询意图分类。

    

    查询意图分类旨在协助客户找到所需产品，已成为电子商务搜索的重要组成部分。现有的查询意图分类模型要么设计更精细的模型以增强查询的表示学习，要么探索标签图和多任务以帮助模型学习外部信息。然而，这些模型无法从查询和类别中捕捉多粒度匹配特征，这使得它们难以弥补非正式查询和类别之间表达差异的差距。本文提出了一种多粒度匹配注意力网络(MMAN)，其包含三个模块：自匹配模块、字符级匹配模块和语义级匹配模块，以全面提取查询和查询类别交互矩阵的特征。通过这种方式，该模型可以消除查询意图分类中查询和类别之间表达差异的差距。

    Query intent classification, which aims at assisting customers to find desired products, has become an essential component of the e-commerce search. Existing query intent classification models either design more exquisite models to enhance the representation learning of queries or explore label-graph and multi-task to facilitate models to learn external information. However, these models cannot capture multi-granularity matching features from queries and categories, which makes them hard to mitigate the gap in the expression between informal queries and categories.  This paper proposes a Multi-granularity Matching Attention Network (MMAN), which contains three modules: a self-matching module, a char-level matching module, and a semantic-level matching module to comprehensively extract features from the query and a query-category interaction matrix. In this way, the model can eliminate the difference in expression between queries and categories for query intent classification. We conduc
    
[^6]: 计算机科学方法在前列腺癌遗传学中的应用

    Genetic Analysis of Prostate Cancer with Computer Science Methods. (arXiv:2303.15851v1 [cs.IR])

    [http://arxiv.org/abs/2303.15851](http://arxiv.org/abs/2303.15851)

    本文应用数据科学、机器学习和拓扑网络分析方法对不同转移部位的前列腺癌肿瘤进行了基因分析，筛选出了与前列腺癌转移相关的13个基因，准确率达到了92%。

    

    转移性前列腺癌是男性最常见的癌症之一。本文采用数据科学、机器学习和拓扑网络分析方法对不同转移部位的前列腺癌肿瘤进行基因分析。文章提出了一般性的基因表达数据预处理和分析方法来过滤显著基因，并采用机器学习模型和次要肿瘤分类来进一步过滤关键基因。最后，本文对不同类型前列腺癌细胞系样本进行了基因共表达网络分析和社区检测。文章筛选出了与前列腺癌转移相关的13个基因，交叉验证下准确率达到了92%。此外，本文还提供了共表达模式的初步见解。

    Metastatic prostate cancer is one of the most common cancers in men. In the advanced stages of prostate cancer, tumours can metastasise to other tissues in the body, which is fatal. In this thesis, we performed a genetic analysis of prostate cancer tumours at different metastatic sites using data science, machine learning and topological network analysis methods. We presented a general procedure for pre-processing gene expression datasets and pre-filtering significant genes by analytical methods. We then used machine learning models for further key gene filtering and secondary site tumour classification. Finally, we performed gene co-expression network analysis and community detection on samples from different prostate cancer secondary site types. In this work, 13 of the 14,379 genes were selected as the most metastatic prostate cancer related genes, achieving approximately 92% accuracy under cross-validation. In addition, we provide preliminary insights into the co-expression patterns
    
[^7]: 基于级联图卷积网络的多行为推荐

    Multi-Behavior Recommendation with Cascading Graph Convolution Networks. (arXiv:2303.15720v1 [cs.IR])

    [http://arxiv.org/abs/2303.15720](http://arxiv.org/abs/2303.15720)

    本文提出了一种基于级联图卷积网络的多行为推荐模型，能够明确地利用行为链中的依赖关系，以缓解推荐系统数据稀疏或冷启动问题。

    

    多行为推荐利用辅助行为（例如点击和加入购物车）来帮助预测用户在目标行为（例如购买）上的潜在交互，被认为是缓解推荐系统数据稀疏或冷启动问题的有效方法。在现实应用中，多个行为通常按特定顺序进行（例如点击>加入购物车>购买）。在行为链中，后续行为通常比前面的行为展现出更强的用户偏好信号。现有的多行为模型大多未能抓住此类行为链中的依赖关系。为此，本文提出了一种基于级联图卷积网络的新型多行为推荐模型（称为MB-CGCN）。在MB-CGCN中，经过特征变换操作后，从一个行为学习到的嵌入被用作下一个行为嵌入学习的输入特征。这样，我们的模型明确地利用了嵌入学习中的行为依赖性。

    Multi-behavior recommendation, which exploits auxiliary behaviors (e.g., click and cart) to help predict users' potential interactions on the target behavior (e.g., buy), is regarded as an effective way to alleviate the data sparsity or cold-start issues in recommendation. Multi-behaviors are often taken in certain orders in real-world applications (e.g., click>cart>buy). In a behavior chain, a latter behavior usually exhibits a stronger signal of user preference than the former one does. Most existing multi-behavior models fail to capture such dependencies in a behavior chain for embedding learning. In this work, we propose a novel multi-behavior recommendation model with cascading graph convolution networks (named MB-CGCN). In MB-CGCN, the embeddings learned from one behavior are used as the input features for the next behavior's embedding learning after a feature transformation operation. In this way, our model explicitly utilizes the behavior dependencies in embedding learning. Exp
    
[^8]: 高效图像搜索的模型级联

    Model Cascades for Efficient Image Search. (arXiv:2303.15595v1 [cs.IR])

    [http://arxiv.org/abs/2303.15595](http://arxiv.org/abs/2303.15595)

    该论文提出了一种新颖的图像排名算法，使用级联的神经编码器来逐步过滤图像，从而减少了3倍以上的TIR生命周期成本。

    

    现代神经编码器提供了前所未有的文本-图像检索（TIR）准确性。然而，它们高昂的计算成本阻碍了它们在大规模图像搜索中的应用。我们提出了一种新的图像排名算法，它使用逐步增强的神经编码器级联逐步按照它们与给定的文本匹配的好坏程度来过滤图像。 我们的算法将TIR的生命周期成本降低了3倍以上。

    Modern neural encoders offer unprecedented text-image retrieval (TIR) accuracy. However, their high computational cost impedes an adoption to large-scale image searches. We propose a novel image ranking algorithm that uses a cascade of increasingly powerful neural encoders to progressively filter images by how well they match a given text. Our algorithm reduces lifetime TIR costs by over 3x.
    
[^9]: GETT-QA：基于图嵌入的知识图谱问答中的T2T Transformer

    GETT-QA: Graph Embedding based T2T Transformer for Knowledge Graph Question Answering. (arXiv:2303.13284v1 [cs.CL])

    [http://arxiv.org/abs/2303.13284](http://arxiv.org/abs/2303.13284)

    本论文提出了GETT-QA系统，该系统使用T5对自然语言问题生成简化的SPARQL查询，并使用截断的KG嵌入提高了知识图谱问答的性能。

    

    本文提出了一个名为GETT-QA的端到端知识图谱问答系统。GETT-QA使用了T5，这是一种热门的文本到文本预训练语言模型。该模型以自然语言形式的问题作为输入并生成所需SPARQL查询的简化形式。在简化形式中，模型不直接生成实体和关系ID，而是产生相应的实体和关系标签。标签在随后的步骤中与KG实体和关系ID联系起来。为了进一步改进结果，我们指导模型为每个实体生成KG嵌入的截断版本。截断的KG嵌入使得更精细的搜索从而更有效进行消歧。我们发现，T5能够在不改变损失函数的情况下学习截断的KG嵌入，提高了KGQA的性能。因此，我们在Wikidata的LC-QuAD 2.0和SimpleQuestions-Wikidata数据集上报告了端到端KGQA的强大结果。

    In this work, we present an end-to-end Knowledge Graph Question Answering (KGQA) system named GETT-QA. GETT-QA uses T5, a popular text-to-text pre-trained language model. The model takes a question in natural language as input and produces a simpler form of the intended SPARQL query. In the simpler form, the model does not directly produce entity and relation IDs. Instead, it produces corresponding entity and relation labels. The labels are grounded to KG entity and relation IDs in a subsequent step. To further improve the results, we instruct the model to produce a truncated version of the KG embedding for each entity. The truncated KG embedding enables a finer search for disambiguation purposes. We find that T5 is able to learn the truncated KG embeddings without any change of loss function, improving KGQA performance. As a result, we report strong results for LC-QuAD 2.0 and SimpleQuestions-Wikidata datasets on end-to-end KGQA over Wikidata.
    
[^10]: 优化广义基尼指数实现排名公平性

    Optimizing generalized Gini indices for fairness in rankings. (arXiv:2204.06521v4 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2204.06521](http://arxiv.org/abs/2204.06521)

    本文探讨了使用广义基尼福利函数（GGF）作为规范性准则来指定推荐系统应优化的方法，以此实现排名公平性。

    

    越来越多的人关注设计能够对物品生产者或最不满意用户公平的推荐系统。受经济学不平等测量领域的启发，本文探讨了使用广义基尼福利函数（GGF）作为规范性准则来指定推荐系统应优化的方法。GGF根据人口普查中的排名对个体进行加权，将更多的权重放在处境较差的个体上以促进平等。根据这些权重，GGF最小化物品曝光的基尼指数，以促进物品之间的平等，或关注最不满意用户的特定分位数的性能。排名的GGF难以优化，因为它们是不可微分的。我们通过利用非平滑优化和可微排序中使用的投影算子来解决这个挑战。我们使用最多有15k个用户和物品的真实数据集进行实验，结果表明我们的方法可以通过优化GGF有效地促进排名公平性。

    There is growing interest in designing recommender systems that aim at being fair towards item producers or their least satisfied users. Inspired by the domain of inequality measurement in economics, this paper explores the use of generalized Gini welfare functions (GGFs) as a means to specify the normative criterion that recommender systems should optimize for. GGFs weight individuals depending on their ranks in the population, giving more weight to worse-off individuals to promote equality. Depending on these weights, GGFs minimize the Gini index of item exposure to promote equality between items, or focus on the performance on specific quantiles of least satisfied users. GGFs for ranking are challenging to optimize because they are non-differentiable. We resolve this challenge by leveraging tools from non-smooth optimization and projection operators used in differentiable sorting. We present experiments using real datasets with up to 15k users and items, which show that our approach
    


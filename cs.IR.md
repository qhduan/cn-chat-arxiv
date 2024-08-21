# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Touch the Core: Exploring Task Dependence Among Hybrid Targets for Recommendation](https://arxiv.org/abs/2403.17442) | 在混合目标推荐中，通过研究多任务学习问题，探讨了离散转化行为与连续转化之间的依赖关系，解决了核心回归任务对其他任务影响较大的问题。 |
| [^2] | [Non-autoregressive Generative Models for Reranking Recommendation](https://arxiv.org/abs/2402.06871) | 本研究提出了一个非自回归的生成模型用于排序推荐，在多阶段推荐系统中扮演关键角色。该模型旨在提高效率和效果，并解决稀疏训练样本和动态候选项对模型收敛性的挑战。 |
| [^3] | [Finding Already Debunked Narratives via Multistage Retrieval: Enabling Cross-Lingual, Cross-Dataset and Zero-Shot Learning.](http://arxiv.org/abs/2308.05680) | 本研究通过创建新的数据集、评估多语言预训练Transformer模型以及提出多阶段框架来解决了跨语言澄清检索问题。 |

# 详细

[^1]: 触及核心：探索混合目标推荐中任务依赖关系

    Touch the Core: Exploring Task Dependence Among Hybrid Targets for Recommendation

    [https://arxiv.org/abs/2403.17442](https://arxiv.org/abs/2403.17442)

    在混合目标推荐中，通过研究多任务学习问题，探讨了离散转化行为与连续转化之间的依赖关系，解决了核心回归任务对其他任务影响较大的问题。

    

    随着用户行为在商业平台上变得复杂，在线推荐更加关注如何触及核心转化，这些转化与平台的兴趣密切相关。这些核心转化通常是连续的目标，如“观看时间”、“收入”等，它们的预测可以通过之前的离散转化行为来增强。因此，多任务学习（MTL）可以被采用作为学习这些混合目标的范 paradigm。然而，现有的工作主要强调研究离散转化行为之间的顺序依赖关系，而忽视了离散转化与最终连续转化之间的依赖复杂性。此外，同时优化具有更强任务依赖性的混合任务将面临不稳定的问题，其中核心回归任务可能对其他任务产生更大的影响。在本文中，我们研究了具有混合目标的MTL问题。

    arXiv:2403.17442v1 Announce Type: new  Abstract: As user behaviors become complicated on business platforms, online recommendations focus more on how to touch the core conversions, which are highly related to the interests of platforms. These core conversions are usually continuous targets, such as \textit{watch time}, \textit{revenue}, and so on, whose predictions can be enhanced by previous discrete conversion actions. Therefore, multi-task learning (MTL) can be adopted as the paradigm to learn these hybrid targets. However, existing works mainly emphasize investigating the sequential dependence among discrete conversion actions, which neglects the complexity of dependence between discrete conversions and the final continuous conversion. Moreover, simultaneously optimizing hybrid tasks with stronger task dependence will suffer from volatile issues where the core regression task might have a larger influence on other tasks. In this paper, we study the MTL problem with hybrid targets f
    
[^2]: 非自回归的生成模型用于排序推荐

    Non-autoregressive Generative Models for Reranking Recommendation

    [https://arxiv.org/abs/2402.06871](https://arxiv.org/abs/2402.06871)

    本研究提出了一个非自回归的生成模型用于排序推荐，在多阶段推荐系统中扮演关键角色。该模型旨在提高效率和效果，并解决稀疏训练样本和动态候选项对模型收敛性的挑战。

    

    在多阶段推荐系统中，重新排序通过建模项目之间的内部相关性起到了至关重要的作用。重新排序的关键挑战在于在排列的组合空间中探索最佳序列。最近的研究提出了生成器-评估器学习范式，生成器生成多个可行序列，评估器基于估计的列表得分选择最佳序列。生成器至关重要，而生成模型非常适合生成器函数。当前的生成模型采用自回归策略进行序列生成。然而，在实时工业系统中部署自回归模型是具有挑战性的。因此，我们提出了一个非自回归生成模型用于排序推荐（NAR4Rec），以提高效率和效果。为了解决与稀疏训练样本和动态候选项对模型收敛性的挑战，我们引入了一个m

    In a multi-stage recommendation system, reranking plays a crucial role by modeling the intra-list correlations among items.The key challenge of reranking lies in the exploration of optimal sequences within the combinatorial space of permutations. Recent research proposes a generator-evaluator learning paradigm, where the generator generates multiple feasible sequences and the evaluator picks out the best sequence based on the estimated listwise score. Generator is of vital importance, and generative models are well-suited for the generator function. Current generative models employ an autoregressive strategy for sequence generation. However, deploying autoregressive models in real-time industrial systems is challenging. Hence, we propose a Non-AutoRegressive generative model for reranking Recommendation (NAR4Rec) designed to enhance efficiency and effectiveness. To address challenges related to sparse training samples and dynamic candidates impacting model convergence, we introduce a m
    
[^3]: 通过多阶段检索找到已经被澄清的叙述：实现跨语言、跨数据集和零样本学习

    Finding Already Debunked Narratives via Multistage Retrieval: Enabling Cross-Lingual, Cross-Dataset and Zero-Shot Learning. (arXiv:2308.05680v1 [cs.CL])

    [http://arxiv.org/abs/2308.05680](http://arxiv.org/abs/2308.05680)

    本研究通过创建新的数据集、评估多语言预训练Transformer模型以及提出多阶段框架来解决了跨语言澄清检索问题。

    

    检索已经被澄清的叙述的任务旨在检测已经经过事实核查的故事。成功检测到已被澄清的声明不仅减少了专业事实核查人员的手动努力，还可以有助于减缓虚假信息的传播。由于缺乏可用数据，这是一个研究不足的问题，特别是在考虑跨语言任务时，即在检查的在线帖子的语言与事实核查文章的语言不同的情况下进行检索。本文通过以下方式填补了这一空白：（i）创建了一个新颖的数据集，以允许对已被澄清的叙述进行跨语言检索的研究，使用推文作为对事实核查文章数据库的查询；（ii）展示了一个全面的实验，以评估经过微调和现成的多语言预训练Transformer模型在这个任务上的性能；（iii）提出了一个新颖的多阶段框架，将这个跨语言澄清检索问题划分为不同的阶段。

    The task of retrieving already debunked narratives aims to detect stories that have already been fact-checked. The successful detection of claims that have already been debunked not only reduces the manual efforts of professional fact-checkers but can also contribute to slowing the spread of misinformation. Mainly due to the lack of readily available data, this is an understudied problem, particularly when considering the cross-lingual task, i.e. the retrieval of fact-checking articles in a language different from the language of the online post being checked. This paper fills this gap by (i) creating a novel dataset to enable research on cross-lingual retrieval of already debunked narratives, using tweets as queries to a database of fact-checking articles; (ii) presenting an extensive experiment to benchmark fine-tuned and off-the-shelf multilingual pre-trained Transformer models for this task; and (iii) proposing a novel multistage framework that divides this cross-lingual debunk ret
    


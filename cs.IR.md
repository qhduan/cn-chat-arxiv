# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SPAR: Personalized Content-Based Recommendation via Long Engagement Attention](https://arxiv.org/abs/2402.10555) | SPAR是一个基于内容的推荐框架，通过利用PLM、多注意力层和注意力稀疏机制，在会话级别有效地处理长期用户参与历史，提取全面用户兴趣，实现个性化推荐。 |
| [^2] | [Leveraging LLMs for Unsupervised Dense Retriever Ranking](https://arxiv.org/abs/2402.04853) | 本文介绍了一种利用大型语言模型（LLMs）进行无监督选择最佳预训练的密集检索器的新技术。选择合适的检索器对于应用于新的目标语料库并且存在领域转移的情况非常重要。 |
| [^3] | [A Survey on Data-Centric Recommender Systems](https://arxiv.org/abs/2401.17878) | 数据中心推荐系统综述了推荐系统从模型为中心到数据为中心的转变。这篇综述首次系统概述了数据中心推荐系统的基本概念、推荐数据的主要问题以及最近的研究和未来的发展方向。 |
| [^4] | [Model Editing at Scale leads to Gradual and Catastrophic Forgetting](https://arxiv.org/abs/2401.07453) | 评估了当前模型编辑方法在规模化情况下的表现，发现随着模型被顺序编辑多个事实，它会逐渐遗忘先前的事实及执行下游任务的能力。 |
| [^5] | [ChatQA: Building GPT-4 Level Conversational QA Models.](http://arxiv.org/abs/2401.10225) | ChatQA是一系列对话问答模型，可以达到GPT-4级别的准确性。通过两阶段的指令调整方法，可以显著提高大型语言模型在零-shot对话问答中的结果。使用密集检索器进行问答数据集的微调可以实现与最先进的查询重写模型相当的结果，同时降低部署成本。ChatQA-70B在10个对话问答数据集上的平均得分超过了GPT-4，且不依赖于任何来自OpenAI GPT模型的合成数据。 |
| [^6] | [UOEP: User-Oriented Exploration Policy for Enhancing Long-Term User Experiences in Recommender Systems.](http://arxiv.org/abs/2401.09034) | UOEP是一种用户导向的探索策略，针对推荐系统中不同活跃水平的用户群体实现细粒度探索，以增强用户的长期体验。 |
| [^7] | [Adversarial Item Promotion on Visually-Aware Recommender Systems by Guided Diffusion.](http://arxiv.org/abs/2312.15826) | 本论文研究了对视觉感知推荐系统的敌对项目推广的引导扩散，揭示了物品供应商如何通过构建敌对图像来操纵物品暴露率，同时指出了敌对图像存在的问题和挑战。 |
| [^8] | [Density-based User Representation through Gaussian Process Regression for Multi-interest Personalized Retrieval.](http://arxiv.org/abs/2310.20091) | 本研究引入了一种基于密度的用户表示(DURs)，利用高斯过程回归实现了有效的多兴趣推荐和检索。该方法不仅能够捕捉用户的兴趣变化，还具备不确定性感知能力，并且适用于大量用户的规模。 |
| [^9] | [Privacy-Preserving Recommender Systems with Synthetic Query Generation using Differentially Private Large Language Models.](http://arxiv.org/abs/2305.05973) | 提出使用差分隐私大语言模型合成查询的隐私保护推荐系统，可以安全有效地训练深度检索模型并提高检索质量。 |

# 详细

[^1]: SPAR：通过长期参与注意力实现个性化基于内容的推荐

    SPAR: Personalized Content-Based Recommendation via Long Engagement Attention

    [https://arxiv.org/abs/2402.10555](https://arxiv.org/abs/2402.10555)

    SPAR是一个基于内容的推荐框架，通过利用PLM、多注意力层和注意力稀疏机制，在会话级别有效地处理长期用户参与历史，提取全面用户兴趣，实现个性化推荐。

    

    利用用户长期参与历史对个性化内容推荐至关重要。预训练语言模型（PLMs）在自然语言处理领域的成功导致它们被用于编码用户历史和候选项，将内容推荐视为文本语义匹配任务。然而，现有工作仍然在处理非常长的用户历史文本和不足的用户-物品交互方面存在困难。本文介绍了一种基于内容的推荐框架SPAR，有效应对了从长期用户参与历史中提取全面用户兴趣的挑战。它通过利用PLM、多注意力层和注意力稀疏机制以会话为基础对用户的历史进行编码。用户和物品侧特征被充分融合进行参与预测，同时保持双方的独立表示，这对于实际模型部署是有效的。

    arXiv:2402.10555v1 Announce Type: cross  Abstract: Leveraging users' long engagement histories is essential for personalized content recommendations. The success of pretrained language models (PLMs) in NLP has led to their use in encoding user histories and candidate items, framing content recommendations as textual semantic matching tasks. However, existing works still struggle with processing very long user historical text and insufficient user-item interaction. In this paper, we introduce a content-based recommendation framework, SPAR, which effectively tackles the challenges of holistic user interest extraction from the long user engagement history. It achieves so by leveraging PLM, poly-attention layers and attention sparsity mechanisms to encode user's history in a session-based manner. The user and item side features are sufficiently fused for engagement prediction while maintaining standalone representations for both sides, which is efficient for practical model deployment. Mor
    
[^2]: 利用LLMs进行无监督的密集检索排名

    Leveraging LLMs for Unsupervised Dense Retriever Ranking

    [https://arxiv.org/abs/2402.04853](https://arxiv.org/abs/2402.04853)

    本文介绍了一种利用大型语言模型（LLMs）进行无监督选择最佳预训练的密集检索器的新技术。选择合适的检索器对于应用于新的目标语料库并且存在领域转移的情况非常重要。

    

    本文介绍了一种利用大型语言模型（LLMs）确定特定测试（目标）语料库最合适的密集检索器的新颖无监督技术。选择合适的密集检索器对于许多采用这些在公开数据集上训练的检索器进行编码或在新的私有目标语料库中进行搜索的信息检索应用程序至关重要。当密集检索器应用于与原始训练集在领域或任务上有差异的目标语料库时，其有效性可能会大大降低。在目标语料库没有标注的情况下，例如在零样本场景中，无法直接评估模型在目标语料库上的效果。因此，无监督选择最佳预训练的密集检索器，特别是在领域迁移条件下，成为一个关键挑战。现有的密集检索器排序方法在解决这些领域迁移问题方面存在不足。

    This paper introduces a novel unsupervised technique that utilizes large language models (LLMs) to determine the most suitable dense retriever for a specific test(target) corpus. Selecting the appropriate dense retriever is vital for numerous IR applications that employ these retrievers, trained on public datasets, to encode or conduct searches within a new private target corpus. The effectiveness of a dense retriever can significantly diminish when applied to a target corpus that diverges in domain or task from the original training set. The problem becomes more pronounced in cases where the target corpus is unlabeled, e.g. in zero-shot scenarios, rendering direct evaluation of the model's effectiveness on the target corpus unattainable. Therefore, the unsupervised selection of an optimally pre-trained dense retriever, especially under conditions of domain shift, emerges as a critical challenge. Existing methodologies for ranking dense retrievers fall short in addressing these domain 
    
[^3]: 数据中心推荐系统综述

    A Survey on Data-Centric Recommender Systems

    [https://arxiv.org/abs/2401.17878](https://arxiv.org/abs/2401.17878)

    数据中心推荐系统综述了推荐系统从模型为中心到数据为中心的转变。这篇综述首次系统概述了数据中心推荐系统的基本概念、推荐数据的主要问题以及最近的研究和未来的发展方向。

    

    推荐系统已成为应对信息过载的重要工具，适用于各种实际场景。最近推荐系统的发展趋势出现了范式转变，从模型为中心的创新转向数据质量和数量的重要性。这一变化引出了数据中心推荐系统（Data-Centric RS）的概念，标志着该领域的重要发展。本综述首次系统地概述了数据中心推荐系统，包括1）推荐数据和数据中心推荐系统的基本概念；2）推荐数据面临的三个主要问题；3）为解决这些问题而开展的最近研究；以及4）数据中心推荐系统可能的未来发展方向。

    Recommender systems (RS) have become essential tools for mitigating information overload in a range of real-world scenarios. Recent trends in RS have seen a paradigm shift, moving the spotlight from model-centric innovations to the importance of data quality and quantity. This evolution has given rise to the concept of data-centric recommender systems (Data-Centric RS), marking a significant development in the field. This survey provides the first systematic overview of Data-Centric RS, covering 1) the foundational concepts of recommendation data and Data-Centric RS; 2) three primary issues in recommendation data; 3) recent research developed to address these issues; and 4) several potential future directions in Data-Centric RS.
    
[^4]: 规模化模型编辑会导致渐进性和突发性遗忘

    Model Editing at Scale leads to Gradual and Catastrophic Forgetting

    [https://arxiv.org/abs/2401.07453](https://arxiv.org/abs/2401.07453)

    评估了当前模型编辑方法在规模化情况下的表现，发现随着模型被顺序编辑多个事实，它会逐渐遗忘先前的事实及执行下游任务的能力。

    

    在大型语言模型中编辑知识是一种具有吸引力的能力，它使我们能够在预训练期间纠正错误学习的事实，同时使用不断增长的新事实列表更新模型。我们认为，为了使模型编辑具有实际效用，我们必须能够对同一模型进行多次编辑。因此，我们评估了当前规模下的模型编辑方法，重点关注两种最先进的方法：ROME 和 MEMIT。我们发现，随着模型被顺序编辑多个事实，它不断地遗忘先前编辑过的事实以及执行下游任务的能力。这种遗忘分为两个阶段--初始的渐进性遗忘阶段，随后是突然或灾难性的遗忘。

    arXiv:2401.07453v2 Announce Type: replace-cross  Abstract: Editing knowledge in large language models is an attractive capability to have which allows us to correct incorrectly learnt facts during pre-training, as well as update the model with an ever-growing list of new facts. While existing model editing techniques have shown promise, they are usually evaluated using metrics for reliability, specificity and generalization over one or few edits. We argue that for model editing to have practical utility, we must be able to make multiple edits to the same model. With this in mind, we evaluate the current model editing methods at scale, focusing on two state of the art methods: ROME and MEMIT. We find that as the model is edited sequentially with multiple facts, it continually forgets previously edited facts and the ability to perform downstream tasks. This forgetting happens in two phases -- an initial gradual but progressive forgetting phase followed by abrupt or catastrophic forgettin
    
[^5]: ChatQA: 构建GPT-4级对话问答模型

    ChatQA: Building GPT-4 Level Conversational QA Models. (arXiv:2401.10225v1 [cs.CL])

    [http://arxiv.org/abs/2401.10225](http://arxiv.org/abs/2401.10225)

    ChatQA是一系列对话问答模型，可以达到GPT-4级别的准确性。通过两阶段的指令调整方法，可以显著提高大型语言模型在零-shot对话问答中的结果。使用密集检索器进行问答数据集的微调可以实现与最先进的查询重写模型相当的结果，同时降低部署成本。ChatQA-70B在10个对话问答数据集上的平均得分超过了GPT-4，且不依赖于任何来自OpenAI GPT模型的合成数据。

    

    在这项工作中，我们介绍了ChatQA，一系列具有GPT-4级别准确性的对话问答模型。具体地，我们提出了一个两阶段的指令调整方法，可以显著提高大型语言模型（LLM）在零-shot对话问答中的结果。为了处理对话问答中的检索问题，我们在多轮问答数据集上进行了密集检索器的微调，这样可以提供与使用最先进的查询重写模型相当的结果，同时大大降低部署成本。值得注意的是，我们的ChatQA-70B可以在10个对话问答数据集的平均分上超过GPT-4（54.14 vs. 53.90），而不依赖于OpenAI GPT模型的任何合成数据。

    In this work, we introduce ChatQA, a family of conversational question answering (QA) models, that obtain GPT-4 level accuracies. Specifically, we propose a two-stage instruction tuning method that can significantly improve the zero-shot conversational QA results from large language models (LLMs). To handle retrieval in conversational QA, we fine-tune a dense retriever on a multi-turn QA dataset, which provides comparable results to using the state-of-the-art query rewriting model while largely reducing deployment cost. Notably, our ChatQA-70B can outperform GPT-4 in terms of average score on 10 conversational QA datasets (54.14 vs. 53.90), without relying on any synthetic data from OpenAI GPT models.
    
[^6]: UOEP: 用户导向的探索策略以增强推荐系统中的长期用户体验

    UOEP: User-Oriented Exploration Policy for Enhancing Long-Term User Experiences in Recommender Systems. (arXiv:2401.09034v1 [cs.IR])

    [http://arxiv.org/abs/2401.09034](http://arxiv.org/abs/2401.09034)

    UOEP是一种用户导向的探索策略，针对推荐系统中不同活跃水平的用户群体实现细粒度探索，以增强用户的长期体验。

    

    强化学习（RL）已经在推荐系统中得到广泛应用，有效地探索用户的兴趣，以提升用户的长期体验。然而，现代推荐系统中存在着数千万个项目之间的不同用户行为模式，这增加了探索的难度。例如，不同活跃水平的用户行为需要不同强度的探索，而之前的研究往往忽视了这一方面，对所有用户应用统一的探索策略，最终损害了用户的长期体验。为了解决这些挑战，我们提出了用户导向的探索策略（UOEP），一种在用户群体中实现细粒度探索的新方法。我们首先构建了一个分布式评论家，它允许在不同的累积奖励反馈的分位数水平下进行策略优化，表示具有不同活动水平的用户群体。在这个评论家的指导下，我们设计了一组不同的演员。

    Reinforcement learning (RL) has gained traction for enhancing user long-term experiences in recommender systems by effectively exploring users' interests. However, modern recommender systems exhibit distinct user behavioral patterns among tens of millions of items, which increases the difficulty of exploration. For example, user behaviors with different activity levels require varying intensity of exploration, while previous studies often overlook this aspect and apply a uniform exploration strategy to all users, which ultimately hurts user experiences in the long run. To address these challenges, we propose User-Oriented Exploration Policy (UOEP), a novel approach facilitating fine-grained exploration among user groups. We first construct a distributional critic which allows policy optimization under varying quantile levels of cumulative reward feedbacks from users, representing user groups with varying activity levels. Guided by this critic, we devise a population of distinct actors 
    
[^7]: 对视觉感知推荐系统的敌对项目推广的引导扩散研究

    Adversarial Item Promotion on Visually-Aware Recommender Systems by Guided Diffusion. (arXiv:2312.15826v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2312.15826](http://arxiv.org/abs/2312.15826)

    本论文研究了对视觉感知推荐系统的敌对项目推广的引导扩散，揭示了物品供应商如何通过构建敌对图像来操纵物品暴露率，同时指出了敌对图像存在的问题和挑战。

    

    视觉感知推荐系统在那些视觉元素对用户潜在偏好的推断有显著贡献的领域中得到了广泛应用。尽管纳入视觉信息有望提高推荐的准确性和缓解冷启动问题，但需要指出的是，包含物品图像可能会引入重大的安全挑战。一些现有的工作表明，物品供应商可以通过构建敌对图像来操纵物品暴露率以其利益。然而，这些工作无法揭示视觉感知推荐系统面对敌对图像时的真实脆弱性，原因如下：（1）生成的敌对图像明显畸变，易于被人类观察者检测到；（2）攻击的有效性不一致，甚至在一些情况下无效。为了揭示面对敌对图像时视觉感知推荐系统的真实脆弱性

    Visually-aware recommender systems have found widespread application in domains where visual elements significantly contribute to the inference of users' potential preferences. While the incorporation of visual information holds the promise of enhancing recommendation accuracy and alleviating the cold-start problem, it is essential to point out that the inclusion of item images may introduce substantial security challenges. Some existing works have shown that the item provider can manipulate item exposure rates to its advantage by constructing adversarial images. However, these works cannot reveal the real vulnerability of visually-aware recommender systems because (1) The generated adversarial images are markedly distorted, rendering them easily detectable by human observers; (2) The effectiveness of the attacks is inconsistent and even ineffective in some scenarios. To shed light on the real vulnerabilities of visually-aware recommender systems when confronted with adversarial images
    
[^8]: 基于高斯过程回归的密度用户表示方法用于多兴趣个性化检索

    Density-based User Representation through Gaussian Process Regression for Multi-interest Personalized Retrieval. (arXiv:2310.20091v1 [cs.IR])

    [http://arxiv.org/abs/2310.20091](http://arxiv.org/abs/2310.20091)

    本研究引入了一种基于密度的用户表示(DURs)，利用高斯过程回归实现了有效的多兴趣推荐和检索。该方法不仅能够捕捉用户的兴趣变化，还具备不确定性感知能力，并且适用于大量用户的规模。

    

    在设计个性化推荐系统中，准确建模用户的各种多样化和动态的兴趣仍然是一个重大挑战。现有的用户建模方法，如单点和多点表示，存在准确性、多样性、计算成本和适应性方面的局限性。为了克服这些不足，我们引入了一种新颖的模型——基于密度的用户表示(DURs)，它利用高斯过程回归实现有效的多兴趣推荐和检索。我们的方法GPR4DUR利用DURs来捕捉用户的兴趣变化，无需手动调整，同时具备不确定性感知能力，并且适用于大量用户的规模。使用真实世界的离线数据集进行的实验证实了GPR4DUR的适应性和效率，而使用模拟用户的在线实验则证明了它通过有效利用模型的不确定性，能够解决探索-开发的平衡问题。

    Accurate modeling of the diverse and dynamic interests of users remains a significant challenge in the design of personalized recommender systems. Existing user modeling methods, like single-point and multi-point representations, have limitations w.r.t. accuracy, diversity, computational cost, and adaptability. To overcome these deficiencies, we introduce density-based user representations (DURs), a novel model that leverages Gaussian process regression for effective multi-interest recommendation and retrieval. Our approach, GPR4DUR, exploits DURs to capture user interest variability without manual tuning, incorporates uncertainty-awareness, and scales well to large numbers of users. Experiments using real-world offline datasets confirm the adaptability and efficiency of GPR4DUR, while online experiments with simulated users demonstrate its ability to address the exploration-exploitation trade-off by effectively utilizing model uncertainty.
    
[^9]: 使用差分隐私大语言模型合成查询的隐私保护推荐系统.

    Privacy-Preserving Recommender Systems with Synthetic Query Generation using Differentially Private Large Language Models. (arXiv:2305.05973v1 [cs.CL])

    [http://arxiv.org/abs/2305.05973](http://arxiv.org/abs/2305.05973)

    提出使用差分隐私大语言模型合成查询的隐私保护推荐系统，可以安全有效地训练深度检索模型并提高检索质量。

    

    我们提出了一种新颖的方法，使用差分隐私大语言模型（LLMs）开发隐私保护的大规模推荐系统，克服了在训练这些复杂系统时的某些挑战和限制。我们的方法特别适用于基于LLM的推荐系统的新兴领域，但也可以轻松地用于处理自然语言输入表示的任何推荐系统。我们的方法涉及使用DP训练方法，对公开预训练的LLM在查询生成任务上进行微调。生成的模型可以生成私有合成查询，代表原始查询，可以在任何下游非私有推荐训练过程中自由共享，而不会产生任何额外的隐私成本。我们评估了我们的方法对安全训练有效的深度检索模型的能力，我们观察到它们的检索质量有显着的提高，而不会损害查询级别的隐私。

    We propose a novel approach for developing privacy-preserving large-scale recommender systems using differentially private (DP) large language models (LLMs) which overcomes certain challenges and limitations in DP training these complex systems. Our method is particularly well suited for the emerging area of LLM-based recommender systems, but can be readily employed for any recommender systems that process representations of natural language inputs. Our approach involves using DP training methods to fine-tune a publicly pre-trained LLM on a query generation task. The resulting model can generate private synthetic queries representative of the original queries which can be freely shared for any downstream non-private recommendation training procedures without incurring any additional privacy cost. We evaluate our method on its ability to securely train effective deep retrieval models, and we observe significant improvements in their retrieval quality without compromising query-level pri
    


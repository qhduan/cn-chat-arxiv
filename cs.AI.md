# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Personalizing explanations of AI-driven hints to users cognitive abilities: an empirical evaluation](https://arxiv.org/abs/2403.04035) | 该研究调查了如何个性化智能辅导系统生成的提示的解释，以帮助促进学生学习，实证结果表明，该个性化方法显著提高了目标用户与提示解释的互动、理解和学习效果。 |
| [^2] | [Assessment of Reinforcement Learning for Macro Placement.](http://arxiv.org/abs/2302.11014) | 本论文提供了基于强化学习的宏观布局方法以及Circuit Training (CT)实现的开源代码和评估。研究人员评估了CT相对于多个可替代的宏观布局方法，并进行了学术性混合尺寸布局基准测试和消融和稳定性研究，为未来的相关研究提供了方向。 |
| [^3] | [Increasing Fairness via Combination with Learning Guarantees.](http://arxiv.org/abs/2301.10813) | 该论文提出了一种公平质量度量方法，名为判别风险，旨在反映个体和群体公平性。此外，研究者还讨论了公平性是否可以在理论上得到保证。 |

# 详细

[^1]: 个性化AI驱动提示对用户认知能力的解释：一项实证评估

    Personalizing explanations of AI-driven hints to users cognitive abilities: an empirical evaluation

    [https://arxiv.org/abs/2403.04035](https://arxiv.org/abs/2403.04035)

    该研究调查了如何个性化智能辅导系统生成的提示的解释，以帮助促进学生学习，实证结果表明，该个性化方法显著提高了目标用户与提示解释的互动、理解和学习效果。

    

    我们调查了个性化解释智能辅导系统生成的提示，以证明它们提供提示促进学生学习的有效性。个性化针对具有两种特征（认知需求和认真度）较低水平的学生，旨在增强这些学生对解释的参与，基于先前研究发现，这些学生不会自然参与解释，但如果他们这样做将会受益。为了评估个性化的有效性，我们进行了一项用户研究，我们发现我们提出的个性化显著增加了我们目标用户与提示解释的互动、他们对提示的理解以及他们的学习。因此，这项工作为个性化AI驱动解释提供了有价值的见解，适用于如学习等认知要求高的任务。

    arXiv:2403.04035v1 Announce Type: new  Abstract: We investigate personalizing the explanations that an Intelligent Tutoring System generates to justify the hints it provides to students to foster their learning. The personalization targets students with low levels of two traits, Need for Cognition and Conscientiousness, and aims to enhance these students' engagement with the explanations, based on prior findings that these students do not naturally engage with the explanations but they would benefit from them if they do. To evaluate the effectiveness of the personalization, we conducted a user study where we found that our proposed personalization significantly increases our target users' interaction with the hint explanations, their understanding of the hints and their learning. Hence, this work provides valuable insights into effectively personalizing AI-driven explanations for cognitively demanding tasks such as learning.
    
[^2]: 强化学习在宏观布局中的评估

    Assessment of Reinforcement Learning for Macro Placement. (arXiv:2302.11014v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.11014](http://arxiv.org/abs/2302.11014)

    本论文提供了基于强化学习的宏观布局方法以及Circuit Training (CT)实现的开源代码和评估。研究人员评估了CT相对于多个可替代的宏观布局方法，并进行了学术性混合尺寸布局基准测试和消融和稳定性研究，为未来的相关研究提供了方向。

    

    我们提供了Google Brain深度强化学习方法在宏观布局及其Circuit Training (CT)实现的开放透明实现和评估，并在GitHub中实现了CT的关键"黑盒"元素，澄清了CT与Nature论文之间的差异。我们开发并发布了新的对开放实现的测试用例。我们评估了CT及多个可替代的宏观布局方法，所有的评估流程和相关脚本都在GitHub上公开。我们的实验还包括了学术性混合尺寸布局基准测试，以及消融和稳定性研究。我们评论了Nature和CT的影响，以及未来研究的方向。

    We provide open, transparent implementation and assessment of Google Brain's deep reinforcement learning approach to macro placement and its Circuit Training (CT) implementation in GitHub. We implement in open source key "blackbox" elements of CT, and clarify discrepancies between CT and Nature paper. New testcases on open enablements are developed and released. We assess CT alongside multiple alternative macro placers, with all evaluation flows and related scripts public in GitHub. Our experiments also encompass academic mixed-size placement benchmarks, as well as ablation and stability studies. We comment on the impact of Nature and CT, as well as directions for future research.
    
[^3]: 通过学习保证提高公平性

    Increasing Fairness via Combination with Learning Guarantees. (arXiv:2301.10813v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.10813](http://arxiv.org/abs/2301.10813)

    该论文提出了一种公平质量度量方法，名为判别风险，旨在反映个体和群体公平性。此外，研究者还讨论了公平性是否可以在理论上得到保证。

    

    随着机器学习系统在越来越多的现实场景中得到广泛应用，对于隐藏在机器学习模型中的潜在歧视的担忧正在增加。许多技术已经被开发出来以增强公平性，包括常用的群体公平性度量和几种结合集成学习的公平感知方法。然而，现有的公平度量只能关注其中之一，即群体公平性或个体公平性，它们之间的硬性兼容性暗示了即使其中之一得到满足，仍可能存在偏见。此外，现有的提升公平性的机制通常只提供经验结果来证明其有效性，但很少有论文讨论公平性是否可以在理论上得到保证。为了解决这些问题，本文提出了一种公平质量度量方法——判别风险，以反映个体和群体公平性两个方面。此外，我们还研究了p...

    The concern about underlying discrimination hidden in ML models is increasing, as ML systems have been widely applied in more and more real-world scenarios and any discrimination hidden in them will directly affect human life. Many techniques have been developed to enhance fairness including commonly-used group fairness measures and several fairness-aware methods combining ensemble learning. However, existing fairness measures can only focus on one aspect -- either group or individual fairness, and the hard compatibility among them indicates a possibility of remaining biases even if one of them is satisfied. Moreover, existing mechanisms to boost fairness usually present empirical results to show validity, yet few of them discuss whether fairness can be boosted with certain theoretical guarantees. To address these issues, we propose a fairness quality measure named discriminative risk in this paper to reflect both individual and group fairness aspects. Furthermore, we investigate the p
    


# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RAP: Retrieval-Augmented Planner for Adaptive Procedure Planning in Instructional Videos](https://arxiv.org/abs/2403.18600) | 提出了一种新的实际设置，称为指导视频中的自适应程序规划，克服了在实际场景中步骤长度变化的模型不具有泛化能力、理解步骤之间的时间关系知识对于生成合理且可执行的计划至关重要以及用步骤级标签或序列级标签标注指导视频耗时且劳动密集的问题 |
| [^2] | [ChatDiet: Empowering Personalized Nutrition-Oriented Food Recommender Chatbots through an LLM-Augmented Framework](https://arxiv.org/abs/2403.00781) | 这项研究介绍了ChatDiet，一个借助LLM技术构建的框架，能够帮助个性化营养导向食品推荐聊天机器人提供个性化和可解释的推荐。 |
| [^3] | [Are LLMs Ready for Real-World Materials Discovery?](https://arxiv.org/abs/2402.05200) | LLMs在材料科学中的应用受限，无法实现实际应用。我们提出了基于材料科学知识和假设测试的MatSci-LLMs框架，并描述了关键的材料科学信息提取挑战。 |
| [^4] | [Towards Autonomous Supply Chains: Definition, Characteristics, Conceptual Framework, and Autonomy Levels.](http://arxiv.org/abs/2401.14183) | 本论文提出了自主供应链（ASC）的正式定义、特征和辅助概念，提出了一个分层概念框架和供应链自治参考模型。通过案例研究展示了基于这些概念的初始ASC实施。 |
| [^5] | [Realism in Action: Anomaly-Aware Diagnosis of Brain Tumors from Medical Images Using YOLOv8 and DeiT.](http://arxiv.org/abs/2401.03302) | 本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤，并解决了在罕见情况下的肿瘤检测问题。研究使用了来自国家脑映射实验室的数据集，通过修改样本数量和患者分布，使模型能够应对真实世界场景中的异常情况。 |
| [^6] | [Couples can be tractable: New algorithms and hardness results for the Hospitals / Residents problem with Couples.](http://arxiv.org/abs/2311.00405) | 本研究提出了一种新的多项式时间算法，用于解决带夫妻的医院/居民问题，并证明了算法的多项式时间可解性以及对稳定B匹配问题的应用。算法能够找到一个接近可行的稳定匹配，并应用于不同类型的实例。 |
| [^7] | [Non-Smooth Weakly-Convex Finite-sum Coupled Compositional Optimization.](http://arxiv.org/abs/2310.03234) | 本文研究了一种新的组合优化问题，称为非光滑弱凸有限和耦合组合优化(NSWC FCCO)，通过扩展已有的研究，我们研究了非光滑弱凸FCCO的问题，并提出了一种单循环算法来找到Moreau环的ε-稳定点。 |
| [^8] | [WASA: WAtermark-based Source Attribution for Large Language Model-Generated Data.](http://arxiv.org/abs/2310.00646) | 本文提出了一种基于水印的框架WASA，通过允许大型语言模型生成带有嵌入源信息的合成文本水印来解决源归属和数据来源的问题。 |
| [^9] | [Benchmarking Cognitive Biases in Large Language Models as Evaluators.](http://arxiv.org/abs/2309.17012) | 本研究对15个不同大小的大型语言模型进行了评估，发现它们作为评估器存在认知偏差，尤其在文本质量评估中表现出较强的偏见，这对其鲁棒性提出了质疑。同时，研究还发现了人类和机器偏好之间的相关性。 |
| [^10] | [TempFuser: Learning Tactical and Agile Flight Maneuvers in Aerial Dogfights using a Long Short-Term Temporal Fusion Transformer.](http://arxiv.org/abs/2308.03257) | TempFuser是一种长短时序融合转换器，能够学习空中格斗中的战术和敏捷飞行动作。经过训练，模型成功地学会了复杂的战斗动作，并在面对高级对手时展现出人类一样的战术动作。 |
| [^11] | [Unified Embedding Based Personalized Retrieval in Etsy Search.](http://arxiv.org/abs/2306.04833) | 本论文提出了一种将图形、转换和基于术语的嵌入结合起来的统一嵌入模型，并利用端到端训练模型进行个性化检索，以解决Etsy搜索中的语义差距问题。同时，本文分享了特征工程、硬负采样策略和应用变压器模型的新策略，以构建具有工业规模的模型来改善整体搜索体验。 |
| [^12] | [An Interpretable Machine Learning System to Identify EEG Patterns on the Ictal-Interictal-Injury Continuum.](http://arxiv.org/abs/2211.05207) | 该论文设计了一种可解释的深度学习模型，以预测ICU脑电监测中常见的6种脑波图案的存在，并提供高质量的解释和三种解释方法，这对于建立AI的信任和临床采用至关重要。 |

# 详细

[^1]: RAP：检索增强型规划器用于指导视频中的自适应程序规划

    RAP: Retrieval-Augmented Planner for Adaptive Procedure Planning in Instructional Videos

    [https://arxiv.org/abs/2403.18600](https://arxiv.org/abs/2403.18600)

    提出了一种新的实际设置，称为指导视频中的自适应程序规划，克服了在实际场景中步骤长度变化的模型不具有泛化能力、理解步骤之间的时间关系知识对于生成合理且可执行的计划至关重要以及用步骤级标签或序列级标签标注指导视频耗时且劳动密集的问题

    

    指导视频中的程序规划涉及根据初始和目标状态的视觉观察生成一系列动作步骤。尽管这一任务取得了快速进展，仍然存在一些关键挑战需要解决：（1）自适应程序：先前的工作存在一个不切实际的假设，即动作步骤的数量是已知且固定的，导致在实际场景中，步骤长度变化的模型不具有泛化能力。（2）时间关系：理解步骤之间的时间关系知识对于生成合理且可执行的计划至关重要。（3）注释成本：用步骤级标签（即时间戳）或序列级标签（即动作类别）标注指导视频是耗时且劳动密集的，限制了其泛化能力到大规模数据集。在这项工作中，我们提出了一个新的实际设置，称为指导视频中的自适应程序规划

    arXiv:2403.18600v1 Announce Type: cross  Abstract: Procedure Planning in instructional videos entails generating a sequence of action steps based on visual observations of the initial and target states. Despite the rapid progress in this task, there remain several critical challenges to be solved: (1) Adaptive procedures: Prior works hold an unrealistic assumption that the number of action steps is known and fixed, leading to non-generalizable models in real-world scenarios where the sequence length varies. (2) Temporal relation: Understanding the step temporal relation knowledge is essential in producing reasonable and executable plans. (3) Annotation cost: Annotating instructional videos with step-level labels (i.e., timestamp) or sequence-level labels (i.e., action category) is demanding and labor-intensive, limiting its generalizability to large-scale datasets.In this work, we propose a new and practical setting, called adaptive procedure planning in instructional videos, where the
    
[^2]: ChatDiet：通过LLM增强框架赋能个性化营养导向食品推荐聊天机器人

    ChatDiet: Empowering Personalized Nutrition-Oriented Food Recommender Chatbots through an LLM-Augmented Framework

    [https://arxiv.org/abs/2403.00781](https://arxiv.org/abs/2403.00781)

    这项研究介绍了ChatDiet，一个借助LLM技术构建的框架，能够帮助个性化营养导向食品推荐聊天机器人提供个性化和可解释的推荐。

    

    食物对健康的深远影响使得先进的营养导向食品推荐服务成为必要。传统方法往往缺乏个性化、可解释性和互动性等关键元素。虽然大型语言模型（LLMs）带来了解释性和可解释性，但它们单独的使用未能实现真正的个性化。本文介绍了ChatDiet，一种新颖的LLM驱动框架，专门设计用于个性化营养导向食品推荐聊天机器人。ChatDiet集成了个人和人群模型，辅以一个协调器，无缝检索和处理相关信息。其结果是动态提供个性化和可解释的食品推荐，根据个人用户喜好定制。我们对ChatDiet进行了评估，包括一个引人入胜的案例研究，在案例研究中建立了一个因果个人模型来估计个人营养效果。

    arXiv:2403.00781v1 Announce Type: cross  Abstract: The profound impact of food on health necessitates advanced nutrition-oriented food recommendation services. Conventional methods often lack the crucial elements of personalization, explainability, and interactivity. While Large Language Models (LLMs) bring interpretability and explainability, their standalone use falls short of achieving true personalization. In this paper, we introduce ChatDiet, a novel LLM-powered framework designed specifically for personalized nutrition-oriented food recommendation chatbots. ChatDiet integrates personal and population models, complemented by an orchestrator, to seamlessly retrieve and process pertinent information. The result is a dynamic delivery of personalized and explainable food recommendations, tailored to individual user preferences. Our evaluation of ChatDiet includes a compelling case study, where we establish a causal personal model to estimate individual nutrition effects. Our assessmen
    
[^3]: LLMs是否准备好应用于真实世界的材料发现？

    Are LLMs Ready for Real-World Materials Discovery?

    [https://arxiv.org/abs/2402.05200](https://arxiv.org/abs/2402.05200)

    LLMs在材料科学中的应用受限，无法实现实际应用。我们提出了基于材料科学知识和假设测试的MatSci-LLMs框架，并描述了关键的材料科学信息提取挑战。

    

    大型语言模型（LLMs）为材料科学中的强大语言处理工具提供了令人兴奋的可能性，加快了材料研究的进展。然而，LLMs在实际应用中仍存在不足，无法成为实用的材料科学工具。本文通过展示LLMs在材料科学中的相关失败案例，揭示了LLMs在理解和推理复杂、相互关联的材料科学知识方面的现有限制。鉴于这些缺点，我们提出了一种开发基于材料科学知识和假设生成与测试的材料科学LLMs（MatSci-LLMs）的框架。实现高性能的MatSci-LLMs的路径在很大程度上取决于建立高质量的多模态数据集，这些数据集来自科学文献，其中存在各种信息提取挑战。因此，我们描述了关键的材料科学信息提取挑战。

    Large Language Models (LLMs) create exciting possibilities for powerful language processing tools to accelerate research in materials science. While LLMs have great potential to accelerate materials understanding and discovery, they currently fall short in being practical materials science tools. In this position paper, we show relevant failure cases of LLMs in materials science that reveal current limitations of LLMs related to comprehending and reasoning over complex, interconnected materials science knowledge. Given those shortcomings, we outline a framework for developing Materials Science LLMs (MatSci-LLMs) that are grounded in materials science knowledge and hypothesis generation followed by hypothesis testing. The path to attaining performant MatSci-LLMs rests in large part on building high-quality, multi-modal datasets sourced from scientific literature where various information extraction challenges persist. As such, we describe key materials science information extraction cha
    
[^4]: 强调自主供应链：定义、特征、概念框架和自主水平

    Towards Autonomous Supply Chains: Definition, Characteristics, Conceptual Framework, and Autonomy Levels. (arXiv:2401.14183v1 [cs.AI])

    [http://arxiv.org/abs/2401.14183](http://arxiv.org/abs/2401.14183)

    本论文提出了自主供应链（ASC）的正式定义、特征和辅助概念，提出了一个分层概念框架和供应链自治参考模型。通过案例研究展示了基于这些概念的初始ASC实施。

    

    最近的全球性干扰，如大流行和地缘政治冲突，深刻暴露了传统供应链的脆弱性，需要探索更有弹性的替代方案。自主供应链（ASC）作为潜在解决方案出现，提供了在动荡的贸易环境中增加可见性、灵活性和弹性的优势。尽管工业界和学术界多年来一直在讨论ASC，但缺乏良好建立的理论基础。本文通过提供ASC的正式定义以及其定义特征和辅助概念来填补这一研究空白。我们提出了一个名为MIISI模型的分层概念框架。通过一个重点关注肉类供应链的案例研究，展示了基于这一概念模型的初始ASC实施。此外，我们还引入了一个七级供应链自治参考模型，描绘了实现完全供应链自治的轨迹。

    Recent global disruptions, such as the pandemic and geopolitical conflicts, have profoundly exposed vulnerabilities in traditional supply chains, requiring exploration of more resilient alternatives. Autonomous supply chains (ASCs) have emerged as a potential solution, offering increased visibility, flexibility, and resilience in turbulent trade environments. Despite discussions in industry and academia over several years, ASCs lack well-established theoretical foundations. This paper addresses this research gap by presenting a formal definition of ASC along with its defining characteristics and auxiliary concepts. We propose a layered conceptual framework called the MIISI model. An illustrative case study focusing on the meat supply chain demonstrates an initial ASC implementation based on this conceptual model. Additionally, we introduce a seven-level supply chain autonomy reference model, delineating a trajectory towards achieving a full supply chain autonomy. Recognising that this 
    
[^5]: 行动中的现实主义：使用YOLOv8和DeiT从医学图像中诊断脑肿瘤的异常感知

    Realism in Action: Anomaly-Aware Diagnosis of Brain Tumors from Medical Images Using YOLOv8 and DeiT. (arXiv:2401.03302v1 [eess.IV])

    [http://arxiv.org/abs/2401.03302](http://arxiv.org/abs/2401.03302)

    本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤，并解决了在罕见情况下的肿瘤检测问题。研究使用了来自国家脑映射实验室的数据集，通过修改样本数量和患者分布，使模型能够应对真实世界场景中的异常情况。

    

    在医学科学领域，由于脑肿瘤在患者中的罕见程度，可靠地检测和分类脑肿瘤仍然是一个艰巨的挑战。因此，在异常情况下检测肿瘤的能力对于确保及时干预和改善患者结果至关重要。本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤。来自国家脑映射实验室（NBML）的精选数据集包括81名患者，其中包括30例肿瘤病例和51例正常病例。检测和分类流程被分为两个连续的任务。检测阶段包括全面的数据分析和预处理，以修改图像样本和每个类别的患者数量，以符合真实世界场景中的异常分布（9个正常样本对应1个肿瘤样本）。此外，在测试中除了常见的评估指标外，我们还采用了... [摘要长度已达到上限]

    In the field of medical sciences, reliable detection and classification of brain tumors from images remains a formidable challenge due to the rarity of tumors within the population of patients. Therefore, the ability to detect tumors in anomaly scenarios is paramount for ensuring timely interventions and improved patient outcomes. This study addresses the issue by leveraging deep learning (DL) techniques to detect and classify brain tumors in challenging situations. The curated data set from the National Brain Mapping Lab (NBML) comprises 81 patients, including 30 Tumor cases and 51 Normal cases. The detection and classification pipelines are separated into two consecutive tasks. The detection phase involved comprehensive data analysis and pre-processing to modify the number of image samples and the number of patients of each class to anomaly distribution (9 Normal per 1 Tumor) to comply with real world scenarios. Next, in addition to common evaluation metrics for the testing, we emplo
    
[^6]: 夫妻可以被解决：新的算法和对带夫妻的医院/居民问题的难度结果

    Couples can be tractable: New algorithms and hardness results for the Hospitals / Residents problem with Couples. (arXiv:2311.00405v1 [cs.DS])

    [http://arxiv.org/abs/2311.00405](http://arxiv.org/abs/2311.00405)

    本研究提出了一种新的多项式时间算法，用于解决带夫妻的医院/居民问题，并证明了算法的多项式时间可解性以及对稳定B匹配问题的应用。算法能够找到一个接近可行的稳定匹配，并应用于不同类型的实例。

    

    在本文中，我们研究了带夫妻的医院/居民问题（Hospitals / Residents problem with Couples，简称HRC），其中解的一种是稳定匹配，或者报告不存在。我们提出了一种新的多项式时间算法，可以在带夫妻的HRC实例中找到一个接近可行的稳定匹配（通过最多调整医院的容量1个单位），其中夫妻的偏好是子响应性的（即，如果一个成员转移到一个更好的医院，那么夫妻也会变得更好）和子完备性的（即，对于每对个别可接受的医院都是夫妻一起可接受的），通过将其规约到稳定固定问题的实例。我们还提出了一个针对HRC子响应性、子完备实例的多项式时间算法，该实例是一个双重市场，或者所有夫妻属于几种可能类型之一。我们证明了我们的算法也意味着稳定B匹配问题的多项式时间可解性，其中底层图是带有环的多重图。

    In this paper we study the {\sc Hospitals / Residents problem with Couples} ({\sc hrc}), where a solution is a stable matching or a report that none exists. We present a novel polynomial-time algorithm that can find a near-feasible stable matching (adjusting the hospitals' capacities by at most 1) in an {\sc hrc} instance where the couples' preferences are sub-responsive (i.e., if one member switches to a better hospital, than the couple also improves) and sub-complete (i.e., each pair of hospitals that are individually acceptable to both members are jointly acceptable for the couple) by reducing it to an instance of the {\sc Stable Fixtures} problem. We also present a polynomial-time algorithm for {\sc hrc} in a sub-responsive, sub-complete instance that is a Dual Market, or where all couples are one of several possible types. We show that our algorithm also implies the polynomial-time solvability of a stable b-matching problem, where the underlying graph is a multigraph with loops.  
    
[^7]: 非光滑弱凸有限和耦合组合优化

    Non-Smooth Weakly-Convex Finite-sum Coupled Compositional Optimization. (arXiv:2310.03234v1 [math.OC])

    [http://arxiv.org/abs/2310.03234](http://arxiv.org/abs/2310.03234)

    本文研究了一种新的组合优化问题，称为非光滑弱凸有限和耦合组合优化(NSWC FCCO)，通过扩展已有的研究，我们研究了非光滑弱凸FCCO的问题，并提出了一种单循环算法来找到Moreau环的ε-稳定点。

    

    本文研究了一类新的组合优化问题，称为非光滑弱凸有限和耦合组合优化(NSWC FCCO)。由于其在机器学习和人工智能领域的广泛应用以及其解决基于经验风险最小化的随机算法的局限性，FCCO引起了越来越多的关注。然而，目前对于FCCO的研究假设内外函数都是光滑的，限制了其能够解决更多种类的问题的潜力。我们的研究从非光滑弱凸FCCO的角度进行了扩展，其中外函数是弱凸且非递减的，内函数是弱凸的。我们分析了一种单循环算法，并确定其在找到Moreau环的ε-稳定点的复杂度。

    This paper investigates new families of compositional optimization problems, called $\underline{\bf n}$on-$\underline{\bf s}$mooth $\underline{\bf w}$eakly-$\underline{\bf c}$onvex $\underline{\bf f}$inite-sum $\underline{\bf c}$oupled $\underline{\bf c}$ompositional $\underline{\bf o}$ptimization (NSWC FCCO). There has been a growing interest in FCCO due to its wide-ranging applications in machine learning and AI, as well as its ability to address the shortcomings of stochastic algorithms based on empirical risk minimization. However, current research on FCCO presumes that both the inner and outer functions are smooth, limiting their potential to tackle a more diverse set of problems. Our research expands on this area by examining non-smooth weakly-convex FCCO, where the outer function is weakly convex and non-decreasing, and the inner function is weakly-convex. We analyze a single-loop algorithm and establish its complexity for finding an $\epsilon$-stationary point of the Moreau env
    
[^8]: WASA：基于水印的大型语言模型生成数据的源归属

    WASA: WAtermark-based Source Attribution for Large Language Model-Generated Data. (arXiv:2310.00646v1 [cs.LG])

    [http://arxiv.org/abs/2310.00646](http://arxiv.org/abs/2310.00646)

    本文提出了一种基于水印的框架WASA，通过允许大型语言模型生成带有嵌入源信息的合成文本水印来解决源归属和数据来源的问题。

    

    大型语言模型（LLM）的出色性能和其商业化的巨大潜力引发了对其训练数据知识产权（IP）的严重关注。特别是，LLM生成的合成文本可能侵犯被用于训练LLM的数据的知识产权。为此，我们能够（a）通过水印识别出对LLM生成的合成文本做出贡献的数据提供者（源归属）；以及（b）验证文本数据是否来自于某个数据提供者对LLM进行了训练（数据来源）。在本文中，我们展示了通过水印技术可以解决这两个问题，即通过让LLM生成具有嵌入源信息的合成文本水印来实现。我们确定了这种水印技术框架的关键特性（例如源归属准确性、抵抗对手攻击的鲁棒性），并提出了一个满足这些要求的WAtermarking for Source Attribution（WASA）框架.

    The impressive performances of large language models (LLMs) and their immense potential for commercialization have given rise to serious concerns over the intellectual property (IP) of their training data. In particular, the synthetic texts generated by LLMs may infringe the IP of the data being used to train the LLMs. To this end, it is imperative to be able to (a) identify the data provider who contributed to the generation of a synthetic text by an LLM (source attribution) and (b) verify whether the text data from a data provider has been used to train an LLM (data provenance). In this paper, we show that both problems can be solved by watermarking, i.e., by enabling an LLM to generate synthetic texts with embedded watermarks that contain information about their source(s). We identify the key properties of such watermarking frameworks (e.g., source attribution accuracy, robustness against adversaries), and propose a WAtermarking for Source Attribution (WASA) framework that satisfies
    
[^9]: 作为评估器的大型语言模型中认知偏差的基准测试

    Benchmarking Cognitive Biases in Large Language Models as Evaluators. (arXiv:2309.17012v1 [cs.CL])

    [http://arxiv.org/abs/2309.17012](http://arxiv.org/abs/2309.17012)

    本研究对15个不同大小的大型语言模型进行了评估，发现它们作为评估器存在认知偏差，尤其在文本质量评估中表现出较强的偏见，这对其鲁棒性提出了质疑。同时，研究还发现了人类和机器偏好之间的相关性。

    

    最近的研究表明，大型语言模型（LLMs）通过简单的提示和上下文学习作为自动评估器非常有效。本研究组装了15个大小不同的LLMs，并通过其他LLMs的偏好排名来评估它们的输出响应，例如System Star比System Square更好。然后，我们引入了用于评估LLMs输出中六种不同认知偏差的认知偏差基准测试（CoBBLEr），如自我中心偏差，即模型更喜欢将自己的输出在评估中排名较高。我们发现LLMs是有偏见的文本质量评估器，在每个评估中都表现出对我们偏见基准的强烈迹象（在所有模型上的平均比较约为40%），这对它们作为评估器的鲁棒性提出了质疑。此外，我们还研究了人类和机器偏好之间的相关性，并计算了平均的Rank-Biased O值。

    Large Language Models (LLMs) have recently been shown to be effective as automatic evaluators with simple prompting and in-context learning. In this work, we assemble 15 LLMs of four different size ranges and evaluate their output responses by preference ranking from the other LLMs as evaluators, such as System Star is better than System Square. We then evaluate the quality of ranking outputs introducing the Cognitive Bias Benchmark for LLMs as Evaluators (CoBBLEr), a benchmark to measure six different cognitive biases in LLM evaluation outputs, such as the Egocentric bias where a model prefers to rank its own outputs highly in evaluation. We find that LLMs are biased text quality evaluators, exhibiting strong indications on our bias benchmark (average of 40% of comparisons across all models) within each of their evaluations that question their robustness as evaluators. Furthermore, we examine the correlation between human and machine preferences and calculate the average Rank-Biased O
    
[^10]: TempFuser: 使用长短时序融合转换器学习空中格斗中的战术和敏捷飞行动作

    TempFuser: Learning Tactical and Agile Flight Maneuvers in Aerial Dogfights using a Long Short-Term Temporal Fusion Transformer. (arXiv:2308.03257v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2308.03257](http://arxiv.org/abs/2308.03257)

    TempFuser是一种长短时序融合转换器，能够学习空中格斗中的战术和敏捷飞行动作。经过训练，模型成功地学会了复杂的战斗动作，并在面对高级对手时展现出人类一样的战术动作。

    

    在空中战斗中，空战动作对战术机动和敏捷战斗机的空气动力学都提出了复杂的挑战。在本文中，我们介绍了TempFuser，一种新颖的长短时序融合转换器，旨在学习空中格斗中的战术和敏捷飞行动作。我们的方法利用两种不同的基于LSTM的输入嵌入来编码长期稀疏和短期密集的状态表示。通过将这些嵌入通过转换器编码器进行整合，我们的模型捕获了战斗机的战术和敏捷性，使其能够生成端到端的飞行指令，确保占据优势位置并超越对手。经过对高保真飞行模拟器中多种类型对手飞机的广泛训练，我们的模型成功地学习了执行复杂的战斗动作，且始终表现优于多个基准模型。值得注意的是，我们的模型在面对高级对手时展现出人类一样的战术动作。

    In aerial combat, dogfighting poses intricate challenges that demand an understanding of both strategic maneuvers and the aerodynamics of agile fighter aircraft. In this paper, we introduce TempFuser, a novel long short-term temporal fusion transformer designed to learn tactical and agile flight maneuvers in aerial dogfights. Our approach employs two distinct LSTM-based input embeddings to encode long-term sparse and short-term dense state representations. By integrating these embeddings through a transformer encoder, our model captures the tactics and agility of fighter jets, enabling it to generate end-to-end flight commands that secure dominant positions and outmaneuver the opponent. After extensive training against various types of opponent aircraft in a high-fidelity flight simulator, our model successfully learns to perform complex fighter maneuvers, consistently outperforming several baseline models. Notably, our model exhibits human-like strategic maneuvers even when facing adv
    
[^11]: Etsy搜索中统一嵌入式个性化检索的方法

    Unified Embedding Based Personalized Retrieval in Etsy Search. (arXiv:2306.04833v1 [cs.IR])

    [http://arxiv.org/abs/2306.04833](http://arxiv.org/abs/2306.04833)

    本论文提出了一种将图形、转换和基于术语的嵌入结合起来的统一嵌入模型，并利用端到端训练模型进行个性化检索，以解决Etsy搜索中的语义差距问题。同时，本文分享了特征工程、硬负采样策略和应用变压器模型的新策略，以构建具有工业规模的模型来改善整体搜索体验。

    

    基于嵌入式神经网络的信息检索已经成为解决尾查询中经常出现的语义差距问题的普遍方法。与此同时，热门查询通常缺乏上下文，有广泛的意图，用户历史互动的附加上下文有助于解决问题。本文介绍了我们解决语义差距问题的新方法，以及一种用于个性化语义检索的端到端训练模型。我们建议学习一种统一的嵌入模型，包括基于图形、变压器和术语的嵌入，同时分享了我们的设计选择，以在性能和效率之间实现最佳权衡。我们分享了特征工程、硬负采样策略和变压器模型的应用方面的经验教训，包括用于提高搜索相关性和部署此类模型的一种新颖的预训练策略和其他技巧。我们的个性化检索模型显着提高了整体搜索体验。

    Embedding-based neural retrieval is a prevalent approach to address the semantic gap problem which often arises in product search on tail queries. In contrast, popular queries typically lack context and have a broad intent where additional context from users historical interaction can be helpful. In this paper, we share our novel approach to address both: the semantic gap problem followed by an end to end trained model for personalized semantic retrieval. We propose learning a unified embedding model incorporating graph, transformer and term-based embeddings end to end and share our design choices for optimal tradeoff between performance and efficiency. We share our learnings in feature engineering, hard negative sampling strategy, and application of transformer model, including a novel pre-training strategy and other tricks for improving search relevance and deploying such a model at industry scale. Our personalized retrieval model significantly improves the overall search experience,
    
[^12]: 一种可解释的机器学习系统来识别癫痫-间隙-损伤连续状态下的脑电图图案

    An Interpretable Machine Learning System to Identify EEG Patterns on the Ictal-Interictal-Injury Continuum. (arXiv:2211.05207v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.05207](http://arxiv.org/abs/2211.05207)

    该论文设计了一种可解释的深度学习模型，以预测ICU脑电监测中常见的6种脑波图案的存在，并提供高质量的解释和三种解释方法，这对于建立AI的信任和临床采用至关重要。

    

    在许多医学领域，人们呼吁在用于临床工作的机器学习系统中增加可解释性。在本文中，我们设计了一个可解释的深度学习模型，用于预测ICU脑电监测中常见的6种脑波图案（癫痫、LPD、GPD、LRDA、GRDA、其他）的存在。每个预测都配有一个高质量的解释，借助于专门的用户界面提供支持。此新型模型架构学习了一组原型示例（“原型”），并通过将新的EEG片段与这些原型进行比较来做出决策。这些原型可以是单类（仅与一个类相关）或双类（与两个类相关）。我们提出了三种主要的模型解释方法：1）使用全局结构保持方法，将1275维cEEG潜在特征映射到二维空间中，可视化癫痫-间隙-损伤连续状态，从而深入了解其高维结构。2）我们提出了一种交互式解释方法，使人类专家能够查询模型预测的不同方面，并以自然语言接收经过专家验证的解释。3）我们可视化了导致模型做出某个决策的输入的最重要特征，允许详细检查输入和输出之间的关系。总的来说，我们展示了解释性模型分类EEG图案和提供专家友好的解释的实用性，这两个方面对于建立AI的信任和临床采用至关重要。

    In many medical subfields, there is a call for greater interpretability in the machine learning systems used for clinical work. In this paper, we design an interpretable deep learning model to predict the presence of 6 types of brainwave patterns (Seizure, LPD, GPD, LRDA, GRDA, other) commonly encountered in ICU EEG monitoring. Each prediction is accompanied by a high-quality explanation delivered with the assistance of a specialized user interface. This novel model architecture learns a set of prototypical examples (``prototypes'') and makes decisions by comparing a new EEG segment to these prototypes. These prototypes are either single-class (affiliated with only one class) or dual-class (affiliated with two classes).  We present three main ways of interpreting the model: 1) Using global-structure preserving methods, we map the 1275-dimensional cEEG latent features to a 2D space to visualize the ictal-interictal-injury continuum and gain insight into its high-dimensional structure. 2
    


# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DUMA: a Dual-Mind Conversational Agent with Fast and Slow Thinking.](http://arxiv.org/abs/2310.18075) | DUMA是一种具有快速和慢速思考能力的双重思维对话代理框架，通过利用两个生成型大型语言模型，实现了根据情况在直观响应和深思熟虑的问题解决过程之间无缝切换的能力。 |
| [^2] | [AutoDiff: combining Auto-encoder and Diffusion model for tabular data synthesizing.](http://arxiv.org/abs/2310.15479) | 使用自动编码器和扩散模型结合的AutoDiff模型可以有效地生成合成的表格数据，克服了表格数据中的异构特征和特征间相关性的挑战，生成的数据与真实数据在统计上非常相似，并在机器学习任务中表现良好。 |
| [^3] | [Random Forest Dissimilarity for High-Dimension Low Sample Size Classification.](http://arxiv.org/abs/2310.14710) | 针对高维度低样本(HDLSS)分类问题，本论文提出了一种基于随机森林相似度的学习预计算SVM核方法(RFSVM)，通过在40个公共HDLSS分类数据集上的实验证明，该方法在HDLSS问题上优于现有方法并且保持了非常一致的结果。 |
| [^4] | [Uncertainty-Aware Decision Transformer for Stochastic Driving Environments.](http://arxiv.org/abs/2309.16397) | 本论文提出了一种针对随机驾驶环境的不确定性感知决策Transformer（UNREST），通过估计状态的不确定性并相应地分割序列，取代了全局回报。这项工作通过解决在不确定性环境中过于乐观的问题，为离线强化学习在自主驾驶任务中的应用提供了一种有效的解决方案。 |
| [^5] | [VisIT-Bench: A Benchmark for Vision-Language Instruction Following Inspired by Real-World Use.](http://arxiv.org/abs/2308.06595) | VisIT-Bench是一个用于评价真实世界中视觉语言模型指示遵循的基准，包含了各种任务并提供了详细描述，可以自动评估多模态生成的质量差距。 |
| [^6] | [SHAMSUL: Simultaneous Heatmap-Analysis to investigate Medical Significance Utilizing Local interpretability methods.](http://arxiv.org/abs/2307.08003) | 本研究利用四种解释性方法探索深度神经网络在医学领域的可解释性，通过热图分析解释神经网络的预测结果，并针对特定病理类别进行定量和定性研究。 |
| [^7] | [DUET: 2D Structured and Approximately Equivariant Representations.](http://arxiv.org/abs/2306.16058) | DUET是一种2D结构化且近似等变表示方法，相比于其他方法，可以在保留输入变换信息的同时具有更好的可控性和更高的准确性。 |
| [^8] | [Sasha: creative goal-oriented reasoning in smart homes with large language models.](http://arxiv.org/abs/2305.09802) | 本论文研究了在智能家居中使用大型语言模型实现用户命令的创意目标导向推理。实验结果显示，这种方法可以创造性地推理，以实现挑战性的目标。 |
| [^9] | [Language Models can Solve Computer Tasks.](http://arxiv.org/abs/2303.17491) | 本文研究表明，预训练的大型语言模型代理可以通过一个简单的提示方案使用自然语言执行计算机任务，该方法取得了很好的效果并在MiniWoB++基准测试中超越了监督学习和强化学习方法。 |
| [^10] | [Understanding and Exploring the Whole Set of Good Sparse Generalized Additive Models.](http://arxiv.org/abs/2303.16047) | 提出一种有效而准确地近似稀疏广义可加模型（GAMs）的Rashomon集的技术，并使用这些近似模型来解决实际应用的挑战。 |

# 详细

[^1]: DUMA：具有快速和慢速思考能力的双重思维对话代理

    DUMA: a Dual-Mind Conversational Agent with Fast and Slow Thinking. (arXiv:2310.18075v1 [cs.CL])

    [http://arxiv.org/abs/2310.18075](http://arxiv.org/abs/2310.18075)

    DUMA是一种具有快速和慢速思考能力的双重思维对话代理框架，通过利用两个生成型大型语言模型，实现了根据情况在直观响应和深思熟虑的问题解决过程之间无缝切换的能力。

    

    受到人类认知的双过程理论启发，我们介绍了DUMA，一种新颖的对话代理框架，通过利用两个用于快速和慢速思考的生成型大型语言模型（LLM），体现了双重思维机制。快速思考模型作为主要接口用于外部交互和初始响应生成，根据完整响应的复杂性评估是否需要调用慢速思考模型。一旦被调用，慢速思考模型接管对话，在细致规划、推理和工具利用方面进行工作，提供经过充分分析的响应。这种双重思维配置允许根据情况在直观响应和深思熟虑的问题解决过程之间无缝切换。我们构建了一个用于处理房地产行业在线咨询的对话代理。实验证明，我们的方法在效果和效率之间取得了平衡。

    Inspired by the dual-process theory of human cognition, we introduce DUMA, a novel conversational agent framework that embodies a dual-mind mechanism through the utilization of two generative Large Language Models (LLMs) dedicated to fast and slow thinking respectively. The fast thinking model serves as the primary interface for external interactions and initial response generation, evaluating the necessity for engaging the slow thinking model based on the complexity of the complete response. When invoked, the slow thinking model takes over the conversation, engaging in meticulous planning, reasoning, and tool utilization to provide a well-analyzed response. This dual-mind configuration allows for a seamless transition between intuitive responses and deliberate problem-solving processes based on the situation. We have constructed a conversational agent to handle online inquiries in the real estate industry. The experiment proves that our method balances effectiveness and efficiency, an
    
[^2]: AutoDiff:结合自动编码器和扩散模型用于表格数据合成

    AutoDiff: combining Auto-encoder and Diffusion model for tabular data synthesizing. (arXiv:2310.15479v1 [stat.ML])

    [http://arxiv.org/abs/2310.15479](http://arxiv.org/abs/2310.15479)

    使用自动编码器和扩散模型结合的AutoDiff模型可以有效地生成合成的表格数据，克服了表格数据中的异构特征和特征间相关性的挑战，生成的数据与真实数据在统计上非常相似，并在机器学习任务中表现良好。

    

    扩散模型已成为现代机器学习许多子领域中合成数据生成的主要范式，包括计算机视觉、语言模型或语音合成。在本文中，我们利用扩散模型的力量来生成合成的表格数据。表格数据中的异构特征一直是表格数据合成的主要障碍，我们通过使用自动编码器的架构来解决这个问题。与最先进的表格合成器相比，我们模型生成的合成表格在统计上与真实数据非常相似，并在机器学习工具的下游任务中表现良好。我们在15个公开可用的数据集上进行了实验。值得注意的是，我们的模型灵活地捕捉了特征之间的相关性，这是表格数据合成中长期存在的挑战。如若接纳了论文，我们的代码将根据要求提供，并且将公开发布。

    Diffusion model has become a main paradigm for synthetic data generation in many subfields of modern machine learning, including computer vision, language model, or speech synthesis. In this paper, we leverage the power of diffusion model for generating synthetic tabular data. The heterogeneous features in tabular data have been main obstacles in tabular data synthesis, and we tackle this problem by employing the auto-encoder architecture. When compared with the state-of-the-art tabular synthesizers, the resulting synthetic tables from our model show nice statistical fidelities to the real data, and perform well in downstream tasks for machine learning utilities. We conducted the experiments over 15 publicly available datasets. Notably, our model adeptly captures the correlations among features, which has been a long-standing challenge in tabular data synthesis. Our code is available upon request and will be publicly released if paper is accepted.
    
[^3]: 高维度低样本分类的随机森林差异性

    Random Forest Dissimilarity for High-Dimension Low Sample Size Classification. (arXiv:2310.14710v1 [stat.ML])

    [http://arxiv.org/abs/2310.14710](http://arxiv.org/abs/2310.14710)

    针对高维度低样本(HDLSS)分类问题，本论文提出了一种基于随机森林相似度的学习预计算SVM核方法(RFSVM)，通过在40个公共HDLSS分类数据集上的实验证明，该方法在HDLSS问题上优于现有方法并且保持了非常一致的结果。

    

    高维度低样本(HDLSS)问题在机器学习的实际应用中很常见。从医学影像到文本处理，传统的机器学习算法通常无法从这样的数据中学习到最佳的概念。我们在之前的工作中提出了一种基于差异性的多视角分类方法，即随机森林差异性(RFD)，该方法在这类问题上取得了最先进的结果。在本研究中，我们将该方法的核心原则转化为解决HDLSS分类问题的方法，通过使用随机森林相似度作为学习的预计算SVM核(RFSVM)。我们展示了这样的学习相似度度量在这种分类上特别适用和准确。通过对40个公共HDLSS分类数据集进行的实验，配合严格的统计分析，结果显示RFSVM方法在大多数HDLSS问题上优于现有方法，并且同时非常连贯。

    High dimension, low sample size (HDLSS) problems are numerous among real-world applications of machine learning. From medical images to text processing, traditional machine learning algorithms are usually unsuccessful in learning the best possible concept from such data. In a previous work, we proposed a dissimilarity-based approach for multi-view classification, the Random Forest Dissimilarity (RFD), that perfoms state-of-the-art results for such problems. In this work, we transpose the core principle of this approach to solving HDLSS classification problems, by using the RF similarity measure as a learned precomputed SVM kernel (RFSVM). We show that such a learned similarity measure is particularly suited and accurate for this classification context. Experiments conducted on 40 public HDLSS classification datasets, supported by rigorous statistical analyses, show that the RFSVM method outperforms existing methods for the majority of HDLSS problems and remains at the same time very co
    
[^4]: 针对随机驾驶环境的不确定性感知决策Transformer

    Uncertainty-Aware Decision Transformer for Stochastic Driving Environments. (arXiv:2309.16397v1 [cs.LG])

    [http://arxiv.org/abs/2309.16397](http://arxiv.org/abs/2309.16397)

    本论文提出了一种针对随机驾驶环境的不确定性感知决策Transformer（UNREST），通过估计状态的不确定性并相应地分割序列，取代了全局回报。这项工作通过解决在不确定性环境中过于乐观的问题，为离线强化学习在自主驾驶任务中的应用提供了一种有效的解决方案。

    

    离线强化学习（RL）已经成为一种无需主动交互的学习策略的有希望框架，因此在自主驾驶任务中尤其吸引人。最近Transformers的成功启发了将离线RL视为序列建模，这在长期任务中表现出色。然而，在具有不确定性的环境中，它们过于乐观，错误地假设相同的目标可以通过相同的动作一致实现。在本文中，我们引入了一种针对随机驾驶环境的不确定性感知决策Transformer（UNREST），不引入额外的转换模型或复杂的生成模型来进行规划。具体而言，UNREST通过转换与回报之间的条件互信息来估计状态的不确定性，并相应地分割序列。通过发现驾驶环境的“不确定性累积”和“时间局部性”特性，UNREST将决策Transformer中的全局回报替换为较少的部分回报。

    Offline Reinforcement Learning (RL) has emerged as a promising framework for learning policies without active interactions, making it especially appealing for autonomous driving tasks. Recent successes of Transformers inspire casting offline RL as sequence modeling, which performs well in long-horizon tasks. However, they are overly optimistic in stochastic environments with incorrect assumptions that the same goal can be consistently achieved by identical actions. In this paper, we introduce an UNcertainty-awaRE deciSion Transformer (UNREST) for planning in stochastic driving environments without introducing additional transition or complex generative models. Specifically, UNREST estimates state uncertainties by the conditional mutual information between transitions and returns, and segments sequences accordingly. Discovering the `uncertainty accumulation' and `temporal locality' properties of driving environments, UNREST replaces the global returns in decision transformers with less 
    
[^5]: VisIT-Bench: 一个受真实世界使用启发的视觉语言指示评估基准

    VisIT-Bench: A Benchmark for Vision-Language Instruction Following Inspired by Real-World Use. (arXiv:2308.06595v1 [cs.CL])

    [http://arxiv.org/abs/2308.06595](http://arxiv.org/abs/2308.06595)

    VisIT-Bench是一个用于评价真实世界中视觉语言模型指示遵循的基准，包含了各种任务并提供了详细描述，可以自动评估多模态生成的质量差距。

    

    我们引入了VisIT-Bench（Visual InsTruction Benchmark），这是一个评价用于真实世界使用的视觉语言模型的指示遵循基准。我们的起点是策划了70个“指示家族”，我们认为指示调优的视觉语言模型应该能够解决这些家族。任务不仅限于VQAv2和COCO等评估，涵盖了从基本识别到游戏玩法和创造性生成的各种任务。在策划之后，我们的数据集包括592个测试查询，每个查询都带有一个人工编写的指示条件化的字幕。这些描述展现了特定指示因素，例如对于询问店面对于轮椅用户的易访问性的指示，条件化的字幕描述了斜坡/潜在障碍物。这些描述使得我们可以：1）收集每个实例的人工验证的参考输出；2）使用仅文本的语言模型对候选多模态生成进行自动评估，与人类判断相一致。我们量化了质量差距。

    We introduce VisIT-Bench (Visual InsTruction Benchmark), a benchmark for evaluation of instruction-following vision-language models for real-world use. Our starting point is curating 70 'instruction families' that we envision instruction tuned vision-language models should be able to address. Extending beyond evaluations like VQAv2 and COCO, tasks range from basic recognition to game playing and creative generation. Following curation, our dataset comprises 592 test queries, each with a human-authored instruction-conditioned caption. These descriptions surface instruction-specific factors, e.g., for an instruction asking about the accessibility of a storefront for wheelchair users, the instruction-conditioned caption describes ramps/potential obstacles. These descriptions enable 1) collecting human-verified reference outputs for each instance; and 2) automatic evaluation of candidate multimodal generations using a text-only LLM, aligning with human judgment. We quantify quality gaps be
    
[^6]: SHAMSUL: 利用本地可解释性方法进行同时热图分析以研究医学意义

    SHAMSUL: Simultaneous Heatmap-Analysis to investigate Medical Significance Utilizing Local interpretability methods. (arXiv:2307.08003v1 [eess.IV])

    [http://arxiv.org/abs/2307.08003](http://arxiv.org/abs/2307.08003)

    本研究利用四种解释性方法探索深度神经网络在医学领域的可解释性，通过热图分析解释神经网络的预测结果，并针对特定病理类别进行定量和定性研究。

    

    深度神经网络的可解释性已成为医疗和健康领域的一个热门议题。这种关注来源于对透明度、法律和伦理考虑以及这些深度神经网络在临床决策支持系统中生成的预测的医学意义的担忧。为了解决这个问题，我们的研究深入探讨了四种已建立的解释性方法: 局部可解释模型无关解释(LIME)、Shapley加性解释(SHAP)、梯度加权类别激活映射(Grad-CAM)和层内相关传播(LRP)。我们利用多标签多类胸部放射学数据集的迁移学习方法，旨在解释与特定病理类别相关的预测。我们的分析涵盖了单标签和多标签预测，通过定量和定性研究提供了全面和公正的评估，

    The interpretability of deep neural networks has become a subject of great interest within the medical and healthcare domain. This attention stems from concerns regarding transparency, legal and ethical considerations, and the medical significance of predictions generated by these deep neural networks in clinical decision support systems. To address this matter, our study delves into the application of four well-established interpretability methods: Local Interpretable Model-agnostic Explanations (LIME), Shapley Additive exPlanations (SHAP), Gradient-weighted Class Activation Mapping (Grad-CAM), and Layer-wise Relevance Propagation (LRP). Leveraging the approach of transfer learning with a multi-label-multi-class chest radiography dataset, we aim to interpret predictions pertaining to specific pathology classes. Our analysis encompasses both single-label and multi-label predictions, providing a comprehensive and unbiased assessment through quantitative and qualitative investigations, w
    
[^7]: DUET: 2D结构化且近似等变表示

    DUET: 2D Structured and Approximately Equivariant Representations. (arXiv:2306.16058v1 [cs.LG])

    [http://arxiv.org/abs/2306.16058](http://arxiv.org/abs/2306.16058)

    DUET是一种2D结构化且近似等变表示方法，相比于其他方法，可以在保留输入变换信息的同时具有更好的可控性和更高的准确性。

    

    多视图自监督学习(MSSL)基于学习相对于一组输入变换的不变性。然而，不变性从表示中部分或完全移除与变换相关的信息，这可能对需要这些信息的特定下游任务的性能造成损害。我们提出了2D结构化和等变表示，称为DUET，它们是以矩阵结构组织的2D表示，并且对作用于输入数据的变换具有等变性。DUET表示保留有关输入变换的信息，同时保持语义表达能力。与SimCLR（Chen等，2020）（无结构和不变性）和ESSL（Dangovski等，2022）（无结构和等变性）相比，DUET表示的结构化和等变性使得生成具有更低的重建误差的可控性成为可能，而SimCLR或ESSL则无法实现可控性。DUET还实现了更高的准确性。

    Multiview Self-Supervised Learning (MSSL) is based on learning invariances with respect to a set of input transformations. However, invariance partially or totally removes transformation-related information from the representations, which might harm performance for specific downstream tasks that require such information. We propose 2D strUctured and EquivarianT representations (coined DUET), which are 2d representations organized in a matrix structure, and equivariant with respect to transformations acting on the input data. DUET representations maintain information about an input transformation, while remaining semantically expressive. Compared to SimCLR (Chen et al., 2020) (unstructured and invariant) and ESSL (Dangovski et al., 2022) (unstructured and equivariant), the structured and equivariant nature of DUET representations enables controlled generation with lower reconstruction error, while controllability is not possible with SimCLR or ESSL. DUET also achieves higher accuracy fo
    
[^8]: Sasha: 大型语言模型在智能家居中的创意目标导向推理

    Sasha: creative goal-oriented reasoning in smart homes with large language models. (arXiv:2305.09802v1 [cs.HC])

    [http://arxiv.org/abs/2305.09802](http://arxiv.org/abs/2305.09802)

    本论文研究了在智能家居中使用大型语言模型实现用户命令的创意目标导向推理。实验结果显示，这种方法可以创造性地推理，以实现挑战性的目标。

    

    智能家居的使用者与设备的交互都有明确或隐含的目标。现有的家庭助手能够轻松地实现明确的目标，例如"打开灯"。然而，在更自然的交流中，人们往往会描述隐含的目标。例如，我们可以让别人"让房间变得舒适"而不是描述具体的步骤。当前的系统很难解决这种歧义，因为需要将模糊的意图与具体的设备联系起来。我们从通用大型语言模型（LLMs）的角度来解决这个问题，这些模型经过大量语料库的训练，并通过灵活的方式适应下游任务。我们探讨了使用LLMs控制设备并创建自动化程序以满足用户命令的隐含目标。在以用户为中心的研究中，我们发现LLMs可以创造性地推理，以实现挑战性的目标，同时也揭示了降低其有用性的差距。我们使用Sasha解决了这些差距：这是一个在智能家居中使用LLMs进行灵活解释用户命令和设备控制的创意目标导向推理系统。

    Every smart home user interaction has an explicit or implicit goal. Existing home assistants easily achieve explicit goals, e.g., "turn on the light". In more natural communication, however, humans tend to describe implicit goals. We can, for example, ask someone to "make it cozy" rather than describe the specific steps involved. Current systems struggle with this ambiguity since it requires them to relate vague intent to specific devices. We approach this problem of flexibly achieving user goals from the perspective of general-purpose large language models (LLMs) trained on gigantic corpora and adapted to downstream tasks with remarkable flexibility. We explore the use of LLMs for controlling devices and creating automation routines to meet the implicit goals of user commands. In a user-focused study, we find that LLMs can reason creatively to achieve challenging goals, while also revealing gaps that diminish their usefulness. We address these gaps with Sasha: a system for creative, g
    
[^9]: 语言模型能够解决计算机任务

    Language Models can Solve Computer Tasks. (arXiv:2303.17491v1 [cs.CL])

    [http://arxiv.org/abs/2303.17491](http://arxiv.org/abs/2303.17491)

    本文研究表明，预训练的大型语言模型代理可以通过一个简单的提示方案使用自然语言执行计算机任务，该方法取得了很好的效果并在MiniWoB++基准测试中超越了监督学习和强化学习方法。

    

    能够在计算机上执行通用任务的代理可以通过自动化重复任务和协助复杂问题的解决来提高效率和生产力。理想情况下，这些代理应该能够通过自然语言命令解决新的计算机任务。然而，先前解决这个问题的方法需要大量专家示范和任务特定的奖励函数，这两者对于新任务来说都不切实际。在这项工作中，我们展示了一个预先训练的大型语言模型（LLM）代理可以使用一个简单的提示方案（RCI），通过自然语言指导执行计算机任务，并在批评和改进输出的过程中取得很好的效果。RCI方法在自动化计算机任务方面明显优于现有的LLM方法，并在MiniWoB++基准测试中超越了监督学习（SL）和强化学习（RL）方法。RCI方法使用每个任务仅有的少数示范，与最新的SL+RL方法相竞争。

    Agents capable of carrying out general tasks on a computer can improve efficiency and productivity by automating repetitive tasks and assisting in complex problem-solving. Ideally, such agents should be able to solve new computer tasks presented to them through natural language commands. However, previous approaches to this problem require large amounts of expert demonstrations and task-specific reward functions, both of which are impractical for new tasks. In this work, we show that a pre-trained large language model (LLM) agent can execute computer tasks guided by natural language using a simple prompting scheme where the agent recursively criticizes and improves its output (RCI). The RCI approach significantly outperforms existing LLM methods for automating computer tasks and surpasses supervised learning (SL) and reinforcement learning (RL) approaches on the MiniWoB++ benchmark. RCI is competitive with the state-of-the-art SL+RL method, using only a handful of demonstrations per ta
    
[^10]: 理解和探索稀疏广义可加模型的整个优秀集合

    Understanding and Exploring the Whole Set of Good Sparse Generalized Additive Models. (arXiv:2303.16047v1 [cs.LG])

    [http://arxiv.org/abs/2303.16047](http://arxiv.org/abs/2303.16047)

    提出一种有效而准确地近似稀疏广义可加模型（GAMs）的Rashomon集的技术，并使用这些近似模型来解决实际应用的挑战。

    

    在实际应用中，机器学习模型与领域专家之间的交互至关重要；然而，通常只生成单个模型的经典机器学习范式不利于此类交互。近似和探索Rashomon集，即所有近乎最优模型的集合，通过提供用户可搜索的空间包含多样性模型的方法，解决了这一实际挑战，领域专家可以从中选择。我们提出了一种有效而准确地近似稀疏广义可加模型（GAMs）的Rashomon集的技术。我们提供了用于近似具有固定支持集的GAMs的Rashomon集的椭球形算法，并使用这些椭球形近似了许多不同支持集的Rashomon集。近似的Rashomon集为解决实际挑战，例如（1）研究模型类的变量重要性；（2）在用户指定约束条件下查找模型，提供了重要的基础。

    In real applications, interaction between machine learning model and domain experts is critical; however, the classical machine learning paradigm that usually produces only a single model does not facilitate such interaction. Approximating and exploring the Rashomon set, i.e., the set of all near-optimal models, addresses this practical challenge by providing the user with a searchable space containing a diverse set of models from which domain experts can choose. We present a technique to efficiently and accurately approximate the Rashomon set of sparse, generalized additive models (GAMs). We present algorithms to approximate the Rashomon set of GAMs with ellipsoids for fixed support sets and use these ellipsoids to approximate Rashomon sets for many different support sets. The approximated Rashomon set serves as a cornerstone to solve practical challenges such as (1) studying the variable importance for the model class; (2) finding models under user-specified constraints (monotonicity
    


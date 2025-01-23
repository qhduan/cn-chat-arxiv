# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [UrbanVLP: A Multi-Granularity Vision-Language Pre-Trained Foundation Model for Urban Indicator Prediction](https://arxiv.org/abs/2403.16831) | UrbanVLP是一种多粒度信息集成的视觉语言预训练模型，旨在克服目前城市指标预测中预训练模型的局限性，提高了可解释性和精度 |
| [^2] | [Learning To Guide Human Decision Makers With Vision-Language Models](https://arxiv.org/abs/2403.16501) | 提出了“学习指导”（LTG）框架，旨在解决专家可能过度依赖机器决策和面临无助于模型放弃的决策的问题。 |
| [^3] | [Cartoon Hallucinations Detection: Pose-aware In Context Visual Learning](https://arxiv.org/abs/2403.15048) | 该研究提出了一种用于检测由TTI模型生成的卡通角色图像中视觉幻觉的系统，通过结合姿势感知上下文视觉学习和视觉语言模型，利用RGB图像和姿势信息，实现了更准确的决策，显著提高了视觉幻觉的识别能力，推动了TTI模型在非照片真实领域的发展。 |
| [^4] | [Optimal Transport for Domain Adaptation through Gaussian Mixture Models](https://arxiv.org/abs/2403.13847) | 通过高斯混合模型进行域自适应的最优输运，可以实现源域和目标域混合成分之间的匹配，从而在失效诊断中取得最先进的性能。 |
| [^5] | [Can We Verify Step by Step for Incorrect Answer Detection?](https://arxiv.org/abs/2402.10528) | 通过推理链来预测大型语言模型输出的准确性，我们引入了一个新的基准R2PE，并提出了处理可辨识性评分（PDS）框架。 |
| [^6] | [Enhancing Textbook Question Answering Task with Large Language Models and Retrieval Augmented Generation](https://arxiv.org/abs/2402.05128) | 本论文通过引入检索增强生成（RAG）技术和利用迁移学习来处理长文本和提升推理能力，为教科书问答任务带来了显著的改进。 |
| [^7] | [The Boundaries of Tractability in Hierarchical Task Network Planning.](http://arxiv.org/abs/2401.14174) | 本论文研究了在层次化任务网络规划中三个经典问题的可计算边界，提供了在特定条件下将多项式时间可解性结果从原始任务网络推广到一般任务网络的方法，并且给出了这三个问题的参数化复杂度分析。 |
| [^8] | [DISCOUNT: Distributional Counterfactual Explanation With Optimal Transport.](http://arxiv.org/abs/2401.13112) | 本文提出了使用最优传输进行分布式对抗解释的方法DISCOUNT，将对抗解释的概念扩展到整个输入输出分布，并通过统计置信度来支撑这一方法。 |
| [^9] | [Learning to Mask and Permute Visual Tokens for Vision Transformer Pre-Training.](http://arxiv.org/abs/2306.07346) | 本论文提出了一种新的自监督预训练方法MaPeT，不同于现有的使用掩码图像模型的方法，该方法使用自回归和置换预测来捕获图像块内的依赖关系并减少数据噪声的影响，从而提高了下游任务的一致性。 |
| [^10] | [Stochastic Submodular Bandits with Delayed Composite Anonymous Bandit Feedback.](http://arxiv.org/abs/2303.13604) | 本论文研究了具有随机次模收益和全赌徒延迟反馈的组合多臂赌博机问题，研究了三种延迟反馈模型并导出了后悔上限。研究结果表明，算法能够在考虑延迟组合匿名反馈时胜过其他全赌徒方法。 |

# 详细

[^1]: UrbanVLP：用于城市指标预测的多粒度视觉语言预训练基础模型

    UrbanVLP: A Multi-Granularity Vision-Language Pre-Trained Foundation Model for Urban Indicator Prediction

    [https://arxiv.org/abs/2403.16831](https://arxiv.org/abs/2403.16831)

    UrbanVLP是一种多粒度信息集成的视觉语言预训练模型，旨在克服目前城市指标预测中预训练模型的局限性，提高了可解释性和精度

    

    城市指标预测旨在利用数据驱动方法推断不同城市景观中的社会经济指标。然而，目前流行的预训练模型，特别是依赖卫星图像的模型，面临着双重挑战。首先，仅集中在卫星数据中的宏观级别模式可能引入偏见，在微观级别缺乏细致的细节，例如某地的建筑细节。其次，预训练模型缺乏可解释性，限制了它们在提供城市规划透明证据方面的实用性。针对这些问题，本文设计了一种新颖的Vision-Language Pre-Trained Model（UrbanVLP）。我们的UrbanVLP无缝整合来自宏观（卫星）和微观（街景）级别的多粒度信息，克服了先前预训练模型的局限性。此外，它引入了自动生成文本和校准，提高了在下游应用中的可解释性。

    arXiv:2403.16831v1 Announce Type: cross  Abstract: Urban indicator prediction aims to infer socio-economic metrics in diverse urban landscapes using data-driven methods. However, prevalent pre-trained models, particularly those reliant on satellite imagery, face dual challenges. Firstly, concentrating solely on macro-level patterns from satellite data may introduce bias, lacking nuanced details at micro levels, such as architectural details at a place. Secondly, the lack of interpretability in pre-trained models limits their utility in providing transparent evidence for urban planning. In response to these issues, we devise a novel Vision-Language Pre-Trained Model (UrbanVLP) in this paper. Our UrbanVLP seamlessly integrates multi-granularity information from both macro (satellite) and micro (street-view) levels, overcoming the limitations of prior pre-trained models. Moreover, it introduces automatic text generation and calibration, elevating interpretability in downstream application
    
[^2]: 学习使用视觉-语言模型指导人类决策者

    Learning To Guide Human Decision Makers With Vision-Language Models

    [https://arxiv.org/abs/2403.16501](https://arxiv.org/abs/2403.16501)

    提出了“学习指导”（LTG）框架，旨在解决专家可能过度依赖机器决策和面临无助于模型放弃的决策的问题。

    

    越来越多的人对开发人工智能以协助人类进行高风险任务中的决策表现出兴趣，比如医学诊断，旨在提高决策质量和减少认知负担。主流方法是将专家与机器学习模型合作，将更安全的决策下放，让前者专注于需要他们关注的情况。然而，在高风险场景中，这种“责任分工”设置是不够的。

    arXiv:2403.16501v1 Announce Type: new  Abstract: There is increasing interest in developing AIs for assisting human decision making in \textit{high-stakes} tasks, such as medical diagnosis, for the purpose of improving decision quality and reducing cognitive strain.   %   Mainstream approaches team up an expert with a machine learning model to which safer decisions are offloaded, thus letting the former focus on cases that demand their attention.   %   This \textit{separation of responsibilities} setup, however, is inadequate for high-stakes scenarios. On the one hand, the expert may end up over-relying on the machine's decisions due to \textit{anchoring bias}, thus losing the human oversight that is increasingly being required by regulatory agencies to ensure trustworthy AI. On the other hand, the expert is left entirely unassisted on the (typically hardest) decisions on which the model abstained.   %   As a remedy, we introduce \textit{learning to guide} (LTG), an alternative framewo
    
[^3]: 卡通幻觉检测: 姿势感知上下文视觉学习

    Cartoon Hallucinations Detection: Pose-aware In Context Visual Learning

    [https://arxiv.org/abs/2403.15048](https://arxiv.org/abs/2403.15048)

    该研究提出了一种用于检测由TTI模型生成的卡通角色图像中视觉幻觉的系统，通过结合姿势感知上下文视觉学习和视觉语言模型，利用RGB图像和姿势信息，实现了更准确的决策，显著提高了视觉幻觉的识别能力，推动了TTI模型在非照片真实领域的发展。

    

    大规模文本到图像（TTI）模型已经成为各种生成领域中生成训练数据的常见方法。然而，视觉幻觉，尤其是在非照片真实风格如卡通人物中包含了感知上关键的缺陷，依然是一个令人担忧的问题。我们提出了一种新颖的用于检测TTI模型生成的卡通角色图像的视觉幻觉检测系统。我们的方法利用了姿势感知上下文视觉学习（PA-ICVL）与视觉语言模型（VLMs），同时利用RGB图像和姿势信息。通过从一个经过微调的姿势估计器中获得姿势指导，我们使VLM能够做出更准确的决策。实验结果表明，在识别视觉幻觉方面，与仅依赖于RGB图像的基线方法相比，取得了显著的改进。这项研究通过减轻视觉幻觉，推动了TTI模型在非照片真实领域的潜力。

    arXiv:2403.15048v1 Announce Type: cross  Abstract: Large-scale Text-to-Image (TTI) models have become a common approach for generating training data in various generative fields. However, visual hallucinations, which contain perceptually critical defects, remain a concern, especially in non-photorealistic styles like cartoon characters. We propose a novel visual hallucination detection system for cartoon character images generated by TTI models. Our approach leverages pose-aware in-context visual learning (PA-ICVL) with Vision-Language Models (VLMs), utilizing both RGB images and pose information. By incorporating pose guidance from a fine-tuned pose estimator, we enable VLMs to make more accurate decisions. Experimental results demonstrate significant improvements in identifying visual hallucinations compared to baseline methods relying solely on RGB images. This research advances TTI models by mitigating visual hallucinations, expanding their potential in non-photorealistic domains.
    
[^4]: 通过高斯混合模型进行域自适应的最优输运

    Optimal Transport for Domain Adaptation through Gaussian Mixture Models

    [https://arxiv.org/abs/2403.13847](https://arxiv.org/abs/2403.13847)

    通过高斯混合模型进行域自适应的最优输运，可以实现源域和目标域混合成分之间的匹配，从而在失效诊断中取得最先进的性能。

    

    在这篇论文中，我们探讨了通过最优输运进行域自适应的方法。我们提出了一种新颖的方法，即通过高斯混合模型对数据分布进行建模。这种策略使我们能够通过等价的离散问题解决连续最优输运。最优输运解决方案为我们提供了源域和目标域混合成分之间的匹配。通过这种匹配，我们可以在域之间映射数据点，或者将标签从源域组件转移到目标域。我们在失效诊断的两个域自适应基准测试中进行了实验，结果表明我们的方法具有最先进的性能。

    arXiv:2403.13847v1 Announce Type: cross  Abstract: In this paper we explore domain adaptation through optimal transport. We propose a novel approach, where we model the data distributions through Gaussian mixture models. This strategy allows us to solve continuous optimal transport through an equivalent discrete problem. The optimal transport solution gives us a matching between source and target domain mixture components. From this matching, we can map data points between domains, or transfer the labels from the source domain components towards the target domain. We experiment with 2 domain adaptation benchmarks in fault diagnosis, showing that our methods have state-of-the-art performance.
    
[^5]: 我们能否逐步验证错误答案检测？

    Can We Verify Step by Step for Incorrect Answer Detection?

    [https://arxiv.org/abs/2402.10528](https://arxiv.org/abs/2402.10528)

    通过推理链来预测大型语言模型输出的准确性，我们引入了一个新的基准R2PE，并提出了处理可辨识性评分（PDS）框架。

    

    Chain-of-Thought（CoT）提示在增强大型语言模型（LLMs）的推理能力方面取得了重大进展。先前的研究开发了各种扩展的CoT，主要集中在增强最终任务的性能上。此外，已经有研究评估了CoT中推理链的质量。这引发了一个有趣的问题：通过仔细审查它们生成的推理链，是否可以预测LLMs输出的准确性？为了回答这个研究问题，我们引入了一个基准，R2PE，专门设计用于探究不同领域涵盖五个不同推理任务中推理链与性能之间的关系。该基准旨在基于推理步骤衡量LLMs最终输出的虚假性。为了充分利用多个推理链中的信息，我们提出了打败常识分数（PDS）框架。

    arXiv:2402.10528v1 Announce Type: cross  Abstract: Chain-of-Thought (CoT) prompting has marked a significant advancement in enhancing the reasoning capabilities of large language models (LLMs). Previous studies have developed various extensions of CoT, which focus primarily on enhancing end-task performance. In addition, there has been research on assessing the quality of reasoning chains in CoT. This raises an intriguing question: Is it possible to predict the accuracy of LLM outputs by scrutinizing the reasoning chains they generate? To answer this research question, we introduce a benchmark, R2PE, designed specifically to explore the relationship between reasoning chains and performance in various reasoning tasks spanning five different domains. This benchmark aims to measure the falsehood of the final output of LLMs based on the reasoning steps. To make full use of information in multiple reasoning chains, we propose the process discernibility score (PDS) framework that beats the a
    
[^6]: 用大型语言模型和检索增强生成提升教科书问答任务

    Enhancing Textbook Question Answering Task with Large Language Models and Retrieval Augmented Generation

    [https://arxiv.org/abs/2402.05128](https://arxiv.org/abs/2402.05128)

    本论文通过引入检索增强生成（RAG）技术和利用迁移学习来处理长文本和提升推理能力，为教科书问答任务带来了显著的改进。

    

    教科书问答（TQA）是人工智能中的一项具有挑战性的任务，由于上下文和多模式数据的复杂性。尽管以前的研究在任务上取得了显著的进展，但仍存在一些限制，包括模型推理能力不足和无法捕捉长文本中的上下文信息。大型语言模型（LLMs）的引入革命了人工智能领域，然而，直接应用LLMs经常会导致不准确的答案。本文提出了一种方法来处理TQA中领域外情景，即概念分布在不同课程中，该方法结合了检索增强生成（RAG）技术和迁移学习来处理长文本并提升推理能力。通过对LLM模型Llama-2进行监督微调并加入RAG，我们的架构优于基线，在验证集上的准确度提高了4.12%，在测试集上提高了9.84%。

    Textbook question answering (TQA) is a challenging task in artificial intelligence due to the complex nature of context and multimodal data. Although previous research has significantly improved the task, there are still some limitations including the models' weak reasoning and inability to capture contextual information in the lengthy context. The introduction of large language models (LLMs) has revolutionized the field of AI, however, directly applying LLMs often leads to inaccurate answers. This paper proposes a methodology that handle the out-of-domain scenario in TQA where concepts are spread across different lessons by incorporating the retrieval augmented generation (RAG) technique and utilize transfer learning to handle the long context and enhance reasoning abilities. Through supervised fine-tuning of the LLM model Llama-2 and the incorporation of RAG, our architecture outperforms the baseline, achieving a 4.12% accuracy improvement on validation set and 9.84% on test set for 
    
[^7]: 划分在层次化任务网络规划中的可计算边界

    The Boundaries of Tractability in Hierarchical Task Network Planning. (arXiv:2401.14174v1 [cs.CC])

    [http://arxiv.org/abs/2401.14174](http://arxiv.org/abs/2401.14174)

    本论文研究了在层次化任务网络规划中三个经典问题的可计算边界，提供了在特定条件下将多项式时间可解性结果从原始任务网络推广到一般任务网络的方法，并且给出了这三个问题的参数化复杂度分析。

    

    我们研究了在层次化任务网络规划中三个经典问题的可计算边界：提供计划的验证，是否存在可执行计划以及是否可以通过某个计划达到给定状态。我们证明了在常量偏序宽度的原始任务网络（以及其推广形式）上，这三个问题都可以在多项式时间内解决，而对于后两个问题，这种情况仅在对状态空间进行可证明的必要限制下才成立。接下来，我们提供了一个算法元定理及相应的下界，以确定从原始任务网络到一般任务网络的一般多项式时间可解性结果可以被提升的严格条件。最后，我们通过分析这三个问题的参数化复杂度来丰富我们的研究，并且证明了(1)通过将偏序宽度替换为t，可以实现这三个问题的固定参数可计算性。

    We study the complexity-theoretic boundaries of tractability for three classical problems in the context of Hierarchical Task Network Planning: the validation of a provided plan, whether an executable plan exists, and whether a given state can be reached by some plan. We show that all three problems can be solved in polynomial time on primitive task networks of constant partial order width (and a generalization thereof), whereas for the latter two problems this holds only under a provably necessary restriction to the state space. Next, we obtain an algorithmic meta-theorem along with corresponding lower bounds to identify tight conditions under which general polynomial-time solvability results can be lifted from primitive to general task networks. Finally, we enrich our investigation by analyzing the parameterized complexity of the three considered problems, and show that (1) fixed-parameter tractability for all three problems can be achieved by replacing the partial order width with t
    
[^8]: DISCOUNT: 使用最优传输进行分布式对抗解释

    DISCOUNT: Distributional Counterfactual Explanation With Optimal Transport. (arXiv:2401.13112v1 [cs.AI])

    [http://arxiv.org/abs/2401.13112](http://arxiv.org/abs/2401.13112)

    本文提出了使用最优传输进行分布式对抗解释的方法DISCOUNT，将对抗解释的概念扩展到整个输入输出分布，并通过统计置信度来支撑这一方法。

    

    对抗解释是在黑盒决策模型中提供洞察力和可解释性的事实方法，通过确定导致不同结果的替代输入实例来实现。本文将对抗解释的概念扩展到分布上下文，从个体数据点扩大到整个输入输出分布，命名为分布式对抗解释。在分布式对抗解释中，我们的重点转向分析事实和对抗的分布属性，类似于评估个体实例及其结果决策的经典方法。我们利用最优传输来构建一个机会约束优化问题，旨在导出与事实对应的对抗分布，以统计置信度做支撑。我们提出的优化方法DISCOUNT在输入和输出分布之间平衡这种置信度。

    Counterfactual Explanations (CE) is the de facto method for providing insight and interpretability in black-box decision-making models by identifying alternative input instances that lead to different outcomes. This paper extends the concept of CEs to a distributional context, broadening the scope from individual data points to entire input and output distributions, named Distributional Counterfactual Explanation (DCE). In DCE, our focus shifts to analyzing the distributional properties of the factual and counterfactual, drawing parallels to the classical approach of assessing individual instances and their resulting decisions. We leverage Optimal Transport (OT) to frame a chance-constrained optimization problem, aiming to derive a counterfactual distribution that closely aligns with its factual counterpart, substantiated by statistical confidence. Our proposed optimization method, DISCOUNT, strategically balances this confidence across both input and output distributions. This algorit
    
[^9]: 学习用于视觉Transformer预训练的掩码和置换视觉令牌。

    Learning to Mask and Permute Visual Tokens for Vision Transformer Pre-Training. (arXiv:2306.07346v1 [cs.CV])

    [http://arxiv.org/abs/2306.07346](http://arxiv.org/abs/2306.07346)

    本论文提出了一种新的自监督预训练方法MaPeT，不同于现有的使用掩码图像模型的方法，该方法使用自回归和置换预测来捕获图像块内的依赖关系并减少数据噪声的影响，从而提高了下游任务的一致性。

    

    使用自监督预训练技术已成为提高图像分类等视觉任务性能的有前途的方法。最近的方法使用掩码图像模型范式，通过重构与随机掩码图像块相关联的视觉令牌来预训练骨干网络。然而，这种掩蔽方法会在预训练过程中引入噪声进入输入数据，导致性能下降。此外，输入掩蔽忽略了受损块之间的依赖关系，增加了下游微调任务中观察到的不一致性。为了解决这些问题，我们提出了一种新的自监督预训练方法，名为掩蔽和置换视觉变压器（MaPeT），它使用自回归和置换预测来捕获块内依赖性。此外，MaPeT使用辅助位置信息来减少预训练和微调阶段中的差异性。

    The use of self-supervised pre-training has emerged as a promising approach to enhance the performance of visual tasks such as image classification. In this context, recent approaches have employed the Masked Image Modeling paradigm, which pre-trains a backbone by reconstructing visual tokens associated with randomly masked image patches. This masking approach, however, introduces noise into the input data during pre-training, leading to discrepancies that can impair performance during the fine-tuning phase. Furthermore, input masking neglects the dependencies between corrupted patches, increasing the inconsistencies observed in downstream fine-tuning tasks. To overcome these issues, we propose a new self-supervised pre-training approach, named Masked and Permuted Vision Transformer (MaPeT), that employs autoregressive and permuted predictions to capture intra-patch dependencies. In addition, MaPeT employs auxiliary positional information to reduce the disparity between the pre-trainin
    
[^10]: 带有延迟组合匿名赌徒反馈的随机次模赌博算法

    Stochastic Submodular Bandits with Delayed Composite Anonymous Bandit Feedback. (arXiv:2303.13604v1 [cs.LG])

    [http://arxiv.org/abs/2303.13604](http://arxiv.org/abs/2303.13604)

    本论文研究了具有随机次模收益和全赌徒延迟反馈的组合多臂赌博机问题，研究了三种延迟反馈模型并导出了后悔上限。研究结果表明，算法能够在考虑延迟组合匿名反馈时胜过其他全赌徒方法。

    

    本文研究了组合多臂赌博机问题，其中包含了期望下的随机次模收益和全赌徒延迟反馈，延迟反馈被假定为组合和匿名。也就是说，延迟反馈是由过去行动的奖励组成的，这些奖励由子组件构成，其未知的分配方式。研究了三种延迟反馈模型：有界对抗模型、随机独立模型和随机条件独立模型，并针对每种延迟模型导出了后悔界。忽略问题相关参数，我们证明了所有延迟模型的后悔界为 $\tilde{O}(T^{2/3} + T^{1/3} \nu)$，其中 $T$ 是时间范围，$\nu$ 是三种情况下不同定义的延迟参数，因此展示了带有延迟的补偿项。所考虑的算法被证明能够胜过其他考虑了延迟组合匿名反馈的全赌徒方法。

    This paper investigates the problem of combinatorial multiarmed bandits with stochastic submodular (in expectation) rewards and full-bandit delayed feedback, where the delayed feedback is assumed to be composite and anonymous. In other words, the delayed feedback is composed of components of rewards from past actions, with unknown division among the sub-components. Three models of delayed feedback: bounded adversarial, stochastic independent, and stochastic conditionally independent are studied, and regret bounds are derived for each of the delay models. Ignoring the problem dependent parameters, we show that regret bound for all the delay models is $\tilde{O}(T^{2/3} + T^{1/3} \nu)$ for time horizon $T$, where $\nu$ is a delay parameter defined differently in the three cases, thus demonstrating an additive term in regret with delay in all the three delay models. The considered algorithm is demonstrated to outperform other full-bandit approaches with delayed composite anonymous feedbac
    


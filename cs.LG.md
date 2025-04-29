# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fix-Con: Automatic Fault Localization and Repair of Deep Learning Model Conversions](https://rss.arxiv.org/abs/2312.15101) | 本文提出了一种自动化的故障定位和修复方法Fix-Con，用于在深度学习模型转换过程中修复由转换引入的故障。Fix-Con能够检测和修复模型输入、参数、超参数和模型图方面的故障，提高转换模型的部署和预测正确性。 |
| [^2] | [Unsolvable Problem Detection: Evaluating Trustworthiness of Vision Language Models](https://arxiv.org/abs/2403.20331) | 本文提出了一个新颖且重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型在视觉问答任务中能否在面对不可解问题时保持答案的能力，并通过广泛实验发现大多数模型存在改进的空间。 |
| [^3] | [Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation](https://arxiv.org/abs/2403.19103) | PRISM是一种算法，可以自动识别人类可解释且易传递的提示，从而有效生成所需概念，仅使用黑盒访问T2I模型。 |
| [^4] | [Hyperparameters in Continual Learning: a Reality Check](https://arxiv.org/abs/2403.09066) | 超参数对于连续学习的重要性被强调，提出了一个涉及超参数调整和评估阶段的评估协议。 |
| [^5] | [Human Curriculum Effects Emerge with In-Context Learning in Neural Networks](https://arxiv.org/abs/2402.08674) | 人类学习对示例课程和任务结构有敏感性。研究发现，在神经网络和语言模型中，通过上下文学习方法可以同时获得分组和交错训练的优势。 |
| [^6] | [Predictive Churn with the Set of Good Models](https://arxiv.org/abs/2402.07745) | 本文研究了在现代大众市场应用中，随时间更新的机器学习模型可能导致不稳定的预测结果，通过研究预测性多样性，量化了这种预测性流失，并通过Rashomon集合来分析模型更新中的预期流失。 |
| [^7] | [Causal Q-Aggregation for CATE Model Selection.](http://arxiv.org/abs/2310.16945) | 该论文提出了一种基于Q集成的CATE模型选择方法，其通过使用双重鲁棒损失实现了统计上的最佳预测模型选择遗憾率 |
| [^8] | [Large language models in biomedical natural language processing: benchmarks, baselines, and recommendations.](http://arxiv.org/abs/2305.16326) | 本文研究了GPT-3和GPT-4在生物医学自然语言处理中的表现，分析了它们可能产生的错误类型，并提供了使用这些模型的建议。 |
| [^9] | [FEDORA: Flying Event Dataset fOr Reactive behAvior.](http://arxiv.org/abs/2305.14392) | FEDORA是一个飞行事件数据集，解决了现有数据集缺少完整数据和时间分辨率的问题，旨在帮助在资源受限环境下实现基于视觉的自主导航和避障。 |
| [^10] | [NoisyHate: Benchmarking Content Moderation Machine Learning Models with Human-Written Perturbations Online.](http://arxiv.org/abs/2303.10430) | 本文提出了一个包含人类编写的在线扰动的测试集，用于毒性言论检测模型的评估。 |
| [^11] | [Finding Minimum-Cost Explanations for Predictions made by Tree Ensembles.](http://arxiv.org/abs/2303.09271) | 本研究提出了一种高效的oracle系统，能够寻找树集成模型预测的最小代价解释，该算法比目前最先进的替代方案的运行表现更好。m-MARCO算法可以计算每个预测的单个最小解释，并证明相对于枚举所有最小解释的MARCO算法，我们的方法具有两倍的总体加速比。 |
| [^12] | [Precise High-Dimensional Asymptotics for Quantifying Heterogeneous Transfers.](http://arxiv.org/abs/2010.11750) | 本文利用随机矩阵理论在线性回归设置中，对于具有两个任务的高维情况下的常用估计量的超额风险进行了精确渐近分析。 |

# 详细

[^1]: 修复-Con：深度学习模型转换的自动故障定位和修复

    Fix-Con: Automatic Fault Localization and Repair of Deep Learning Model Conversions

    [https://rss.arxiv.org/abs/2312.15101](https://rss.arxiv.org/abs/2312.15101)

    本文提出了一种自动化的故障定位和修复方法Fix-Con，用于在深度学习模型转换过程中修复由转换引入的故障。Fix-Con能够检测和修复模型输入、参数、超参数和模型图方面的故障，提高转换模型的部署和预测正确性。

    

    在不同深度学习框架之间进行模型转换是一种常见的步骤，可以最大程度地增加模型在设备之间的兼容性，并利用可能只在一个深度学习框架中提供的优化功能。然而，这个转换过程可能存在错误，导致转换后的模型无法部署或存在问题，严重降低了其预测的正确性。我们提出了一种自动化的故障定位和修复方法，Fix-Con，在深度学习框架之间进行模型转换时使用。Fix-Con能够检测和修复在转换过程中引入的模型输入、参数、超参数和模型图的故障。Fix-Con使用从调查转换问题中挖掘出的一组故障类型来定位转换模型中潜在的转换故障，并适当修复它们，例如使用源模型的参数替换目标模型的参数。这一过程在数据集中的每个图像上进行迭代执行。

    Converting deep learning models between frameworks is a common step to maximize model compatibility across devices and leverage optimization features that may be exclusively provided in one deep learning framework. However, this conversion process may be riddled with bugs, making the converted models either undeployable or problematic, considerably degrading their prediction correctness.   We propose an automated approach for fault localization and repair, Fix-Con, during model conversion between deep learning frameworks. Fix-Con is capable of detecting and fixing faults introduced in model input, parameters, hyperparameters, and the model graph during conversion.   Fix-Con uses a set of fault types mined from surveying conversion issues raised to localize potential conversion faults in the converted target model, and then repairs them appropriately, e.g. replacing the parameters of the target model with those from the source model. This is done iteratively for every image in the datas
    
[^2]: 不可解问题检测：评估视觉语言模型的可信度

    Unsolvable Problem Detection: Evaluating Trustworthiness of Vision Language Models

    [https://arxiv.org/abs/2403.20331](https://arxiv.org/abs/2403.20331)

    本文提出了一个新颖且重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型在视觉问答任务中能否在面对不可解问题时保持答案的能力，并通过广泛实验发现大多数模型存在改进的空间。

    

    本文介绍了一个新颖而重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型（VLMs）在视觉问答（VQA）任务中面对不可解问题时保持答案的能力。UPD包括三个不同的设置：缺失答案检测（AAD）、不兼容答案集检测（IASD）和不兼容视觉问题检测（IVQD）。通过广泛的实验深入研究UPD问题表明，大多数VLMs，包括GPT-4V和LLaVA-Next-34B，在各种程度上都很难应对我们的基准测试，突显了改进的重要空间。为了解决UPD，我们探索了无需训练和基于训练的解决方案，提供了对其有效性和局限性的新见解。我们希望我们的见解，以及在提议的UPD设置内的未来努力，将增强对VLMs的更广泛理解和发展。

    arXiv:2403.20331v1 Announce Type: cross  Abstract: This paper introduces a novel and significant challenge for Vision Language Models (VLMs), termed Unsolvable Problem Detection (UPD). UPD examines the VLM's ability to withhold answers when faced with unsolvable problems in the context of Visual Question Answering (VQA) tasks. UPD encompasses three distinct settings: Absent Answer Detection (AAD), Incompatible Answer Set Detection (IASD), and Incompatible Visual Question Detection (IVQD). To deeply investigate the UPD problem, extensive experiments indicate that most VLMs, including GPT-4V and LLaVA-Next-34B, struggle with our benchmarks to varying extents, highlighting significant room for the improvements. To address UPD, we explore both training-free and training-based solutions, offering new insights into their effectiveness and limitations. We hope our insights, together with future efforts within the proposed UPD settings, will enhance the broader understanding and development of
    
[^3]: 用于个性化文本到图像生成的自动化黑盒提示工程

    Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation

    [https://arxiv.org/abs/2403.19103](https://arxiv.org/abs/2403.19103)

    PRISM是一种算法，可以自动识别人类可解释且易传递的提示，从而有效生成所需概念，仅使用黑盒访问T2I模型。

    

    提示工程对于控制文本到图像（T2I）生成模型的输出是有效的，但由于需要手动制作提示而导致工作繁重。这一挑战促使了自动提示生成算法的发展。然而，这些方法通常在T2I模型之间的可传递性方面遇到困难，需要对基础模型进行白盒访问，并产生非直观的提示。在这项工作中，我们介绍了PRISM，这是一种算法，可以仅使用黑盒访问T2I模型就自动识别人类可解释且易传递的提示，从而有效生成所需概念。受大型语言模型（LLM）越狱的启发，PRISM利用LLM的上下文学习能力来迭代地改进给定参考图像的候选提示分布。我们的实验展示了PRISM在为对象、样式等生成准确提示方面的多样性和有效性。

    arXiv:2403.19103v1 Announce Type: cross  Abstract: Prompt engineering is effective for controlling the output of text-to-image (T2I) generative models, but it is also laborious due to the need for manually crafted prompts. This challenge has spurred the development of algorithms for automated prompt generation. However, these methods often struggle with transferability across T2I models, require white-box access to the underlying model, and produce non-intuitive prompts. In this work, we introduce PRISM, an algorithm that automatically identifies human-interpretable and transferable prompts that can effectively generate desired concepts given only black-box access to T2I models. Inspired by large language model (LLM) jailbreaking, PRISM leverages the in-context learning ability of LLMs to iteratively refine the candidate prompts distribution for given reference images. Our experiments demonstrate the versatility and effectiveness of PRISM in generating accurate prompts for objects, sty
    
[^4]: Continual Learning中的超参数：现实检验

    Hyperparameters in Continual Learning: a Reality Check

    [https://arxiv.org/abs/2403.09066](https://arxiv.org/abs/2403.09066)

    超参数对于连续学习的重要性被强调，提出了一个涉及超参数调整和评估阶段的评估协议。

    

    不同的连续学习（CL）算法旨在在CL过程中有效地缓解稳定性和可塑性之间的权衡，为了实现这一目标，调整每种算法的适当超参数是必不可少的。本文主张现行的评估协议既不切实际，也无法有效评估连续学习算法的能力。

    arXiv:2403.09066v1 Announce Type: new  Abstract: Various algorithms for continual learning (CL) have been designed with the goal of effectively alleviating the trade-off between stability and plasticity during the CL process. To achieve this goal, tuning appropriate hyperparameters for each algorithm is essential. As an evaluation protocol, it has been common practice to train a CL algorithm using diverse hyperparameter values on a CL scenario constructed with a benchmark dataset. Subsequently, the best performance attained with the optimal hyperparameter value serves as the criterion for evaluating the CL algorithm. In this paper, we contend that this evaluation protocol is not only impractical but also incapable of effectively assessing the CL capability of a CL algorithm. Returning to the fundamental principles of model evaluation in machine learning, we propose an evaluation protocol that involves Hyperparameter Tuning and Evaluation phases. Those phases consist of different datase
    
[^5]: 使用上下文学习的神经网络中出现人类课程效应

    Human Curriculum Effects Emerge with In-Context Learning in Neural Networks

    [https://arxiv.org/abs/2402.08674](https://arxiv.org/abs/2402.08674)

    人类学习对示例课程和任务结构有敏感性。研究发现，在神经网络和语言模型中，通过上下文学习方法可以同时获得分组和交错训练的优势。

    

    人类学习对规则结构和训练中所使用的示例课程非常敏感。在由简洁规则控制的任务中，当相关示例在多次试验中被分组时，学习更加稳健；但在缺乏这样的规则的情况下，交错训练更加有效。迄今为止，没有神经模型能够同时捕捉到这些看似矛盾的效应。在本文中，我们展示了“上下文学习”（ICL）在使用元学习进行训练的神经网络和大型语言模型（LLMs）中自发产生了同样的权衡。ICL是通过内层循环算法在激活动力学中实现的一种“上下文内学习”（in-context learning）的能力，可以在没有权重更改的情况下学习新任务。对预训练的LLMs和元学习变压器进行的实验表明，ICL在涉及规则结构的任务中展示出了人类所示的分组优势，而同时进行权重学习则复制了人类在缺少这样结构的任务上所观察到的交错优势。

    Human learning is sensitive to rule-like structure and the curriculum of examples used for training. In tasks governed by succinct rules, learning is more robust when related examples are blocked across trials, but in the absence of such rules, interleaving is more effective. To date, no neural model has simultaneously captured these seemingly contradictory effects. Here we show that this same tradeoff spontaneously emerges with "in-context learning" (ICL) both in neural networks trained with metalearning and in large language models (LLMs). ICL is the ability to learn new tasks "in context" - without weight changes - via an inner-loop algorithm implemented in activation dynamics. Experiments with pretrained LLMs and metalearning transformers show that ICL exhibits the blocking advantage demonstrated in humans on a task involving rule-like structure, and conversely, that concurrent in-weight learning reproduces the interleaving advantage observed in humans on tasks lacking such structu
    
[^6]: 使用一组好模型进行预测性客户流失

    Predictive Churn with the Set of Good Models

    [https://arxiv.org/abs/2402.07745](https://arxiv.org/abs/2402.07745)

    本文研究了在现代大众市场应用中，随时间更新的机器学习模型可能导致不稳定的预测结果，通过研究预测性多样性，量化了这种预测性流失，并通过Rashomon集合来分析模型更新中的预期流失。

    

    现代大众市场应用中的机器学习模型经常会随时间进行更新。面临的一个主要挑战是，尽管整体性能在提升，但这些更新可能会以不可预测的方式改变特定模型的预测结果。在实践中，研究人员通过量化模型更新前后的不稳定预测数量来衡量预测性流失。本文通过预测性多样性的角度研究了这种效应，即在一组接近最优模型（Rashomon集合）中存在冲突预测的普遍性。我们展示了如何利用传统的预测性多样性度量来研究这组潜在模型的预期流失，即可能用于替换基线模型在部署中的模型集合。我们从不同的角度给出了模型集内Rashomon集合之间预期流失的理论结果，并通过Rashomon集合表征了模型更新中的预期流失，结合我们的分析结果。

    Machine learning models in modern mass-market applications are often updated over time. One of the foremost challenges faced is that, despite increasing overall performance, these updates may flip specific model predictions in unpredictable ways. In practice, researchers quantify the number of unstable predictions between models pre and post update -- i.e., predictive churn. In this paper, we study this effect through the lens of predictive multiplicity -- i.e., the prevalence of conflicting predictions over the set of near-optimal models (the Rashomon set). We show how traditional measures of predictive multiplicity can be used to examine expected churn over this set of prospective models -- i.e., the set of models that may be used to replace a baseline model in deployment. We present theoretical results on the expected churn between models within the Rashomon set from different perspectives. And we characterize expected churn over model updates via the Rashomon set, pairing our analy
    
[^7]: Causal Q-Aggregation for CATE Model Selection（CATE模型选择中的因果Q集成）

    Causal Q-Aggregation for CATE Model Selection. (arXiv:2310.16945v1 [stat.ML])

    [http://arxiv.org/abs/2310.16945](http://arxiv.org/abs/2310.16945)

    该论文提出了一种基于Q集成的CATE模型选择方法，其通过使用双重鲁棒损失实现了统计上的最佳预测模型选择遗憾率

    

    准确估计条件平均处理效应（CATE）是个性化决策的核心。尽管有大量用于CATE估计的模型，但由于因果推断的基本问题，模型选择是一项非常棘手的任务。最近的实证工作提供了有利于具有双重鲁棒性质的代理损失度量和模型集成的证据。然而，对于这些模型的理论理解还不够。直接应用先前的理论工作会由于模型选择问题的非凸性而导致次优的预测模型选择率。我们提供了现有主要CATE集成方法的遗憾率，并提出了一种基于双重鲁棒损失的Q集成的新的CATE模型集成方法。我们的主要结果表明，因果Q集成在预测模型选择的遗憾率上达到了统计上的最优值为$\frac{\log(M)}{n}$（其中$M$为模型数，$n$为样本数），加上高阶估计误差项

    Accurate estimation of conditional average treatment effects (CATE) is at the core of personalized decision making. While there is a plethora of models for CATE estimation, model selection is a nontrivial task, due to the fundamental problem of causal inference. Recent empirical work provides evidence in favor of proxy loss metrics with double robust properties and in favor of model ensembling. However, theoretical understanding is lacking. Direct application of prior theoretical work leads to suboptimal oracle model selection rates due to the non-convexity of the model selection problem. We provide regret rates for the major existing CATE ensembling approaches and propose a new CATE model ensembling approach based on Q-aggregation using the doubly robust loss. Our main result shows that causal Q-aggregation achieves statistically optimal oracle model selection regret rates of $\frac{\log(M)}{n}$ (with $M$ models and $n$ samples), with the addition of higher-order estimation error term
    
[^8]: 生物医学自然语言处理中的大型语言模型: 基准、基线和建议

    Large language models in biomedical natural language processing: benchmarks, baselines, and recommendations. (arXiv:2305.16326v1 [cs.CL])

    [http://arxiv.org/abs/2305.16326](http://arxiv.org/abs/2305.16326)

    本文研究了GPT-3和GPT-4在生物医学自然语言处理中的表现，分析了它们可能产生的错误类型，并提供了使用这些模型的建议。

    

    生物医学文献呈指数级增长，手动筛选和提取知识变得困难。自动从生物医学文献中提取信息的生物医学自然语言处理（BioNLP）技术有助于减轻这种负担。近年来，如GPT-3和GPT-4等大型语言模型（LLMs）因其卓越的性能而受到重视。但是，它们在BioNLP任务中的有效性以及对方法开发和下游用户的影响仍未得到研究。本研究（1）在四个应用程序中在八个BioNLP数据集中建立了GPT-3和GPT-4在零-shot和一-shot设置下的基准表现，包括命名实体识别，关系提取，多标签文档分类和语义相似性和推理；（2）审查了LLMs产生的错误，并将错误分为三种类型：缺失，不一致和不需要的人工内容；（3）提出了使用LLMs的建议。

    Biomedical literature is growing rapidly, making it challenging to curate and extract knowledge manually. Biomedical natural language processing (BioNLP) techniques that can automatically extract information from biomedical literature help alleviate this burden. Recently, large Language Models (LLMs), such as GPT-3 and GPT-4, have gained significant attention for their impressive performance. However, their effectiveness in BioNLP tasks and impact on method development and downstream users remain understudied. This pilot study (1) establishes the baseline performance of GPT-3 and GPT-4 at both zero-shot and one-shot settings in eight BioNLP datasets across four applications: named entity recognition, relation extraction, multi-label document classification, and semantic similarity and reasoning, (2) examines the errors produced by the LLMs and categorized the errors into three types: missingness, inconsistencies, and unwanted artificial content, and (3) provides suggestions for using L
    
[^9]: FEDORA：用于反应行为的飞行事件数据集

    FEDORA: Flying Event Dataset fOr Reactive behAvior. (arXiv:2305.14392v1 [cs.CV])

    [http://arxiv.org/abs/2305.14392](http://arxiv.org/abs/2305.14392)

    FEDORA是一个飞行事件数据集，解决了现有数据集缺少完整数据和时间分辨率的问题，旨在帮助在资源受限环境下实现基于视觉的自主导航和避障。

    

    生物体在飞行中使用极少数的神经元和极低的失误率执行复杂的高速机动，突显了这些资源受限制的生物系统的有效性。近年来，事件驱动硬件逐渐成为在资源受限环境中实现复杂视觉任务的一种有前途的方法。基于视觉的自主导航和避障包括几个独立但相关的任务，如光流估计、深度估计、同时定位与建图（SLAM）、物体检测和识别。为了确保这些任务之间的一致性，他们必须在单个数据集上进行训练。然而，大多数现有数据集只提供所需数据的选定子集，这使得网络间的一致性难以实现。现有数据集的另一个限制是提供的有限时间分辨率。为解决这些限制，我们提出了FEDORA，

    The ability of living organisms to perform complex high speed manoeuvers in flight with a very small number of neurons and an incredibly low failure rate highlights the efficacy of these resource-constrained biological systems. Event-driven hardware has emerged, in recent years, as a promising avenue for implementing complex vision tasks in resource-constrained environments. Vision-based autonomous navigation and obstacle avoidance consists of several independent but related tasks such as optical flow estimation, depth estimation, Simultaneous Localization and Mapping (SLAM), object detection, and recognition. To ensure coherence between these tasks, it is imperative that they be trained on a single dataset. However, most existing datasets provide only a selected subset of the required data. This makes inter-network coherence difficult to achieve. Another limitation of existing datasets is the limited temporal resolution they provide. To address these limitations, we present FEDORA, a 
    
[^10]: NoisyHate：在人类编写的在线扰动下对内容审核机器学习模型进行基准测试

    NoisyHate: Benchmarking Content Moderation Machine Learning Models with Human-Written Perturbations Online. (arXiv:2303.10430v1 [cs.LG])

    [http://arxiv.org/abs/2303.10430](http://arxiv.org/abs/2303.10430)

    本文提出了一个包含人类编写的在线扰动的测试集，用于毒性言论检测模型的评估。

    

    在社交媒体上，具有有害内容的在线文本是一种威胁，可能会引起网络骚扰。尽管许多平台采取了措施，例如基于机器学习的仇恨言论检测系统来减少其影响，但那些有害内容发布者仍然可以通过修改有害词汇的拼写来逃避系统。这些修改后的单词也称为人类编写的文本扰动。许多研究开发了一定的技术来生成对抗样本，以帮助机器学习模型获得识别这些扰动的能力。然而，机器生成的扰动与人类编写的扰动之间仍存在差距。在本文中，我们介绍了一个包含人类编写的在线扰动的基准测试集，用于毒性言论检测模型。我们还招募了一组工人来评估此测试集的质量并删除低质量的样本。同时，为了检查我们的扰动是否可以归一化为其干净版本，我们还创建了一个相关的测试集。

    Online texts with toxic content are a threat in social media that might cause cyber harassment. Although many platforms applied measures, such as machine learning-based hate-speech detection systems, to diminish their effect, those toxic content publishers can still evade the system by modifying the spelling of toxic words. Those modified words are also known as human-written text perturbations. Many research works developed certain techniques to generate adversarial samples to help the machine learning models obtain the ability to recognize those perturbations. However, there is still a gap between those machine-generated perturbations and human-written perturbations. In this paper, we introduce a benchmark test set containing human-written perturbations online for toxic speech detection models. We also recruited a group of workers to evaluate the quality of this test set and dropped low-quality samples. Meanwhile, to check if our perturbation can be normalized to its clean version, w
    
[^11]: 寻找树集成模型预测的最小代价解释

    Finding Minimum-Cost Explanations for Predictions made by Tree Ensembles. (arXiv:2303.09271v1 [cs.LG])

    [http://arxiv.org/abs/2303.09271](http://arxiv.org/abs/2303.09271)

    本研究提出了一种高效的oracle系统，能够寻找树集成模型预测的最小代价解释，该算法比目前最先进的替代方案的运行表现更好。m-MARCO算法可以计算每个预测的单个最小解释，并证明相对于枚举所有最小解释的MARCO算法，我们的方法具有两倍的总体加速比。

    

    当机器学习模型作为关键系统的决策支持时，能够解释为何模型做出特定预测的能力至关重要。提供的解释必须是可证明的，并且最好不包含冗余信息，即最小解释。本文旨在寻找树集成模型预测的解释，这些解释不仅是最小的，而且在成本函数方面也是最小的。为此，我们首先提出了一个高效的“神谕”系统，可以确定解释的正确性，在计算最小解释时超越了当前最先进的替代方案的运行表现数个数量级。其次，我们改编了来自相关工作的叫做MARCO的算法（将其称为m-MARCO），目的是计算每个预测的单个最小解释，并证明相对于枚举所有最小解释的MARCO算法，我们的方法具有两倍的总体加速比。

    The ability to explain why a machine learning model arrives at a particular prediction is crucial when used as decision support by human operators of critical systems. The provided explanations must be provably correct, and preferably without redundant information, called minimal explanations. In this paper, we aim at finding explanations for predictions made by tree ensembles that are not only minimal, but also minimum with respect to a cost function.  To this end, we first present a highly efficient oracle that can determine the correctness of explanations, surpassing the runtime performance of current state-of-the-art alternatives by several orders of magnitude when computing minimal explanations.  Secondly, we adapt an algorithm called MARCO from related works (calling it m-MARCO) for the purpose of computing a single minimum explanation per prediction, and demonstrate an overall speedup factor of two compared to the MARCO algorithm which enumerates all minimal explanations.  Final
    
[^12]: 量化异构转移的精确高维渐近分析

    Precise High-Dimensional Asymptotics for Quantifying Heterogeneous Transfers. (arXiv:2010.11750v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2010.11750](http://arxiv.org/abs/2010.11750)

    本文利用随机矩阵理论在线性回归设置中，对于具有两个任务的高维情况下的常用估计量的超额风险进行了精确渐近分析。

    

    最近，学习一个任务时使用来自另一个任务的样本的问题引起了广泛关注。本文提出了一个基本问题：什么时候将来自两个任务的数据合并比单独学习一个任务更好？直观上，从一个任务到另一个任务的转移效应取决于数据集的转移，如样本大小和协方差矩阵。然而，量化这种转移效应是具有挑战性的，因为我们需要比较联合学习和单任务学习之间的风险，并且一个任务是否比另一个任务具有比较优势取决于两个任务之间确切的数据集转移类型。本文利用随机矩阵理论在具有两个任务的线性回归设置中解决了这一挑战。我们给出了在高维情况下一些常用估计量的超额风险的精确渐近分析，当样本大小与特征维度成比例增加时，固定比例。精确渐近分析以样本大小的函数形式给出。

    The problem of learning one task with samples from another task has received much interest recently. In this paper, we ask a fundamental question: when is combining data from two tasks better than learning one task alone? Intuitively, the transfer effect from one task to another task depends on dataset shifts such as sample sizes and covariance matrices. However, quantifying such a transfer effect is challenging since we need to compare the risks between joint learning and single-task learning, and the comparative advantage of one over the other depends on the exact kind of dataset shift between both tasks. This paper uses random matrix theory to tackle this challenge in a linear regression setting with two tasks. We give precise asymptotics about the excess risks of some commonly used estimators in the high-dimensional regime, when the sample sizes increase proportionally with the feature dimension at fixed ratios. The precise asymptotics is provided as a function of the sample sizes 
    


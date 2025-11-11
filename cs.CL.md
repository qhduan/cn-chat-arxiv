# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Likelihood-based Mitigation of Evaluation Bias in Large Language Models](https://arxiv.org/abs/2402.15987) | 该论文研究了基于大型语言模型（LLM）的评估器中的似然偏差，并提出了一种缓解这种偏差的方法。 |
| [^2] | [SoLA: Solver-Layer Adaption of LLM for Better Logic Reasoning](https://arxiv.org/abs/2402.11903) | 提出了一种新颖的求解器层适应（SoLA）方法，在LLM中引入求解器层，不同地引导解决方案朝向可满足性 |

# 详细

[^1]: 基于似然的大型语言模型评估偏差的缓解

    Likelihood-based Mitigation of Evaluation Bias in Large Language Models

    [https://arxiv.org/abs/2402.15987](https://arxiv.org/abs/2402.15987)

    该论文研究了基于大型语言模型（LLM）的评估器中的似然偏差，并提出了一种缓解这种偏差的方法。

    

    大型语言模型(LLMs)被广泛用于评估自然语言生成任务的自动化指标。然而，似然作为衡量LLM对句子可信度的指标，可能会因句子表面差异（如词序和句子结构）而变化。因此，如果将LLMs用于评估，可能存在似然偏差：它们可能会高估具有较高似然性的句子，而低估具有较低似然性的句子。本文对LLM评估器中似然偏差的存在和影响进行了研究。我们还提出了一种缓解似然偏差的方法。我们的方法利用高度偏置的实例作为少样本示例进行上下文学习。我们在评估数据到文本和语法错误纠正任务时的实验结果显示，我们测试的几种LLMs显示出似然偏差。此外，我们提出的方法成功地减轻了这种偏差

    arXiv:2402.15987v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are widely used to evaluate natural language generation tasks as automated metrics. However, the likelihood, a measure of LLM's plausibility for a sentence, can vary due to superficial differences in sentences, such as word order and sentence structure. It is therefore possible that there might be a likelihood bias if LLMs are used for evaluation: they might overrate sentences with higher likelihoods while underrating those with lower likelihoods. In this paper, we investigate the presence and impact of likelihood bias in LLM-based evaluators. We also propose a method to mitigate the likelihood bias. Our method utilizes highly biased instances as few-shot examples for in-context learning. Our experiments in evaluating the data-to-text and grammatical error correction tasks reveal that several LLMs we test display a likelihood bias. Furthermore, our proposed method successfully mitigates this bias, also impr
    
[^2]: SoLA: 为了更好的逻辑推理而对LLM进行求解层调整

    SoLA: Solver-Layer Adaption of LLM for Better Logic Reasoning

    [https://arxiv.org/abs/2402.11903](https://arxiv.org/abs/2402.11903)

    提出了一种新颖的求解器层适应（SoLA）方法，在LLM中引入求解器层，不同地引导解决方案朝向可满足性

    

    针对大型语言模型（LLMs）在逻辑推理上面临的挑战，先前的努力试图通过工具学习来改变问题求解。虽然在小规模问题上已经取得了进展，但由于规模庞大且表达复杂，解决工业案例仍然困难。在本文中，我们提出了一种新颖的求解层适应（SoLA）方法，我们在LLM中引入了一个求解器作为新层，不同地引导解决方案朝向可满足性。在SoLA中，LLM旨在理解自然语言描述的搜索空间，并识别最高质量的局部解，而求解器层则专注于初始解不满足的约束条件。借助MaxSAT作为桥梁，我们定义了前向和后向传递梯度，使最终模型能够收敛到一个满足的解或证明不可满足性。后门理论确保SoLA能够获得准确性

    arXiv:2402.11903v1 Announce Type: cross  Abstract: Considering the challenges faced by large language models (LLMs) on logical reasoning, prior efforts have sought to transform problem-solving through tool learning. While progress has been made on small-scale problems, solving industrial cases remains difficult due to their large scale and intricate expressions. In this paper, we propose a novel solver-layer adaptation (SoLA) method, where we introduce a solver as a new layer of the LLM to differentially guide solutions towards satisfiability. In SoLA, LLM aims to comprehend the search space described in natural language and identify local solutions of the highest quality, while the solver layer focuses solely on constraints not satisfied by the initial solution. Leveraging MaxSAT as a bridge, we define forward and backward transfer gradients, enabling the final model to converge to a satisfied solution or prove unsatisfiability. The backdoor theory ensures that SoLA can obtain accurat
    


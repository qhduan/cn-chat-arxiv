# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Narration-of-Thought: Inference-Time Scaffolding for Defeasible Ethical Reasoning in Large Language Models](https://arxiv.org/abs/2606.26366) | 本文提出一种名为“思维叙述”的系统提示方法，通过将思维链结构化为五个特定部分，显著减少了大型语言模型在伦理推理中忽视利益相关者和压制不确定性的错误，无需额外训练或微调。 |
| [^2] | [Semantic Representation Learning of Scientific Literature based on Adaptive Feature and Graph Neural Network.](http://arxiv.org/abs/2311.00296) | 提出了一种基于自适应特征和图神经网络的科学文献语义表示学习方法，通过全局和局部考虑科学文献特征，并使用图注意机制对具有引用关系的特征进行加权求和，以更好地表达不同科学文献特征之间的关联。 |

# 详细

[^1]: 思维叙述：大型语言模型中可废止伦理推理的推理时脚手架

    Narration-of-Thought: Inference-Time Scaffolding for Defeasible Ethical Reasoning in Large Language Models

    [https://arxiv.org/abs/2606.26366](https://arxiv.org/abs/2606.26366)

    本文提出一种名为“思维叙述”的系统提示方法，通过将思维链结构化为五个特定部分，显著减少了大型语言模型在伦理推理中忽视利益相关者和压制不确定性的错误，无需额外训练或微调。

    

    arXiv:2606.26366v1 公告类型：新 摘要：针对道德困境的标准思维链存在两种失败模式：利益相关者崩溃（思维链中最多只提及一个与结果相关的当事方）和不确定性抑制（在做出行动承诺前，没有明确提及未知或保留意见）。我们引入了思维叙述（NoT），这是一种系统提示，将思维链结构化为五个部分：主角、利益相关者、两步后果、不确定性、然后承诺。NoT无需额外训练、参数或微调。在来自三家供应商的四个生成器上的100个每日困境场景中，NoT将每个模型上的利益相关者崩溃率从高达31%降至1%以下，将不确定性抑制率从高达72%降至1-24%。一个匹配预算的详细思维链控制实验排除了令牌消耗作为有效成分；NoT在四个生成器中的三个上，在利益相关者数量上保持了+0.79至+0.90的Cliff's delta优势，在不确定性得分上保持了+0.65至+0.93的优势，并且一个部分消融实验归因了这些改进。

    arXiv:2606.26366v1 Announce Type: new  Abstract: Standard chain-of-thought on moral dilemmas exhibits two failure modes: stakeholder collapse (the trace names at most one party with a stake in the outcome) and uncertainty suppression (no explicit unknowns or hedges before committing to an action). We introduce narration-of-thought (NoT), a system prompt that structures chain-of-thought into five sections: protagonist, stakeholders, two-step consequences, uncertainty, then commitment. NoT adds no training, parameters, or fine-tuning. On 100 DailyDilemmas scenarios across four generators from three vendors, NoT cuts stakeholder collapse from up to 31% to under 1% and uncertainty suppression from up to 72% to 1-24% on every model. A matched-budget verbose-CoT control rules out token spend as the active ingredient; NoT retains Cliff's delta advantages of +0.79 to +0.90 on stakeholder count and +0.65 to +0.93 on uncertainty score for three of four generators, and a section ablation attribut
    
[^2]: 基于自适应特征与图神经网络的科学文献语义表示学习

    Semantic Representation Learning of Scientific Literature based on Adaptive Feature and Graph Neural Network. (arXiv:2311.00296v1 [cs.CL])

    [http://arxiv.org/abs/2311.00296](http://arxiv.org/abs/2311.00296)

    提出了一种基于自适应特征和图神经网络的科学文献语义表示学习方法，通过全局和局部考虑科学文献特征，并使用图注意机制对具有引用关系的特征进行加权求和，以更好地表达不同科学文献特征之间的关联。

    

    由于大部分科学文献数据未标记，因此基于无监督图的语义表示学习变得至关重要。同时，为了丰富科学文献的特征，提出了一种基于自适应特征和图神经网络的科学文献语义表示学习方法。通过引入自适应特征方法，全局和局部考虑科学文献的特征。使用图注意机制对具有引用关系的科学文献特征进行求和，并给予每个科学文献不同的特征权重，以更好地表达不同科学文献特征之间的关联。此外，还提出了一种无监督图神经网络语义表示学习方法，通过比较正负局部语义表示与全局图语义表示之间的互信息来学习。

    Because most of the scientific literature data is unmarked, it makes semantic representation learning based on unsupervised graph become crucial. At the same time, in order to enrich the features of scientific literature, a learning method of semantic representation of scientific literature based on adaptive features and graph neural network is proposed. By introducing the adaptive feature method, the features of scientific literature are considered globally and locally. The graph attention mechanism is used to sum the features of scientific literature with citation relationship, and give each scientific literature different feature weights, so as to better express the correlation between the features of different scientific literature. In addition, an unsupervised graph neural network semantic representation learning method is proposed. By comparing the mutual information between the positive and negative local semantic representation of scientific literature and the global graph sema
    


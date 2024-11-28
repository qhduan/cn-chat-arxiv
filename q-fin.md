# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Algorithmic Collusion by Large Language Models](https://arxiv.org/abs/2404.00806) | 大型语言模型的算法定价代理在寡头市场环境中自主勾结，对消费者利益有害，其说明书中的短语变化可能增加勾结。 |
| [^2] | [CaT-GNN: Enhancing Credit Card Fraud Detection via Causal Temporal Graph Neural Networks](https://arxiv.org/abs/2402.14708) | 该论文提出了一种名为CaT-GNN的新型信用卡欺诈检测方法，通过因果不变性学习揭示交易数据中的固有相关性，并引入因果混合策略来增强模型的鲁棒性和可解释性。 |

# 详细

[^1]: 大型语言模型的算法勾结

    Algorithmic Collusion by Large Language Models

    [https://arxiv.org/abs/2404.00806](https://arxiv.org/abs/2404.00806)

    大型语言模型的算法定价代理在寡头市场环境中自主勾结，对消费者利益有害，其说明书中的短语变化可能增加勾结。

    

    arXiv:2404.00806v1 公告类型:交叉摘要:算法定价的兴起引起了对算法勾结的担忧。我们对基于大型语言模型（LLMs）特别是GPT-4的算法定价代理进行实验。我们发现：（1）基于LLM的代理在定价任务上表现出色，（2）基于LLM的定价代理在寡头市场环境中自主勾结，损害消费者利益，（3）LLM说明书中看似无害短语("提示")的变化可能会增加勾结。这些结果也适用于拍卖设置。我们的发现强调了有关算法定价的反垄断监管的必要性，并发现了基于LLM的定价代理所面临的监管挑战。

    arXiv:2404.00806v1 Announce Type: cross  Abstract: The rise of algorithmic pricing raises concerns of algorithmic collusion. We conduct experiments with algorithmic pricing agents based on Large Language Models (LLMs), and specifically GPT-4. We find that (1) LLM-based agents are adept at pricing tasks, (2) LLM-based pricing agents autonomously collude in oligopoly settings to the detriment of consumers, and (3) variation in seemingly innocuous phrases in LLM instructions ("prompts") may increase collusion. These results extend to auction settings. Our findings underscore the need for antitrust regulation regarding algorithmic pricing, and uncover regulatory challenges unique to LLM-based pricing agents.
    
[^2]: 通过因果时间图神经网络增强信用卡欺诈检测

    CaT-GNN: Enhancing Credit Card Fraud Detection via Causal Temporal Graph Neural Networks

    [https://arxiv.org/abs/2402.14708](https://arxiv.org/abs/2402.14708)

    该论文提出了一种名为CaT-GNN的新型信用卡欺诈检测方法，通过因果不变性学习揭示交易数据中的固有相关性，并引入因果混合策略来增强模型的鲁棒性和可解释性。

    

    信用卡欺诈对经济构成重大威胁。尽管基于图神经网络（GNN）的欺诈检测方法表现良好，但它们经常忽视节点的本地结构对预测的因果效应。本文引入了一种新颖的信用卡欺诈检测方法——CaT-GNN（Causal Temporal Graph Neural Networks），利用因果不变性学习来揭示交易数据中的固有相关性。通过将问题分解为发现和干预阶段，CaT-GNN确定交易图中的因果节点，并应用因果混合策略来增强模型的鲁棒性和可解释性。CaT-GNN由两个关键组件组成：Causal-Inspector和Causal-Intervener。Causal-Inspector利用时间注意力机制中的注意力权重来识别因果和环境

    arXiv:2402.14708v1 Announce Type: cross  Abstract: Credit card fraud poses a significant threat to the economy. While Graph Neural Network (GNN)-based fraud detection methods perform well, they often overlook the causal effect of a node's local structure on predictions. This paper introduces a novel method for credit card fraud detection, the \textbf{\underline{Ca}}usal \textbf{\underline{T}}emporal \textbf{\underline{G}}raph \textbf{\underline{N}}eural \textbf{N}etwork (CaT-GNN), which leverages causal invariant learning to reveal inherent correlations within transaction data. By decomposing the problem into discovery and intervention phases, CaT-GNN identifies causal nodes within the transaction graph and applies a causal mixup strategy to enhance the model's robustness and interpretability. CaT-GNN consists of two key components: Causal-Inspector and Causal-Intervener. The Causal-Inspector utilizes attention weights in the temporal attention mechanism to identify causal and environm
    


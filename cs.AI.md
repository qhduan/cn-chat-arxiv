# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Uncertainty quantification in fine-tuned LLMs using LoRA ensembles](https://arxiv.org/abs/2402.12264) | 使用LoRA集成在精调LLMs中提出了一种原则性不确定性量化方法，通过对不同数据域的低秩适应集成分析，推测了模型对特定架构难以学习的数据领域的信号。 |
| [^2] | [On the Evolution of Knowledge Graphs: A Survey and Perspective.](http://arxiv.org/abs/2310.04835) | 本文对各种类型的知识图谱进行了全面调研，介绍了知识提取和推理技术，并展望了结合知识图谱和大语言模型的力量以及知识工程的未来方向。 |
| [^3] | [HurriCast: An Automatic Framework Using Machine Learning and Statistical Modeling for Hurricane Forecasting.](http://arxiv.org/abs/2309.07174) | 本研究提出了HurriCast，一种使用机器学习和统计建模的自动化框架，通过组合ARIMA模型和K-MEANS算法以更好地捕捉飓风趋势，并结合Autoencoder进行改进的飓风模拟，从而有效模拟历史飓风行为并提供详细的未来预测。这项研究通过利用全面且有选择性的数据集，丰富了对飓风模式的理解，并为风险管理策略提供了可操作的见解。 |
| [^4] | [Quantum Circuit Fidelity Improvement with Long Short-Term Memory Networks.](http://arxiv.org/abs/2303.17523) | 本文提出使用长短期记忆网络解决量子计算中的保真度问题，利用时间序列预测方法预测量子电路的保真度。 |

# 详细

[^1]: 使用LoRA集成在精调LLMs中的不确定性量化

    Uncertainty quantification in fine-tuned LLMs using LoRA ensembles

    [https://arxiv.org/abs/2402.12264](https://arxiv.org/abs/2402.12264)

    使用LoRA集成在精调LLMs中提出了一种原则性不确定性量化方法，通过对不同数据域的低秩适应集成分析，推测了模型对特定架构难以学习的数据领域的信号。

    

    精调大型语言模型可以提高特定任务的性能，尽管对于精调模型学到了什么、遗忘了什么以及如何信任其预测仍然缺乏一个一般的理解。我们提出了使用计算效率高的低秩适应集成对精调LLMs进行基于后验逼近的原则性不确定性量化。我们使用基于Mistral-7b的低秩适应集成分析了三个常见的多项选择数据集，并对其在精调过程中和之后对不同目标领域的感知复杂性和模型效能进行了定量和定性的结论。具体而言，基于数值实验支持，我们对那些对于给定架构难以学习的数据领域的熵不确定性度量提出了假设。

    arXiv:2402.12264v1 Announce Type: cross  Abstract: Fine-tuning large language models can improve task specific performance, although a general understanding of what the fine-tuned model has learned, forgotten and how to trust its predictions is still missing. We derive principled uncertainty quantification for fine-tuned LLMs with posterior approximations using computationally efficient low-rank adaptation ensembles. We analyze three common multiple-choice datasets using low-rank adaptation ensembles based on Mistral-7b, and draw quantitative and qualitative conclusions on their perceived complexity and model efficacy on the different target domains during and after fine-tuning. In particular, backed by the numerical experiments, we hypothesise about signals from entropic uncertainty measures for data domains that are inherently difficult for a given architecture to learn.
    
[^2]: 关于知识图谱的演化：一项调研和展望

    On the Evolution of Knowledge Graphs: A Survey and Perspective. (arXiv:2310.04835v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2310.04835](http://arxiv.org/abs/2310.04835)

    本文对各种类型的知识图谱进行了全面调研，介绍了知识提取和推理技术，并展望了结合知识图谱和大语言模型的力量以及知识工程的未来方向。

    

    知识图谱（KGs）是多样化知识的结构化表示，广泛应用于各种智能应用。本文对各种类型的知识图谱的演化（静态KGs，动态KGs，时态KGs和事件KGs）和知识提取和推理技术进行了全面调查。此外，我们介绍了不同类型的KGs的实际应用，包括财务分析的案例研究。最后，我们提出了关于知识工程未来方向的展望，包括结合知识图谱和大语言模型（LLMs）的力量以及知识提取、推理和表示的演化。

    Knowledge graphs (KGs) are structured representations of diversified knowledge. They are widely used in various intelligent applications. In this article, we provide a comprehensive survey on the evolution of various types of knowledge graphs (i.e., static KGs, dynamic KGs, temporal KGs, and event KGs) and techniques for knowledge extraction and reasoning. Furthermore, we introduce the practical applications of different types of KGs, including a case study in financial analysis. Finally, we propose our perspective on the future directions of knowledge engineering, including the potential of combining the power of knowledge graphs and large language models (LLMs), and the evolution of knowledge extraction, reasoning, and representation.
    
[^3]: HurriCast：使用机器学习和统计建模的自动化框架用于飓风预测

    HurriCast: An Automatic Framework Using Machine Learning and Statistical Modeling for Hurricane Forecasting. (arXiv:2309.07174v1 [cs.LG])

    [http://arxiv.org/abs/2309.07174](http://arxiv.org/abs/2309.07174)

    本研究提出了HurriCast，一种使用机器学习和统计建模的自动化框架，通过组合ARIMA模型和K-MEANS算法以更好地捕捉飓风趋势，并结合Autoencoder进行改进的飓风模拟，从而有效模拟历史飓风行为并提供详细的未来预测。这项研究通过利用全面且有选择性的数据集，丰富了对飓风模式的理解，并为风险管理策略提供了可操作的见解。

    

    飓风由于其灾害性影响而在美国面临重大挑战。减轻这些风险很重要，保险业在这方面起着重要作用，使用复杂的统计模型进行风险评估。然而，这些模型常常忽视关键的时间和空间飓风模式，并受到数据稀缺的限制。本研究引入了一种改进的方法，将ARIMA模型和K-MEANS相结合，以更好地捕捉飓风趋势，并使用Autoencoder进行改进的飓风模拟。我们的实验证明，这种混合方法有效地模拟了历史飓风行为，同时提供了潜在未来路径和强度的详细预测。此外，通过利用全面而有选择性的数据集，我们的模拟丰富了对飓风模式的当前理解，并为风险管理策略提供了可操作的见解。

    Hurricanes present major challenges in the U.S. due to their devastating impacts. Mitigating these risks is important, and the insurance industry is central in this effort, using intricate statistical models for risk assessment. However, these models often neglect key temporal and spatial hurricane patterns and are limited by data scarcity. This study introduces a refined approach combining the ARIMA model and K-MEANS to better capture hurricane trends, and an Autoencoder for enhanced hurricane simulations. Our experiments show that this hybrid methodology effectively simulate historical hurricane behaviors while providing detailed projections of potential future trajectories and intensities. Moreover, by leveraging a comprehensive yet selective dataset, our simulations enrich the current understanding of hurricane patterns and offer actionable insights for risk management strategies.
    
[^4]: 利用长短期记忆网络提高量子电路保真度

    Quantum Circuit Fidelity Improvement with Long Short-Term Memory Networks. (arXiv:2303.17523v1 [quant-ph])

    [http://arxiv.org/abs/2303.17523](http://arxiv.org/abs/2303.17523)

    本文提出使用长短期记忆网络解决量子计算中的保真度问题，利用时间序列预测方法预测量子电路的保真度。

    

    量子计算已进入噪声中间规模量子（NISQ）时代，目前我们拥有的量子处理器对辐射和温度等环境变量敏感，因此会产生嘈杂的输出。虽然已经有许多算法和应用程序用于NISQ处理器，但我们仍面临着解释其嘈杂结果的不确定性。具体来说，我们对所选择的量子态有多少信心？这种信心很重要，因为NISQ计算机将输出其量子位测量的概率分布，有时很难区分分布是否表示有意义的计算或只是随机噪声。本文提出了一种新方法来解决这个问题，将量子电路保真度预测框架为时间序列预测问题，因此可以利用长短期记忆（LSTM）神经网络的强大能力。一个完整的工作流程来构建训练电路

    Quantum computing has entered the Noisy Intermediate-Scale Quantum (NISQ) era. Currently, the quantum processors we have are sensitive to environmental variables like radiation and temperature, thus producing noisy outputs. Although many proposed algorithms and applications exist for NISQ processors, we still face uncertainties when interpreting their noisy results. Specifically, how much confidence do we have in the quantum states we are picking as the output? This confidence is important since a NISQ computer will output a probability distribution of its qubit measurements, and it is sometimes hard to distinguish whether the distribution represents meaningful computation or just random noise. This paper presents a novel approach to attack this problem by framing quantum circuit fidelity prediction as a Time Series Forecasting problem, therefore making it possible to utilize the power of Long Short-Term Memory (LSTM) neural networks. A complete workflow to build the training circuit d
    


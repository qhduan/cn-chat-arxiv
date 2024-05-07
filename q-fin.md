# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Emoji Driven Crypto Assets Market Reactions](https://arxiv.org/abs/2402.10481) | 该研究利用GPT-4和BERT模型进行多模态情感分析，发现基于表情符号情绪的策略可以帮助避免市场下挫并稳定回报。 |
| [^2] | [Fourier Neural Network Approximation of Transition Densities in Finance.](http://arxiv.org/abs/2309.03966) | 本文提出了FourNet，一种使用高斯激活函数的单层神经网络，能够以任意精度逼近具有已知傅里叶变换的过渡密度，并通过严密的误差分析给出了估计误差和非负性损失的界限。 |
| [^3] | [Unveiling the Potential of Sentiment: Can Large Language Models Predict Chinese Stock Price Movements?.](http://arxiv.org/abs/2306.14222) | 本篇论文研究如何运用大型语言模型提取中文新闻文本信息的情感因素，以期促进知情和高频的投资组合调整。通过建立严格和全面的基准测试与标准化的回测框架，作者对不同类型 LLMs 在该领域内的效果进行了客观评估。 |
| [^4] | [Multiarmed Bandits Problem Under the Mean-Variance Setting.](http://arxiv.org/abs/2212.09192) | 本文将经典的多臂赌博机问题扩展到均值-方差设置，并通过考虑亚高斯臂放松了先前假设，解决了风险-回报权衡的问题。 |

# 详细

[^1]: 基于表情符号的加密资产市场反应

    Emoji Driven Crypto Assets Market Reactions

    [https://arxiv.org/abs/2402.10481](https://arxiv.org/abs/2402.10481)

    该研究利用GPT-4和BERT模型进行多模态情感分析，发现基于表情符号情绪的策略可以帮助避免市场下挫并稳定回报。

    

    在加密货币领域，诸如Twitter之类的社交媒体平台已经成为影响市场趋势和投资者情绪的关键因素。在我们的研究中，我们利用GPT-4和经过微调的基于BERT模型的多模态情感分析，重点关注表情符号情绪对加密货币市场的影响。通过将表情符号转化为可量化的情感数据，我们将这些见解与BTC价格和VCRIX指数等关键市场指标进行了相关联。这种方法可以用于开发旨在利用社交媒体元素识别和预测市场趋势的交易策略。关键是，我们的研究结果表明，基于表情符号情绪的策略可以有助于避免重大市场下挫，并有助于回报的稳定。这项研究强调了将先进的基于人工智能的分析整合到金融策略中的实际益处，并提供了一种新的方式来看待市场预测。

    arXiv:2402.10481v1 Announce Type: cross  Abstract: In the burgeoning realm of cryptocurrency, social media platforms like Twitter have become pivotal in influencing market trends and investor sentiments. In our study, we leverage GPT-4 and a fine-tuned transformer-based BERT model for a multimodal sentiment analysis, focusing on the impact of emoji sentiment on cryptocurrency markets. By translating emojis into quantifiable sentiment data, we correlate these insights with key market indicators like BTC Price and the VCRIX index. This approach may be fed into the development of trading strategies aimed at utilizing social media elements to identify and forecast market trends. Crucially, our findings suggest that strategies based on emoji sentiment can facilitate the avoidance of significant market downturns and contribute to the stabilization of returns. This research underscores the practical benefits of integrating advanced AI-driven analyses into financial strategies, offering a nuan
    
[^2]: 金融中傅里叶神经网络逼近过渡密度

    Fourier Neural Network Approximation of Transition Densities in Finance. (arXiv:2309.03966v1 [q-fin.CP])

    [http://arxiv.org/abs/2309.03966](http://arxiv.org/abs/2309.03966)

    本文提出了FourNet，一种使用高斯激活函数的单层神经网络，能够以任意精度逼近具有已知傅里叶变换的过渡密度，并通过严密的误差分析给出了估计误差和非负性损失的界限。

    

    本文引入了FourNet，一种新颖的单层前馈神经网络（FFNN）方法，用于逼近具有封闭形式的傅里叶变换（即特征函数）可用的过渡密度。FourNet的一个独特特点在于它使用了高斯激活函数，使得精确的傅里叶和逆傅里叶变换成为可能，并与高斯混合模型进行类比。我们从数学上证明了FourNet能够以任意精度逼近过渡密度，并仅使用有限数量的神经元。FourNet的参数通过最小化基于已知特征函数和FFNN的傅里叶变换的损失函数来学习，同时采用了策略性采样方法来增强训练。通过严密而全面的误差分析，我们推导出了$L_2$估计误差和估计密度中非负性的潜在（逐点）损失的信息界限。

    This paper introduces FourNet, a novel single-layer feed-forward neural network (FFNN) method designed to approximate transition densities for which closed-form expressions of their Fourier transforms, i.e. characteristic functions, are available. A unique feature of FourNet lies in its use of a Gaussian activation function, enabling exact Fourier and inverse Fourier transformations and drawing analogies with the Gaussian mixture model. We mathematically establish FourNet's capacity to approximate transition densities in the $L_2$-sense arbitrarily well with finite number of neurons. The parameters of FourNet are learned by minimizing a loss function derived from the known characteristic function and the Fourier transform of the FFNN, complemented by a strategic sampling approach to enhance training. Through a rigorous and comprehensive error analysis, we derive informative bounds for the $L_2$ estimation error and the potential (pointwise) loss of nonnegativity in the estimated densit
    
[^3]: 揭示情感的潜力：大型语言模型能否预测中国股票价格波动？

    Unveiling the Potential of Sentiment: Can Large Language Models Predict Chinese Stock Price Movements?. (arXiv:2306.14222v1 [cs.CL])

    [http://arxiv.org/abs/2306.14222](http://arxiv.org/abs/2306.14222)

    本篇论文研究如何运用大型语言模型提取中文新闻文本信息的情感因素，以期促进知情和高频的投资组合调整。通过建立严格和全面的基准测试与标准化的回测框架，作者对不同类型 LLMs 在该领域内的效果进行了客观评估。

    

    大型语言模型 (LLMs) 的快速发展已引发了广泛的讨论，其中包括它们将如何提高量化股票交易策略的回报的潜力。这些讨论主要围绕着利用 LLMs 的出色理解能力来提取情感因素，从而促进知情和高频的投资组合调整。为了确保这些 LLMs 成功地应用于中国金融文本分析和随后的中国股票市场交易策略开发中，我们提供了一个严格和全面的基准测试以及一个标准化的回测框架，旨在客观评估不同类型 LLMs 在中文新闻文本数据的情感因素提取中的效果。为了说明我们基准测试的工作方式，我们引用了三个不同模型：1）生成式 LLM (ChatGPT)，2）中文语言特定的预训练 LLM (二郎神 RoBERTa)，以及……

    The rapid advancement of Large Language Models (LLMs) has led to extensive discourse regarding their potential to boost the return of quantitative stock trading strategies. This discourse primarily revolves around harnessing the remarkable comprehension capabilities of LLMs to extract sentiment factors which facilitate informed and high-frequency investment portfolio adjustments. To ensure successful implementations of these LLMs into the analysis of Chinese financial texts and the subsequent trading strategy development within the Chinese stock market, we provide a rigorous and encompassing benchmark as well as a standardized back-testing framework aiming at objectively assessing the efficacy of various types of LLMs in the specialized domain of sentiment factor extraction from Chinese news text data. To illustrate how our benchmark works, we reference three distinctive models: 1) the generative LLM (ChatGPT), 2) the Chinese language-specific pre-trained LLM (Erlangshen-RoBERTa), and 
    
[^4]: 均值-方差设置下的多臂赌博机问题

    Multiarmed Bandits Problem Under the Mean-Variance Setting. (arXiv:2212.09192v3 [math.OC] UPDATED)

    [http://arxiv.org/abs/2212.09192](http://arxiv.org/abs/2212.09192)

    本文将经典的多臂赌博机问题扩展到均值-方差设置，并通过考虑亚高斯臂放松了先前假设，解决了风险-回报权衡的问题。

    

    经典的多臂赌博机（MAB）问题涉及一个学习者和一个包含K个独立臂的集合，每个臂都有自己的事前未知独立奖励分布。在有限次选择中的每一次，学习者选择一个臂并接收新信息。学习者经常面临一个勘探-开发困境：通过玩估计奖励最高的臂来利用当前信息，还是探索所有臂以收集更多奖励信息。设计目标旨在最大化所有回合中的期望累积奖励。然而，这样的目标并不考虑风险-回报权衡，而这在许多应用领域，特别是金融和经济领域，常常是一项基本原则。在本文中，我们在Sani等人（2012）的基础上，将经典的MAB问题扩展到均值-方差设置。具体而言，我们通过考虑亚高斯臂放松了Sani等人（2012）做出的独立臂和有界奖励的假设。

    The classical multi-armed bandit (MAB) problem involves a learner and a collection of K independent arms, each with its own ex ante unknown independent reward distribution. At each one of a finite number of rounds, the learner selects one arm and receives new information. The learner often faces an exploration-exploitation dilemma: exploiting the current information by playing the arm with the highest estimated reward versus exploring all arms to gather more reward information. The design objective aims to maximize the expected cumulative reward over all rounds. However, such an objective does not account for a risk-reward tradeoff, which is often a fundamental precept in many areas of applications, most notably in finance and economics. In this paper, we build upon Sani et al. (2012) and extend the classical MAB problem to a mean-variance setting. Specifically, we relax the assumptions of independent arms and bounded rewards made in Sani et al. (2012) by considering sub-Gaussian arms.
    


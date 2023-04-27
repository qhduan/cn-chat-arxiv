# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Maximum Implied Variance Slope -- Practical Aspects.](http://arxiv.org/abs/2304.13610) | 本文发现 Black-Scholes 模型中的无套利约束很宽松，在斜率的大范围内几乎总是会出现套利。 |
| [^2] | [Convexity adjustments \`a la Malliavin.](http://arxiv.org/abs/2304.13402) | 本文提出了一种基于Malliavin calculus的新方法，用于寻找各种传统利率产品的凸性调整近似值。在通用的一因子Cheyette模型族下，我们成功找到了期货、OIS期货、FRAs和CMS的近似值，并表现出了精确度极高的特点。 |
| [^3] | [Learning Volatility Surfaces using Generative Adversarial Networks.](http://arxiv.org/abs/2304.13128) | 本文提出了一种使用GAN高效计算波动率曲面的方法。所提出的GAN模型允许使用浅层网络，从而大大降低了计算成本。实验结果表明，在计算波动率曲面方面具有优势。 |
| [^4] | [Regulatory Markets: The Future of AI Governance.](http://arxiv.org/abs/2304.04914) | 提出一种监管市场的概念，即政府要求受监管对象从私人监管机构购买监管服务，以克服过度依赖行业自律和立法机构缺乏专业知识的局限性，从而逐步实现人工智能的恰当监管。 |
| [^5] | [Dynamic Feature Engineering and model selection methods for temporal tabular datasets with regime changes.](http://arxiv.org/abs/2301.00790) | 本文提出了一种新的机器学习管道，用于在数据制度变化下对时序面板数据集的预测进行排名。使用梯度提升决策树（GBDT）并结合dropout技术的模型具有良好的性能和泛化能力，而动态特征中和则是一种高效而不需要重新训练模型就可以应用于任何机器学习模型中的后处理技术。 |
| [^6] | [Making heads or tails of systemic risk measures.](http://arxiv.org/abs/2206.02582) | 本文建立了系统性风险测量方法和单变量风险测量方法之间的联系，提出新的实证估计方法，并认为基于ES的测量方法更适合于测量网络风险。 |
| [^7] | [Polarization and Quid Pro Quo: The Role of Party Cohesiveness.](http://arxiv.org/abs/2205.07486) | 研究表明，一个利益集团是否能够利用不断增加的意识形态和情感极化获利取决于两个政党社交网络相对凝聚度的聚合度量，而在没有意识形态 polarization 的情况下，两个政党社交网络的相对凝聚度则变得无关紧要。 |
| [^8] | [The Politics of (No) Compromise: Information Acquisition, Policy Discretion, and Reputation.](http://arxiv.org/abs/2111.00522) | 论文讨论了在政策裁量可以被限制的情况下，具有职业忧虑的决策者何时以及如何获取政策相关信息并进行改革决策，并指出公众可以通过在决策者的选择中消除温和政策或现状来鼓励信息获取。 |
| [^9] | [Welfare v. Consent: On the Optimal Penalty for Harassment.](http://arxiv.org/abs/2103.00734) | 该论文提出了一种替代经济学方法，名为同意方法，以促进同意交互和防止非同意交互，突出了同意方法与福利方法之间的差异，其中同意方法对于骚扰的处罚不能为零。 |

# 详细

[^1]: 最大隐含方差斜率——实际应用方面的考虑。

    Maximum Implied Variance Slope -- Practical Aspects. (arXiv:2304.13610v1 [q-fin.PR])

    [http://arxiv.org/abs/2304.13610](http://arxiv.org/abs/2304.13610)

    本文发现 Black-Scholes 模型中的无套利约束很宽松，在斜率的大范围内几乎总是会出现套利。

    

    在Black-Scholes模型中，无套利的存在对隐含方差斜率在对数货币性方面施加了必要的约束，对于大的对数货币性，渐进地成立。这些约束例如被用于SVI隐含波动率参数化中，以确保所得到的笑曲线没有套利。本文表明，这些无套利约束是非常温和的，而在大范围 enforced这些约束的斜率范围内，套利几乎总是能够保证。

    In the Black-Scholes model, the absence of arbitrages imposes necessary constraints on the slope of the implied variance in terms of log-moneyness, asymptotically for large log-moneyness. The constraints are used for example in the SVI implied volatility parameterization to ensure the resulting smile has no arbitrages. This note shows that those no-arbitrage contraints are very mild, and that arbitrage is almost always guaranteed in a large range of slopes where the contraints are enforced.
    
[^2]: Malliavin变换下的凸性调整方法

    Convexity adjustments \`a la Malliavin. (arXiv:2304.13402v1 [q-fin.MF])

    [http://arxiv.org/abs/2304.13402](http://arxiv.org/abs/2304.13402)

    本文提出了一种基于Malliavin calculus的新方法，用于寻找各种传统利率产品的凸性调整近似值。在通用的一因子Cheyette模型族下，我们成功找到了期货、OIS期货、FRAs和CMS的近似值，并表现出了精确度极高的特点。

    

    本文研究利用Malliavin演算法寻找各种传统利率产品的凸性调整近似值的新方法。Malliavin演算为得到凸性调整模板提供了简单的途径。我们在一个通用的一因子Cheyette模型族下找到了期货、OIS期货、FRAs和CMS的近似值，并且发现所获得的公式具有极高的数值精度。

    In this paper, we develop a novel method based on Malliavin calculus to find an approximation for the convexity adjustment for various classical interest rate products. Malliavin calculus provides a simple way to get a template for the convexity adjustment. We find the approximation for Futures, OIS Futures, FRAs, and CMSs under a general family of the one-factor Cheyette model. We have also seen the excellent quality of the numerical accuracy of the formulas obtained.
    
[^3]: 使用生成对抗网络学习波动率曲面

    Learning Volatility Surfaces using Generative Adversarial Networks. (arXiv:2304.13128v1 [q-fin.CP])

    [http://arxiv.org/abs/2304.13128](http://arxiv.org/abs/2304.13128)

    本文提出了一种使用GAN高效计算波动率曲面的方法。所提出的GAN模型允许使用浅层网络，从而大大降低了计算成本。实验结果表明，在计算波动率曲面方面具有优势。

    

    本文提出了一种使用生成对抗网络（GAN）高效计算波动率曲面的方法。这种方法利用了GAN神经网络的特殊结构，一方面可以从训练数据中学习波动率曲面，另一方面可以执行无套利条件。特别地，生成器网络由鉴别器辅助训练，鉴别器评估生成的波动率是否与目标分布相匹配。同时，我们的框架通过引入惩罚项作为正则化项，训练GAN网络以满足无套利约束。所提出的GAN模型允许使用浅层网络，从而大大降低了计算成本。在实验中，我们通过与计算隐含和本地波动率曲面的最先进方法进行对比，展示了所提出的方法的性能。我们的实验结果表明，相对于人工神经网络（ANN）方法，我们的GAN模型在精度和实际应用中都具有优势。

    In this paper, we propose a generative adversarial network (GAN) approach for efficiently computing volatility surfaces. The idea is to make use of the special GAN neural architecture so that on one hand, we can learn volatility surfaces from training data and on the other hand, enforce no-arbitrage conditions. In particular, the generator network is assisted in training by a discriminator that evaluates whether the generated volatility matches the target distribution. Meanwhile, our framework trains the GAN network to satisfy the no-arbitrage constraints by introducing penalties as regularization terms. The proposed GAN model allows the use of shallow networks which results in much less computational costs. In our experiments, we demonstrate the performance of the proposed method by comparing with the state-of-the-art methods for computing implied and local volatility surfaces. We show that our GAN model can outperform artificial neural network (ANN) approaches in terms of accuracy an
    
[^4]: 监管市场：人工智能治理的未来

    Regulatory Markets: The Future of AI Governance. (arXiv:2304.04914v1 [cs.AI])

    [http://arxiv.org/abs/2304.04914](http://arxiv.org/abs/2304.04914)

    提出一种监管市场的概念，即政府要求受监管对象从私人监管机构购买监管服务，以克服过度依赖行业自律和立法机构缺乏专业知识的局限性，从而逐步实现人工智能的恰当监管。

    

    恰当地监管人工智能是一个日益紧迫的政策挑战。立法机构和监管机构缺乏翻译公众需求为法律要求所需的专业知识。过度依赖行业自律未能使AI系统的生产者和使用者对民主要求负责。提出了监管市场的概念，即政府要求受监管对象从私人监管机构购买监管服务。这种方法可以克服命令和控制监管和自我监管的局限性。监管市场可以使政府为AI监管建立政策优先级，同时依靠市场力量和行业研发努力来开创最能实现政策制定者声明目标的监管方法。

    Appropriately regulating artificial intelligence is an increasingly urgent policy challenge. Legislatures and regulators lack the specialized knowledge required to best translate public demands into legal requirements. Overreliance on industry self-regulation fails to hold producers and users of AI systems accountable to democratic demands. Regulatory markets, in which governments require the targets of regulation to purchase regulatory services from a private regulator, are proposed. This approach to AI regulation could overcome the limitations of both command-and-control regulation and self-regulation. Regulatory market could enable governments to establish policy priorities for the regulation of AI, whilst relying on market forces and industry R&D efforts to pioneer the methods of regulation that best achieve policymakers' stated objectives.
    
[^5]: 面向时序表格数据的动态特征工程和模型选择方法在制度变化下的应用

    Dynamic Feature Engineering and model selection methods for temporal tabular datasets with regime changes. (arXiv:2301.00790v2 [q-fin.CP] UPDATED)

    [http://arxiv.org/abs/2301.00790](http://arxiv.org/abs/2301.00790)

    本文提出了一种新的机器学习管道，用于在数据制度变化下对时序面板数据集的预测进行排名。使用梯度提升决策树（GBDT）并结合dropout技术的模型具有良好的性能和泛化能力，而动态特征中和则是一种高效而不需要重新训练模型就可以应用于任何机器学习模型中的后处理技术。

    

    由于严重的非平稳性，将深度学习算法应用于时序面板数据集是困难的，这可能导致过度拟合的模型在制度变化下性能不佳。在本文中，我们提出了一种新的机器学习管道，用于在数据制度变化下对时序面板数据集的预测进行排名。管道评估不同的机器学习模型，包括梯度提升决策树（GBDT）和具有和不具有简单特征工程的神经网络。我们发现，具有dropout的GBDT模型具有高性能、稳健性和泛化能力，而且相对复杂度较低、计算成本较低。然后，我们展示了在线学习技术可以在预测后处理中用于增强结果。特别地，我们提出了动态特征中和，这是一种无需重新训练模型就可以应用于任何机器学习模型的高效过程。

    The application of deep learning algorithms to temporal panel datasets is difficult due to heavy non-stationarities which can lead to over-fitted models that under-perform under regime changes. In this work we propose a new machine learning pipeline for ranking predictions on temporal panel datasets which is robust under regime changes of data. Different machine-learning models, including Gradient Boosting Decision Trees (GBDTs) and Neural Networks with and without simple feature engineering are evaluated in the pipeline with different settings. We find that GBDT models with dropout display high performance, robustness and generalisability with relatively low complexity and reduced computational cost. We then show that online learning techniques can be used in post-prediction processing to enhance the results. In particular, dynamic feature neutralisation, an efficient procedure that requires no retraining of models and can be applied post-prediction to any machine learning model, impr
    
[^6]: 评估系统性风险测量方法

    Making heads or tails of systemic risk measures. (arXiv:2206.02582v2 [q-fin.RM] UPDATED)

    [http://arxiv.org/abs/2206.02582](http://arxiv.org/abs/2206.02582)

    本文建立了系统性风险测量方法和单变量风险测量方法之间的联系，提出新的实证估计方法，并认为基于ES的测量方法更适合于测量网络风险。

    

    本文展示了CoVaR、$\Delta$-CoVaR、CoES、$\Delta$-CoES和MES系统性风险测量方法可以用于在copula确定的分位数处评估单变量风险测量。利用该结果，本文推断了这些测量方法在对幂律尾部和异常值的敏感性以及集成性质方面的实证相关性。此外，提出了一种新的CoES实证估计方法。幂律结果被应用于导出一种新的幂律系数的实证估计方法，该方法取决于$\Delta\text{-CoVaR}/\Delta\text{-CoES}$。模拟和对金融机构大数据集的应用展示了实证效果。本文认为MES不适合用于测量极端风险。而基于ES的测量方法更敏感于幂次律尾部和大量损失，这使得这些测量方法更适合于测量网络风险，但不太适合用于测量系统性风险。

    This paper shows that the CoVaR,$\Delta$-CoVaR,CoES,$\Delta$-CoES and MES systemic risk measures can be represented in terms of the univariate risk measure evaluated at a quantile determined by the copula. The result is applied to derive empirically relevant properties of these measures concerning their sensitivity to power-law tails, outliers and their properties under aggregation. Furthermore, a novel empirical estimator for the CoES is proposed. The power-law result is applied to derive a novel empirical estimator for the power-law coefficient which depends on $\Delta\text{-CoVaR}/\Delta\text{-CoES}$. To show empirical performance simulations and an application of the methods to a large dataset of financial institutions are used. This paper finds that the MES is not suitable for measuring extreme risks. Also, the ES-based measures are more sensitive to power-law tails and large losses. This makes these measures more useful for measuring network risk but less so for systemic risk. Th
    
[^7]: 极化和交换：政党凝聚力的作用。

    Polarization and Quid Pro Quo: The Role of Party Cohesiveness. (arXiv:2205.07486v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2205.07486](http://arxiv.org/abs/2205.07486)

    研究表明，一个利益集团是否能够利用不断增加的意识形态和情感极化获利取决于两个政党社交网络相对凝聚度的聚合度量，而在没有意识形态 polarization 的情况下，两个政党社交网络的相对凝聚度则变得无关紧要。

    

    什么情况下，利益集团才能利用政党之间的意识形态和情感极化来获取利益？我们研究了一个模型，其中利益集团可信地承诺向立法者支付款项，条件是他们投票支持其支持的政策。议员们看重在党内的朋友投票，如果违反其党派的意识形态优先政策投票，则会受到一种意识形态上的不利。情感上的极化，由于其人际关系的本质，在模型中假设议员重视将其投票决策与反对党议员区分开来。我们的主要发现是，两个政党社交网络相对凝聚度的聚合度量确定了利益集团能否利用不断增加的极化获利。然而，如果两个政党之间没有意识形态 polarization，则相对凝聚度的意义将消失。

    When can an interest group exploit ideological and affective polarization between political parties to its advantage? We study a model where an interest group credibly promises payments to legislators conditional on voting for its favored policy. Legislators value voting as their friends within their party, and suffer an ideological-disutility upon voting against their party's ideologically preferred policy. Affective polarization, owing to its interpersonal nature, is modeled by assuming a legislator values distinguishing her voting decision from legislators in the opposite party. Our main finding is that an aggregate measure of relative cohesiveness of social networks in the two parties determines whether the interest group can profitably exploit increasing polarization. However, the significance of relative cohesiveness vanishes if there is no ideological polarization between the two parties.
    
[^8]: (无)妥协的政治：信息获取、政策裁量和声誉

    The Politics of (No) Compromise: Information Acquisition, Policy Discretion, and Reputation. (arXiv:2111.00522v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2111.00522](http://arxiv.org/abs/2111.00522)

    论文讨论了在政策裁量可以被限制的情况下，具有职业忧虑的决策者何时以及如何获取政策相关信息并进行改革决策，并指出公众可以通过在决策者的选择中消除温和政策或现状来鼓励信息获取。

    

    精确的信息对于制定良好的政策，尤其是涉及改革决策的政策至关重要。然而，决策者可能会犹豫不决地获取这些信息，因为某些决策可能对他们未来的职业生涯产生负面影响。我们模拟了具有职业忧虑的决策者在政策裁量可以在前被限制的情况下，如何获取政策相关信息并进行改革决策。通常情况下，具有职业忧虑的决策者相对于没有此类忧虑的决策者，获取信息的动力较弱。在这种情况下，我们证明公众可以通过在决策者的裁量中消除"温和政策"或现状来鼓励信息获取。我们还分析了何时应该将改革决策战略性地委托给具有或没有职业忧虑的决策者。

    Precise information is essential for making good policies, especially those regarding reform decisions. However, decision-makers may hesitate to gather such information if certain decisions could have negative impacts on their future careers. We model how decision-makers with career concerns may acquire policy-relevant information and carry out reform decisions when their policy discretion can be limited ex ante. Typically, decision-makers with career concerns have weaker incentives to acquire information compared to decision-makers without such concerns. In this context, we demonstrate that the public can encourage information acquisition by eliminating either the "moderate policy" or the status quo from decision-makers' discretion. We also analyze when reform decisions should be strategically delegated to decision-makers with or without career concerns.
    
[^9]: 福利与同意：关于骚扰的最优处罚

    Welfare v. Consent: On the Optimal Penalty for Harassment. (arXiv:2103.00734v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2103.00734](http://arxiv.org/abs/2103.00734)

    该论文提出了一种替代经济学方法，名为同意方法，以促进同意交互和防止非同意交互，突出了同意方法与福利方法之间的差异，其中同意方法对于骚扰的处罚不能为零。

    

    确定最优法律政策的经济学方法涉及最大化社会福利函数。我们提出了一种替代方法：同意方法，旨在促进同意交互和防止非同意交互。同意方法不依赖于人际效用比较或有关偏好的价值判断。它不需要任何相对于福利方法的额外信息。我们使用一个样式化模型来突出福利方法和同意方法之间的差异，该模型受到骚扰的经典案例和MeToo运动的启发。在我们的模型中，根据福利方法，骚扰的社会福利最大化惩罚可以为零，但根据同意方法则不然。

    The economic approach to determine optimal legal policies involves maximizing a social welfare function. We propose an alternative: a consent-approach that seeks to promote consensual interactions and deter non-consensual interactions. The consent-approach does not rest upon inter-personal utility comparisons or value judgments about preferences. It does not require any additional information relative to the welfare-approach. We highlight the contrast between the welfare-approach and the consent-approach using a stylized model inspired by seminal cases of harassment and the #MeToo movement. The social welfare maximizing penalty for harassment in our model can be zero under the welfare-approach but not under the consent-approach.
    


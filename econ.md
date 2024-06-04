# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Heterogeneous Productivity Effects of Generative AI](https://arxiv.org/abs/2403.01964) | 意大利对ChatGPT实施禁令后，不同经验的用户生产力表现出差异，经验较少的用户在短期内产出数量和质量均有提升，而经验丰富的用户在更常规的任务上表现出生产力下降。 |
| [^2] | [Machine Learning Who to Nudge: Causal vs Predictive Targeting in a Field Experiment on Student Financial Aid Renewal.](http://arxiv.org/abs/2310.08672) | 通过对学生进行因果与预测性定位的比较，研究探讨了在学生助学金续签领域实验中鼓励对象选择的价值。这项大规模实地实验揭示了定位干预对于不同学生的效果，为干预策略的优化提供了参考。 |
| [^3] | [Cheap Talking Algorithms.](http://arxiv.org/abs/2310.07867) | 该论文研究了在战略信息传递游戏中，利用独立强化学习算法进行训练的发送者和接收者可以收敛到接近最优均衡策略，并且在代理之间的利益冲突下实现了最大化的通信。这一结论稳健，并对信息传递游戏中的均衡选择理论、计算机科学中的算法间通信和人工智能代理市场中的经济学产生了影响。 |
| [^4] | [Identification and Estimation of a Semiparametric Logit Model using Network Data.](http://arxiv.org/abs/2310.07151) | 本文研究了半参数二元网络模型的识别和估计，提出了基于配对统计的估计方法，并通过蒙特卡洛模拟评估了该方法的性能。 |
| [^5] | [Testing for Stationary or Persistent Coefficient Randomness in Predictive Regressions.](http://arxiv.org/abs/2309.04926) | 本研究考虑了预测回归中系数随机性的检验，并发现在随机系数的持久性方面会影响各种检验的功效。我们建议在实际应用中根据潜在随机系数的持久性选择最合适的检验方法。 |
| [^6] | [DeepVol: A Deep Transfer Learning Approach for Universal Asset Volatility Modeling.](http://arxiv.org/abs/2309.02072) | DeepVol是一种用于通用资产波动性建模的深度迁移学习方法，通过一个通用模型有效地捕捉和建模所有金融资产的波动性动态，可能改变对波动性的理解和预测方式。 |
| [^7] | [Unpacking the Black Box: Regulating Algorithmic Decisions.](http://arxiv.org/abs/2110.03443) | 本文研究如何在代理使用复杂的“黑盒”预测函数进行决策的情况下，对算法决策进行最优调控。研究发现，限制代理使用透明度足够高的预测函数是低效的，而针对激励偏差源头的目标化工具可以提供次优解决方案，从而改善福利。 |

# 详细

[^1]: 生成式人工智能的异质生产力效应

    The Heterogeneous Productivity Effects of Generative AI

    [https://arxiv.org/abs/2403.01964](https://arxiv.org/abs/2403.01964)

    意大利对ChatGPT实施禁令后，不同经验的用户生产力表现出差异，经验较少的用户在短期内产出数量和质量均有提升，而经验丰富的用户在更常规的任务上表现出生产力下降。

    

    我们分析了意大利对ChatGPT（一种生成式预训练变换器聊天机器人）的禁令对个人生产力的影响。我们收集了意大利及其他欧洲国家超过36,000名GitHub用户的每日编码输出数量和质量数据，并将这些数据与该禁令的突然宣布结合起来，建立了一个差异性差异框架。在受影响的意大利用户中，我们发现对于经验较少的用户，输出数量和质量短期内增加，而对于经验丰富的用户而言，在更常规的任务上生产力降低。

    arXiv:2403.01964v1 Announce Type: cross  Abstract: We analyse the individual productivity effects of Italy's ban on ChatGPT, a generative pretrained transformer chatbot. We compile data on the daily coding output quantity and quality of over 36,000 GitHub users in Italy and other European countries and combine these data with the sudden announcement of the ban in a difference-in-differences framework. Among the affected users in Italy, we find a short-term increase in output quantity and quality for less experienced users and a decrease in productivity on more routine tasks for experienced users.
    
[^2]: 机器学习中的鼓励对象选择：在学生助学金续签领域实验中的因果与预测性目标定位

    Machine Learning Who to Nudge: Causal vs Predictive Targeting in a Field Experiment on Student Financial Aid Renewal. (arXiv:2310.08672v1 [econ.EM])

    [http://arxiv.org/abs/2310.08672](http://arxiv.org/abs/2310.08672)

    通过对学生进行因果与预测性定位的比较，研究探讨了在学生助学金续签领域实验中鼓励对象选择的价值。这项大规模实地实验揭示了定位干预对于不同学生的效果，为干预策略的优化提供了参考。

    

    在许多情境下，干预可能对某些人比其他人更有效，因此定位干预可能是有益的。我们通过一个规模庞大的实地实验（超过53,000名大学生）来分析在学生助学金续签前使用“鼓励”策略的价值。我们首先使用基线方法进行定位。首先，我们基于一个估计异质处理效应的因果森林进行定位，并根据估计出的拥有最高处理效应的学生来进行处理。接下来，我们评估两种替代的定位策略，一种是针对在没有干预的情况下预测到低助学金续签概率的学生，另一种是针对预测到高概率的学生。预测的基线结果并不是定位的理想标准，而且在先验上也不清楚是优先考虑低、高还是中间的预测。

    In many settings, interventions may be more effective for some individuals than others, so that targeting interventions may be beneficial. We analyze the value of targeting in the context of a large-scale field experiment with over 53,000 college students, where the goal was to use "nudges" to encourage students to renew their financial-aid applications before a non-binding deadline. We begin with baseline approaches to targeting. First, we target based on a causal forest that estimates heterogeneous treatment effects and then assigns students to treatment according to those estimated to have the highest treatment effects. Next, we evaluate two alternative targeting policies, one targeting students with low predicted probability of renewing financial aid in the absence of the treatment, the other targeting those with high probability. The predicted baseline outcome is not the ideal criterion for targeting, nor is it a priori clear whether to prioritize low, high, or intermediate predic
    
[^3]: 廉价对话算法

    Cheap Talking Algorithms. (arXiv:2310.07867v1 [econ.TH])

    [http://arxiv.org/abs/2310.07867](http://arxiv.org/abs/2310.07867)

    该论文研究了在战略信息传递游戏中，利用独立强化学习算法进行训练的发送者和接收者可以收敛到接近最优均衡策略，并且在代理之间的利益冲突下实现了最大化的通信。这一结论稳健，并对信息传递游戏中的均衡选择理论、计算机科学中的算法间通信和人工智能代理市场中的经济学产生了影响。

    

    我们模拟独立的强化学习算法在克劳福德和索贝尔（1982）的战略信息传递游戏中的行为。我们表明，一个发送者和一个接收者一起进行训练，收敛到接近游戏先验最优均衡的策略。因此，通信在与代理之间的利益冲突程度给出的纳什均衡下，按照最大程度进行。这一结论对超参数和游戏的备选规范稳健。我们讨论了信息传递游戏中均衡选择理论、计算机科学中算法间新兴通信工作以及由人工智能代理组成的市场中的宫斗经济学的影响。

    We simulate behaviour of independent reinforcement learning algorithms playing the Crawford and Sobel (1982) game of strategic information transmission. We show that a sender and a receiver training together converge to strategies close to the exante optimal equilibrium of the game. Hence, communication takes place to the largest extent predicted by Nash equilibrium given the degree of conflict of interest between agents. The conclusion is shown to be robust to alternative specifications of the hyperparameters and of the game. We discuss implications for theories of equilibrium selection in information transmission games, for work on emerging communication among algorithms in computer science and for the economics of collusions in markets populated by artificially intelligent agents.
    
[^4]: 使用网络数据进行半参数逻辑模型的识别和估计

    Identification and Estimation of a Semiparametric Logit Model using Network Data. (arXiv:2310.07151v1 [econ.EM])

    [http://arxiv.org/abs/2310.07151](http://arxiv.org/abs/2310.07151)

    本文研究了半参数二元网络模型的识别和估计，提出了基于配对统计的估计方法，并通过蒙特卡洛模拟评估了该方法的性能。

    

    本文研究了半参数二元网络模型的识别和估计，其中未观测到的社会特征是内生的，即未观测到的个体特征影响了感兴趣的二元结果以及在网络内部形成联系的方式。潜在社会特征的确切函数形式是未知的。所提出的估计量是基于网络形成分布相同的匹配对之间的配对。提出了估计量的一致性和渐近分布。通过蒙特卡洛模拟评估了所提出的估计量的有限样本特性。最后，对该研究进行了实证应用。

    This paper studies the identification and estimation of a semiparametric binary network model in which the unobserved social characteristic is endogenous, that is, the unobserved individual characteristic influences both the binary outcome of interest and how links are formed within the network. The exact functional form of the latent social characteristic is not known. The proposed estimators are obtained based on matching pairs of agents whose network formation distributions are the same. The consistency and the asymptotic distribution of the estimators are proposed. The finite sample properties of the proposed estimators in a Monte-Carlo simulation are assessed. We conclude this study with an empirical application.
    
[^5]: 预测回归中固定系数随机性的检验：稳态与持久性系数的影响

    Testing for Stationary or Persistent Coefficient Randomness in Predictive Regressions. (arXiv:2309.04926v1 [econ.EM])

    [http://arxiv.org/abs/2309.04926](http://arxiv.org/abs/2309.04926)

    本研究考虑了预测回归中系数随机性的检验，并发现在随机系数的持久性方面会影响各种检验的功效。我们建议在实际应用中根据潜在随机系数的持久性选择最合适的检验方法。

    

    本研究考虑了预测回归中系数随机性的检验。我们关注系数随机性检验在随机系数的持久性方面的影响。我们发现，当随机系数是稳态的或I(0)时，Nyblom的LM检验在功效上不是最优的，这一点已经针对集成或I(1)随机系数的备择假设得到了证实。我们通过构建一些在随机系数为稳态时具有更高功效的检验来证明这一点，尽管在随机系数为集成时，这些检验在功效上被LM检验所支配。这意味着在不同的背景下，系数随机性的最佳检验是不同的，从而实证研究者应该考虑潜在随机系数的持久性，并相应地选择多个检验。特别是，我们通过理论和数值研究表明，LM检验与一种Wald型检验的乘积是一个较好的检验方法。

    This study considers tests for coefficient randomness in predictive regressions. Our focus is on how tests for coefficient randomness are influenced by the persistence of random coefficient. We find that when the random coefficient is stationary, or I(0), Nyblom's (1989) LM test loses its optimality (in terms of power), which is established against the alternative of integrated, or I(1), random coefficient. We demonstrate this by constructing tests that are more powerful than the LM test when random coefficient is stationary, although these tests are dominated in terms of power by the LM test when random coefficient is integrated. This implies that the best test for coefficient randomness differs from context to context, and practitioners should take into account the persistence of potentially random coefficient and choose from several tests accordingly. In particular, we show through theoretical and numerical investigations that the product of the LM test and a Wald-type test proposed
    
[^6]: DeepVol：一种用于通用资产波动性建模的深度迁移学习方法

    DeepVol: A Deep Transfer Learning Approach for Universal Asset Volatility Modeling. (arXiv:2309.02072v1 [econ.EM])

    [http://arxiv.org/abs/2309.02072](http://arxiv.org/abs/2309.02072)

    DeepVol是一种用于通用资产波动性建模的深度迁移学习方法，通过一个通用模型有效地捕捉和建模所有金融资产的波动性动态，可能改变对波动性的理解和预测方式。

    

    本文介绍了一种新的深度学习波动性模型DeepVol，它在模型的广泛性方面优于传统的计量经济模型。DeepVol利用迁移学习的能力，通过一个通用模型有效地捕捉和建模所有金融资产的波动性动态，包括以前未见过的资产。这与计量经济学文献中的主流做法形成鲜明对比，后者需要为不同数据集训练单独的模型。引入DeepVol为金融行业的波动性建模和预测开辟了新的途径，可能会改变对波动性的理解和预测方式。

    This paper introduces DeepVol, a promising new deep learning volatility model that outperforms traditional econometric models in terms of model generality. DeepVol leverages the power of transfer learning to effectively capture and model the volatility dynamics of all financial assets, including previously unseen ones, using a single universal model. This contrasts to the prevailing practice in econometrics literature, which necessitates training separate models for individual datasets. The introduction of DeepVol opens up new avenues for volatility modeling and forecasting in the finance industry, potentially transforming the way volatility is understood and predicted.
    
[^7]: 揭开黑盒子：调控算法决策

    Unpacking the Black Box: Regulating Algorithmic Decisions. (arXiv:2110.03443v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2110.03443](http://arxiv.org/abs/2110.03443)

    本文研究如何在代理使用复杂的“黑盒”预测函数进行决策的情况下，对算法决策进行最优调控。研究发现，限制代理使用透明度足够高的预测函数是低效的，而针对激励偏差源头的目标化工具可以提供次优解决方案，从而改善福利。

    

    我们展示了如何在一个代理使用复杂的“黑盒”预测函数进行决策（如贷款、医疗测试或招聘）且委托人在了解代理的黑盒模型方面有限的情况下，最优地调控预测算法。我们证明，只要诱导不足，且最优预测函数足够复杂，将代理限制在足够透明的预测函数中是低效的。算法审计有助于提高福利，但其收益取决于审计工具的设计。许多解释工具倾向于最小化整体信息损失，但这通常是低效的，因为它们集中于解释预测函数的平均行为。针对性的工具，如针对激励偏差源头（如过多的假阳性或种族差异）的工具，可以提供次优解决方案。我们提供了对我们理论的实证支持。

    We show how to optimally regulate prediction algorithms in a world where an agent uses complex 'black-box' prediction functions to make decisions such as lending, medical testing, or hiring, and where a principal is limited in how much she can learn about the agent's black-box model. We show that limiting agents to prediction functions that are simple enough to be fully transparent is inefficient as long as the misalignment is limited and first-best prediction functions are sufficiently complex. Algorithmic audits can improve welfare, but the gains depend on the design of the audit tools. Tools that focus on minimizing overall information loss, the focus of many explainer tools, will generally be inefficient since they focus on explaining the average behavior of the prediction function. Targeted tools that focus on the source of incentive misalignment, e.g., excess false positives or racial disparities, can provide second-best solutions. We provide empirical support for our theoretical
    


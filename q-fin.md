# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust SGLD algorithm for solving non-convex distributionally robust optimisation problems](https://arxiv.org/abs/2403.09532) | 本文开发了一种用于解决非凸分布鲁棒优化问题的稳健SGLD算法，并通过实证分析表明，该算法可以优于传统SGLD算法在金融投资组合优化中的应用。 |
| [^2] | [Metawisdom of the Crowd: How Choice Within Aided Decision Making Can Make Crowd Wisdom Robust.](http://arxiv.org/abs/2308.15451) | 本论文研究了群体智慧的元智慧，即群体对辅助决策的集体选择方式如何使群体的准确性提高。通过研究发现，个体以不同的方式选择关注不同的信息可以增强群体智慧的预测多样性。 |

# 详细

[^1]: 用于解决非凸分布鲁棒优化问题的稳健SGLD算法

    Robust SGLD algorithm for solving non-convex distributionally robust optimisation problems

    [https://arxiv.org/abs/2403.09532](https://arxiv.org/abs/2403.09532)

    本文开发了一种用于解决非凸分布鲁棒优化问题的稳健SGLD算法，并通过实证分析表明，该算法可以优于传统SGLD算法在金融投资组合优化中的应用。

    

    在本文中，我们开发了一种特定类型的非凸分布鲁棒优化问题的随机梯度 Langevin 动力学（SGLD）算法。通过推导非渐近收敛界限，我们构建了一种算法，对于任何预先指定的精度$\varepsilon>0$，该算法输出的估计值的期望超额风险最多为$\varepsilon$。作为一个具体的应用，我们使用我们的稳健SGLD算法来解决（正则化的）分布鲁棒 Mean-CVaR 投资组合优化问题，并使用真实的金融数据。我们在经验上证明，通过我们的稳健SGLD算法获得的交易策略优于使用传统的 SGLD 算法解决相应的非鲁棒 Mean-CVaR 投资组合优化问题获得的交易策略。这突显了在优化实际金融市场投资组合时纳入模型不确定性的实际相关性。

    arXiv:2403.09532v1 Announce Type: cross  Abstract: In this paper we develop a Stochastic Gradient Langevin Dynamics (SGLD) algorithm tailored for solving a certain class of non-convex distributionally robust optimisation problems. By deriving non-asymptotic convergence bounds, we build an algorithm which for any prescribed accuracy $\varepsilon>0$ outputs an estimator whose expected excess risk is at most $\varepsilon$. As a concrete application, we employ our robust SGLD algorithm to solve the (regularised) distributionally robust Mean-CVaR portfolio optimisation problem using real financial data. We empirically demonstrate that the trading strategy obtained by our robust SGLD algorithm outperforms the trading strategy obtained when solving the corresponding non-robust Mean-CVaR portfolio optimisation problem using, e.g., a classical SGLD algorithm. This highlights the practical relevance of incorporating model uncertainty when optimising portfolios in real financial markets.
    
[^2]: 群体的元智慧：如何通过选择辅助决策来使群体智慧更加稳健。

    Metawisdom of the Crowd: How Choice Within Aided Decision Making Can Make Crowd Wisdom Robust. (arXiv:2308.15451v1 [econ.GN])

    [http://arxiv.org/abs/2308.15451](http://arxiv.org/abs/2308.15451)

    本论文研究了群体智慧的元智慧，即群体对辅助决策的集体选择方式如何使群体的准确性提高。通过研究发现，个体以不同的方式选择关注不同的信息可以增强群体智慧的预测多样性。

    

    优质的信息可以提高个体判断，但却无法使群体决策更加准确；如果个体以相同的方式选择关注相同的信息，那么赋予群体智慧以预测多样性的机会可能会丧失。决策支持系统，从商业智能软件到公共搜索引擎，通过提供决策辅助来增强决策质量和速度，包括相关信息的离散展示、解释框架或启发式方法，但这些系统也有可能通过选择性展示信息和解释框架来引入判断偏见。我们重新描述群体智慧，将其描述为常常有两个决策，即决策辅助的选择和主要决策。然后，我们将"群体的元智慧"定义为群体对辅助决策的集体选择方式，使群体的准确性高于以同一辅助方式随机分配的比较结果，这种比较考虑了信息内容。

    Quality information can improve individual judgments but nonetheless fail to make group decisions more accurate; if individuals choose to attend to the same information in the same way, the predictive diversity that enables crowd wisdom may be lost. Decision support systems, from business intelligence software to public search engines, present individuals with decision aids -- discrete presentations of relevant information, interpretative frames, or heuristics -to enhance the quality and speed of decision making, but have the potential to bias judgments through the selective presentation of information and interpretative frames. We redescribe the wisdom of the crowd as often having two decisions, the choice of decision aids and then the primary decision. We then define \emph{metawisdom of the crowd} as any pattern by which the collective choice of aids leads to higher crowd accuracy than randomized assignment to the same aids, a comparison that accounts for the information content of
    


# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust SGLD algorithm for solving non-convex distributionally robust optimisation problems](https://arxiv.org/abs/2403.09532) | 本文开发了一种用于解决非凸分布鲁棒优化问题的稳健SGLD算法，并通过实证分析表明，该算法可以优于传统SGLD算法在金融投资组合优化中的应用。 |
| [^2] | [Detecting Consumers' Financial Vulnerability using Open Banking Data: Evidence from UK Payday Loans.](http://arxiv.org/abs/2306.01749) | 本文利用混合Poisson隐Markov方法，以英国发薪日贷款为例，探究了债务陷阱背后的理念，即发薪日贷款加剧了消费者的财务脆弱性，发现某些时变协变量的影响在很大程度上取决于借款人的隐状态。 |

# 详细

[^1]: 用于解决非凸分布鲁棒优化问题的稳健SGLD算法

    Robust SGLD algorithm for solving non-convex distributionally robust optimisation problems

    [https://arxiv.org/abs/2403.09532](https://arxiv.org/abs/2403.09532)

    本文开发了一种用于解决非凸分布鲁棒优化问题的稳健SGLD算法，并通过实证分析表明，该算法可以优于传统SGLD算法在金融投资组合优化中的应用。

    

    在本文中，我们开发了一种特定类型的非凸分布鲁棒优化问题的随机梯度 Langevin 动力学（SGLD）算法。通过推导非渐近收敛界限，我们构建了一种算法，对于任何预先指定的精度$\varepsilon>0$，该算法输出的估计值的期望超额风险最多为$\varepsilon$。作为一个具体的应用，我们使用我们的稳健SGLD算法来解决（正则化的）分布鲁棒 Mean-CVaR 投资组合优化问题，并使用真实的金融数据。我们在经验上证明，通过我们的稳健SGLD算法获得的交易策略优于使用传统的 SGLD 算法解决相应的非鲁棒 Mean-CVaR 投资组合优化问题获得的交易策略。这突显了在优化实际金融市场投资组合时纳入模型不确定性的实际相关性。

    arXiv:2403.09532v1 Announce Type: cross  Abstract: In this paper we develop a Stochastic Gradient Langevin Dynamics (SGLD) algorithm tailored for solving a certain class of non-convex distributionally robust optimisation problems. By deriving non-asymptotic convergence bounds, we build an algorithm which for any prescribed accuracy $\varepsilon>0$ outputs an estimator whose expected excess risk is at most $\varepsilon$. As a concrete application, we employ our robust SGLD algorithm to solve the (regularised) distributionally robust Mean-CVaR portfolio optimisation problem using real financial data. We empirically demonstrate that the trading strategy obtained by our robust SGLD algorithm outperforms the trading strategy obtained when solving the corresponding non-robust Mean-CVaR portfolio optimisation problem using, e.g., a classical SGLD algorithm. This highlights the practical relevance of incorporating model uncertainty when optimising portfolios in real financial markets.
    
[^2]: 利用开放银行数据检测消费者的财务脆弱性：以英国的发薪日贷款为例

    Detecting Consumers' Financial Vulnerability using Open Banking Data: Evidence from UK Payday Loans. (arXiv:2306.01749v1 [stat.AP])

    [http://arxiv.org/abs/2306.01749](http://arxiv.org/abs/2306.01749)

    本文利用混合Poisson隐Markov方法，以英国发薪日贷款为例，探究了债务陷阱背后的理念，即发薪日贷款加剧了消费者的财务脆弱性，发现某些时变协变量的影响在很大程度上取决于借款人的隐状态。

    

    债务陷阱背后的理念是，发薪日贷款加剧了消费者的财务脆弱性。 为了调查这种关系，我们提出了一种混合Poisson隐Markov方法来模拟借款人每期获得发薪日贷款的数量。 鉴于文献中对于财务脆弱性缺乏一致性，我们利用一个隐Markov过程（脆弱和非脆弱）引入金融困境作为一个未观测到的二元变量。利用来自1,817名英国消费者的90,523个匿名交易数据，我们发现某些时变协变量的影响在很大程度上取决于借款人的隐状态。例如，在财务脆弱时，奢侈开支和非经常性收入增加了发薪日贷款的需求，但在非脆弱时则相反。此外，我们证明了近60％的发薪日贷款借款人在连续12周或更长时间保持脆弱状态，其中三分之二的人面临着持续的财务困难。

    Behind the debt trap concept is the rationale that payday loans exacerbate consumers' financial vulnerability. To investigate this relationship, we propose a Mixed Poisson Hidden Markov approach to model the number of payday loans a borrower obtains in each period. Given the lack of agreement in the literature on financial vulnerability, we introduce financial distress as an unobserved binary variable using a hidden Markov process (vulnerable and non-vulnerable). Using data from 90,523 anonymised transactions for 1,817 UK consumers, we find that the effect of certain time-varying covariates depends greatly on the borrower's hidden state. For instance, luxury expenses and non-recurring income increase the need for payday loans when financially vulnerable, but the opposite is true when not vulnerable. Additionally, we demonstrate that almost 60\% of payday loan borrowers remain vulnerable for 12 or more consecutive weeks, with two-thirds experiencing consistent financial difficulties. Fi
    


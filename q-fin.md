# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Matching under Imperfectly Transferable Utility](https://arxiv.org/abs/2403.05222) | 本文研究了不完全可转移效用下的匹配模型及其估计方法，为匹配经济学领域提供了重要贡献。 |
| [^2] | [Regret-Optimal Federated Transfer Learning for Kernel Regression with Applications in American Option Pricing.](http://arxiv.org/abs/2309.04557) | 本论文提出了一种遗憾最优算法的迭代方案，用于联邦迁移学习，在核回归模型中具体化，并提出了一个几乎遗憾最优的启发式算法，可以减小生成参数与专门参数之间的累积偏差。 |
| [^3] | [Why Are Immigrants Always Accused of Stealing People's Jobs?.](http://arxiv.org/abs/2303.13319) | 移民在一个配给职位的匹配模型中可能降低本地工人的就业率，但移民对本地福利的总体影响取决于劳动力市场的状态，当劳动力市场繁荣时，移民的影响可能是正面的。 |

# 详细

[^1]: 不完全可转移效用下的匹配

    Matching under Imperfectly Transferable Utility

    [https://arxiv.org/abs/2403.05222](https://arxiv.org/abs/2403.05222)

    本文研究了不完全可转移效用下的匹配模型及其估计方法，为匹配经济学领域提供了重要贡献。

    

    在本文中，我们研究了具有不完全可转移效用（ITU）的匹配模型。我们提供了激励性的例子，讨论了ITU匹配模型的理论基础，并介绍了估计它们的方法。我们还探讨了相关主题，并概述了相关文献。本文已作为《匹配经济学手册》的一章草案提交，由Che、Chiappori和Salani'e编辑。

    arXiv:2403.05222v1 Announce Type: new  Abstract: In this paper, we examine matching models with imperfectly transferable utility (ITU). We provide motivating examples, discuss the theoretical foundations of ITU matching models and present methods for estimating them. We also explore connected topics and provide an overview of the related literature. This paper has been submitted as a draft chapter for the Handbook of the Economics of Matching, edited by Che, Chiappori and Salani\'e.
    
[^2]: 用于核回归的遗憾最优联邦迁移学习及其在美式期权定价中的应用

    Regret-Optimal Federated Transfer Learning for Kernel Regression with Applications in American Option Pricing. (arXiv:2309.04557v1 [cs.LG])

    [http://arxiv.org/abs/2309.04557](http://arxiv.org/abs/2309.04557)

    本论文提出了一种遗憾最优算法的迭代方案，用于联邦迁移学习，在核回归模型中具体化，并提出了一个几乎遗憾最优的启发式算法，可以减小生成参数与专门参数之间的累积偏差。

    

    我们提出了一种最优的迭代方案，用于联邦迁移学习，其中中心计划者可以访问同一学习模型 $f_{\theta}$ 的数据集 ${\cal D}_1,\dots,{\cal D}_N$。我们的目标是在尊重模型 $f_{\theta(T)}$ 的损失函数的情况下，尽量减小生成参数 $\{\theta_i(t)\}_{t=0}^T$ 在所有 $T$ 次迭代中与每个数据集得到的专门参数$\theta^\star_{1},\ldots,\theta^\star_N$ 的累积偏差。我们仅允许每个专门模型（节点/代理）和中心计划者（服务器）在每次迭代（轮）之间进行持续通信。对于模型 $f_{\theta}$ 是有限秩核回归的情况，我们得出了遗憾最优算法的显式更新公式。通过利用遗憾最优算法中的对称性，我们进一步开发了一种几乎遗憾最优的启发式算法，其运行需要较少的 $\mathcal{O}(Np^2)$ 个基本运算。

    We propose an optimal iterative scheme for federated transfer learning, where a central planner has access to datasets ${\cal D}_1,\dots,{\cal D}_N$ for the same learning model $f_{\theta}$. Our objective is to minimize the cumulative deviation of the generated parameters $\{\theta_i(t)\}_{t=0}^T$ across all $T$ iterations from the specialized parameters $\theta^\star_{1},\ldots,\theta^\star_N$ obtained for each dataset, while respecting the loss function for the model $f_{\theta(T)}$ produced by the algorithm upon halting. We only allow for continual communication between each of the specialized models (nodes/agents) and the central planner (server), at each iteration (round). For the case where the model $f_{\theta}$ is a finite-rank kernel regression, we derive explicit updates for the regret-optimal algorithm. By leveraging symmetries within the regret-optimal algorithm, we further develop a nearly regret-optimal heuristic that runs with $\mathcal{O}(Np^2)$ fewer elementary operati
    
[^3]: 为什么移民总被指责窃取人们的工作?

    Why Are Immigrants Always Accused of Stealing People's Jobs?. (arXiv:2303.13319v1 [econ.GN])

    [http://arxiv.org/abs/2303.13319](http://arxiv.org/abs/2303.13319)

    移民在一个配给职位的匹配模型中可能降低本地工人的就业率，但移民对本地福利的总体影响取决于劳动力市场的状态，当劳动力市场繁荣时，移民的影响可能是正面的。

    

    移民总是被指责窃取人们的工作。然而，在劳动力市场的新古典模型中，每个人都有工作可做，也没有工作可被窃取（没有失业，因此想工作的人都可以工作）。在标准匹配模型中，存在一些失业，但由于劳动力需求完全弹性，因此新进入劳动力市场的人被吸收时不会影响求职者的前景。再次说明，当移民到达时没有工作会被窃取。本文显示，在一个具有就业配给的匹配模型中，移民的进入会降低本地工人的就业率。此外，当劳动力市场不景气时，就业率的降幅更大，因为那时工作更加稀缺。因为移民降低了劳动力市场的紧张程度，使得公司更容易招聘，并改善公司利润。移民对本地福利的总体影响取决于劳动力市场的状态。当劳动力市场出现衰退时总体影响始终是负面的，并且当劳动力市场繁荣时可能是正面的。

    Immigrants are always accused of stealing people's jobs. Yet, in a neoclassical model of the labor market, there are jobs for everybody and no jobs to steal. (There is no unemployment, so anybody who wants to work can work.) In standard matching models, there is some unemployment, but labor demand is perfectly elastic so new entrants into the labor force are absorbed without affecting jobseekers' prospects. Once again, no jobs are stolen when immigrants arrive. This paper shows that in a matching model with job rationing, in contrast, the entry of immigrants reduces the employment rate of native workers. Moreover, the reduction in employment rate is sharper when the labor market is depressed -- because jobs are more scarce then. Because immigration reduces labor-market tightness, it makes it easier for firms to recruit and improves firm profits. The overall effect of immigration on native welfare depends on the state of the labor market. It is always negative when the labor market is i
    


# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Implicit Bias of Policy Gradient in Linear Quadratic Control: Extrapolation to Unseen Initial States](https://arxiv.org/abs/2402.07875) | 本文研究了策略梯度在线性二次调节控制中对未见初始状态的外推问题，发现外推程度取决于训练中系统的探索程度。 |
| [^2] | [Generative Modeling of Discrete Joint Distributions by E-Geodesic Flow Matching on Assignment Manifolds](https://arxiv.org/abs/2402.07846) | 本文提出了一种基于连续归一化流的生成模型，该模型可以逐步分配类别，避免了离散化潜在连续模型时的舍入和样本截断等问题。通过匹配分解离散分布的测地线流，可以高效地训练该模型，并且适用于表示复杂统计依赖关系的非分解离散分布。 |
| [^3] | [On Computationally Efficient Multi-Class Calibration](https://arxiv.org/abs/2402.07821) | 提出了一种在多类别预测问题中多样化的投影平滑校准概念，并且给出了多项式时间复杂度的重新校准算法，从而实现了计算效率和强大的预测保证之间的权衡。 |
| [^4] | [Towards a mathematical theory for consistency training in diffusion models](https://arxiv.org/abs/2402.07802) | 本文探索了面向扩散模型中一致性训练的理论基础，证明了在一致性学习中，步骤数量需要超过$d^{5/2}/\varepsilon$的阶数，能够生成与目标分布接近的样本。 |
| [^5] | [Tuning-Free Stochastic Optimization](https://arxiv.org/abs/2402.07793) | 本文提出了一种无调参的随机优化算法，能够在只给出问题参数的粗略提示的情况下，与最优调参优化算法的性能相匹配。并且在有界的优化领域中证明了此算法的可行性，并探讨了在无界域中的条件。 |
| [^6] | [Scalable Structure Learning for Sparse Context-Specific Causal Systems](https://arxiv.org/abs/2402.07762) | 提出了一种可扩展的混合算法，用于学习特定背景模型，通过结合基于顺序的MCMC算法和稀疏性假设实现可扩展学习，该方法在准确性和可扩展性方面表现良好。 |
| [^7] | [Optimal score estimation via empirical Bayes smoothing](https://arxiv.org/abs/2402.07747) | 该论文研究了通过经验贝叶斯平滑在高维数据中估计未知概率分布的分数函数的问题，提出了一种基于高斯核的正则化分数估计器，在score matching损失函数下达到了最优速率，并揭示了维度增长对样本复杂性的指数级影响。 |
| [^8] | [Graph Structure Inference with BAM: Introducing the Bilinear Attention Mechanism](https://arxiv.org/abs/2402.07735) | 本论文提出了一种利用BAM进行图结构推断的方法。通过神经网络模型，通过变形的耦合模拟输入数据进行训练，仅需通过一次前向传递即可进行推断。通过利用结构方程模型和随机生成的多变量切比雪夫多项式来模拟训练数据，方法能够泛化到线性和各种非线性依赖关系。引入了双线性注意机制（BAM）来处理依赖关系，该机制在转换数据的协方差矩阵水平上运行，并尊重对称正定矩阵流形的几何特性。实证评估证明了方法的有效性和性能。 |
| [^9] | [Generalization Bounds for Heavy-Tailed SDEs through the Fractional Fokker-Planck Equation](https://arxiv.org/abs/2402.07723) | 本论文通过分数阻尼库仑方程证明了重尾SDE的高概率泛化界限，并且相对于参数维度，界限的依赖性要好于p。 |
| [^10] | [Efficient reductions between some statistical models](https://arxiv.org/abs/2402.07717) | 本研究提出了一种在不知道源统计模型参数的情况下，高效地将样本从源模型转换为目标模型的方法，并构造了几个归约方法。这些归约方法能适应不同的问题，例如专家混合模型、相位恢复和信号降噪等，并且可以处理缺失数据。此外，该研究还指出了一个潜在的应用，即将一个差分隐私机制转换为另一个机制。 |
| [^11] | [Model Collapse Demystified: The Case of Regression](https://arxiv.org/abs/2402.07712) | 本研究在核回归的简化环境中解析了模型崩溃现象，并发现了模型能够处理虚假数据与性能完全崩溃之间的交叉点。通过提出基于自适应正则化的策略，成功缓解了模型崩溃问题。这些发现通过实验证实。 |
| [^12] | [Stochastic Gradient Flow Dynamics of Test Risk and its Exact Solution for Weak Features](https://arxiv.org/abs/2402.07626) | 本研究通过路径积分方法探索了连续时间随机梯度流动力学中的测试风险，并在小学习率情况下给出了计算纯梯度流动和随机梯度流动的测试风险曲线之间差异的一般公式。通过应用于一个弱特征模型，我们分析了随机项对动力学的修正效果，并与离散时间随机梯度下降的模拟结果进行了比较，结果显示出一致性。 |
| [^13] | [Global optimality under amenable symmetry constraints](https://arxiv.org/abs/2402.07613) | 该论文研究了在可接受的对称约束条件下的全局最优性问题，提出了一种满足对称性质的函数或度量，并通过引入轨道凸体和coycle等工具解决了这一问题。具体应用包括不变核均值嵌入和基于对称约束的运输方案最优性。这些结果与不变性检验的Hunt-Stein定理相关。 |
| [^14] | [Near-Minimax-Optimal Distributional Reinforcement Learning with a Generative Model](https://arxiv.org/abs/2402.07598) | 本论文提出了一种基于生成模型的近最小极大分布式强化学习算法，该算法在使用生成模型近似回报分布方面具有极小极大优势，解决了一个开放问题，并提供了实验研究结果。 |
| [^15] | [Rethinking Scaling Laws for Learning in Strategic Environments](https://arxiv.org/abs/2402.07588) | 本文重新思考了在战略环境中学习的比例定律，发现战略互动可以打破传统的观点，即模型越大或表达能力越强并不一定会随之提高性能。通过几个战略环境的例子，我们展示了这种现象的影响。 |
| [^16] | [Weisfeiler-Leman at the margin: When more expressivity matters](https://arxiv.org/abs/2402.07568) | 研究探讨了1-WL算法在图同构问题中的表达能力和泛化性能之间的关系，发现增强的表达能力对提高泛化性能并不总是有效。此外，通过引入子图信息和经典的边缘理论，探索了更高表达力与改进泛化性能的条件。梯度流也被证明可以促进模型学习更丰富的表达能力。 |
| [^17] | [A step towards the integration of machine learning and small area estimation](https://arxiv.org/abs/2402.07521) | 本文提出了一个基于机器学习算法的预测模型，可以根据横断面和纵向数据预测任何人群或子人群的特征，并分析了在实际生活中更重要的背景下的性能。 |
| [^18] | [Score-Based Physics-Informed Neural Networks for High-Dimensional Fokker-Planck Equations](https://arxiv.org/abs/2402.07465) | 这项研究提出了一种基于得分函数的求解器来解决高维福克-普朗克方程中的维数灾难问题。与蒙特卡洛和普通PINN相比，该方法能够更准确地处理与布朗运动相关的概率密度函数，并提供快速采样。 |
| [^19] | [On the Distance from Calibration in Sequential Prediction](https://arxiv.org/abs/2402.07458) | 本论文研究了顺序预测中的标定距离，证明了存在一种预测算法可以在敌人选择的二进制序列上实现$O(\sqrt{T})$的标定距离，通过较低的标定距离进行准确近似。 |
| [^20] | [Bandit-Feedback Online Multiclass Classification: Variants and Tradeoffs](https://arxiv.org/abs/2402.07453) | 该论文研究了在对抗在线环境中多类分类中依赖于强盗反馈的代价，自适应对手和随机学习者与无视对手和确定性学习者之间的损失差距。 |
| [^21] | [Top-$K$ ranking with a monotone adversary](https://arxiv.org/abs/2402.07445) | 本文针对具有单调对手的Top-K排名问题，提出了一种加权最大似然估计器(MLE)，在样本复杂度方面接近最优。算法创新包括了对加权MLE的精确且紧密的$\ell_\infty$误差分析，并与加权比较图的谱特性相关联。 |
| [^22] | [Conditional Generative Models are Sufficient to Sample from Any Causal Effect Estimand](https://arxiv.org/abs/2402.07419) | 本文展示了通过条件生成模型的推进计算可以计算任何可辨识的因果效应，并提出了基于扩散的方法用于从图像的任何（条件）干预分布中进行采样。 |
| [^23] | [Conformal Predictive Programming for Chance Constrained Optimization](https://arxiv.org/abs/2402.07407) | 可容许预测规划（CPP）是一种解决受任意随机参数影响的优化问题的方法，通过利用样本和量子引理将机遇受限优化（CCO）问题转化为确定性优化问题，并具备边际概率可行性保证。 |
| [^24] | [Replicability is Asymptotically Free in Multi-armed Bandits](https://arxiv.org/abs/2402.07391) | 本论文研究在多臂赌博机问题中，通过引入探索-再确定算法和连续淘汰算法，以及谨慎选择置信区间的幅度，实现了可复制性，并证明了当时间界足够大时，可复制算法的额外代价是不必要的。 |
| [^25] | [The Limits of Assumption-free Tests for Algorithm Performance](https://arxiv.org/abs/2402.07388) | 这项研究探讨了使用有限数据量回答算法性能问题的基本限制，证明了黑盒测试方法无法准确回答算法在不同训练集上的整体性能和特定模型的性能问题。 |
| [^26] | [Regression Trees for Fast and Adaptive Prediction Intervals](https://arxiv.org/abs/2402.07357) | 该论文提出了一种新的、与模型无关的方法族，用于校准具有局部覆盖保证的回归问题的预测区间。这种方法利用回归树和随机森林训练来创建最粗糙的特征空间划分，以近似条件覆盖，提供了准确、快速和自适应的预测区间。 |
| [^27] | [A Novel Gaussian Min-Max Theorem and its Applications](https://arxiv.org/abs/2402.07356) | 本文介绍了一个新的高斯最小最大定理，扩展了经典定理对于独立但非恒定分布的情况。此外，该定理在高维统计学、机器学习、非光滑优化和信号处理等领域有广泛的应用。 |
| [^28] | [Sampling from the Mean-Field Stationary Distribution](https://arxiv.org/abs/2402.07355) | 本文研究了从均场随机微分方程 (SDE) 的稳态分布中采样的复杂性，并提出了一种解耦的方法。该方法能够在多种情况下提供改进的保证，包括在均场区域优化某些双层神经网络的更好保证。 |
| [^29] | [Noise-Adaptive Confidence Sets for Linear Bandits and Application to Bayesian Optimization](https://arxiv.org/abs/2402.07341) | 这项研究提出了一种对线性强化学习领域中未知噪声水平的自适应置信区间，与已有方法相比，在维度较大时具有显著的改进。此外，针对有界奖励，还提出了一种方差自适应置信区间，具有更好的数值性能。 |
| [^30] | [Random Geometric Graph Alignment with Graph Neural Networks](https://arxiv.org/abs/2402.07340) | 本文研究了在图对齐问题中，通过图神经网络可以高概率恢复正确的顶点对齐。通过特定的特征稀疏性和噪声水平条件，我们证明了图神经网络的有效性，并与直接匹配方法进行了比较。 |
| [^31] | [A Theoretical Analysis of Nash Learning from Human Feedback under General KL-Regularized Preference](https://arxiv.org/abs/2402.07314) | 本论文从理论层面分析了一种关于一般偏好下纳什学习从人类反馈中的方法，通过对两个竞争的LLM进行博弈来找到一种一致生成响应的策略。 |
| [^32] | [HyperBERT: Mixing Hypergraph-Aware Layers with Language Models for Node Classification on Text-Attributed Hypergraphs](https://arxiv.org/abs/2402.07309) | 本文提出了HyperBERT模型，通过在预训练的BERT模型中引入超图感知层，克服了现有方法在节点分类任务上难以捕捉超图结构信息和文本属性的局限性，提高了模型的效果和泛化能力。 |
| [^33] | [Self-Consistent Conformal Prediction](https://arxiv.org/abs/2402.07307) | 自洽的符合预测方法能够提供既符合校准的预测又符合以模型预测的动作为条件的预测区间，为决策者提供了严格的、针对具体动作的决策保证。 |
| [^34] | [Estimating the Mixing Coefficients of Geometrically Ergodic Markov Processes](https://arxiv.org/abs/2402.07296) | 该论文提出了一种方法来估计几何遗传马尔可夫过程的混合系数，我们通过在满足特定条件和无需密度假设的情况下，得到了估计器的预期误差收敛速度和高概率界限。 |
| [^35] | [Depth Separations in Neural Networks: Separating the Dimension from the Accuracy](https://arxiv.org/abs/2402.07248) | 通过研究深度2和深度3神经网络在逼近Lipschitz目标函数时的分离性质，证明了维度诅咒也会在深度2逼近中存在，即使目标函数可以使用深度3高效表示。这为以前确定深度要求的下界提供了新的观点，并且适用于多种激活函数。 |
| [^36] | [Towards Fast Stochastic Sampling in Diffusion Generative Models](https://arxiv.org/abs/2402.07211) | 本文提出了一种在扩散生成模型中进行快速随机采样的方法，通过对分裂积分器进行原则性修改，实现了更高的采样效率。在CIFAR-10数据集上进行实验，100次网络函数评估下的FID分数为2.36。 |
| [^37] | [The Implicit Bias of Gradient Noise: A Symmetry Perspective](https://arxiv.org/abs/2402.07193) | 本研究通过对对称性的存在进行分析，揭示了梯度噪声在随机梯度下降中的隐性偏见。我们发现不同类型的对称性会导致不同的学习动态，其中一类对称性可以自然收敛，而另一类对称性几乎总是发散。此外，我们的研究结果适用于没有对称性的损失函数，对于理解训练动态和解释相关实际问题具有普适性。 |
| [^38] | [Improving LSH via Tensorized Random Projection](https://arxiv.org/abs/2402.07189) | 本文提出了CP-E2LSH和TT-E2LSH两种方法，用于改进局部敏感哈希算法LSH，在处理张量数据的欧几里得距离和余弦相似度时能够提供更快和更空间有效的结果。 |
| [^39] | [PASOA- PArticle baSed Bayesian Optimal Adaptive design](https://arxiv.org/abs/2402.07160) | PASOA是一种新的贝叶斯实验设计程序，通过提供连续的后验分布的准确估计，同时执行顺序设计优化和参数推断。该方法使用 stochastic optimization 和 tempered SMC 来最大化期望信息增益，并提供了一致性的最优设计估计。 |
| [^40] | [Resampling methods for Private Statistical Inference](https://arxiv.org/abs/2402.07131) | 这项研究提出了两种私有变体的非参数bootstrap方法，用于在差分隐私的情况下构建置信区间。方法在计算效率和置信区间长度上相比现有方法有显著改进。 |
| [^41] | [Towards Quantifying the Preconditioning Effect of Adam](https://arxiv.org/abs/2402.07114) | 本论文量化了Adam的预条件效应，结果表明Adam能够减轻病态条件的影响，但会受到维度的限制。 |
| [^42] | [Self-Correcting Self-Consuming Loops for Generative Model Training](https://arxiv.org/abs/2402.07087) | 本论文研究了使用合成数据进行生成模型训练时可能出现的自我消耗循环问题，并提出了一种通过引入理想的修正函数来稳定训练的方法。同时，我们还提出了自我修正函数来近似理想的修正函数，并通过实验证实了其有效性。 |
| [^43] | [Refined Sample Complexity for Markov Games with Independent Linear Function Approximation](https://arxiv.org/abs/2402.07082) | 本文在独立线性函数逼近的马尔科夫博弈中，通过改进AVLPR框架，提出了基于数据依赖的悲观估计方法，解决了多智能体的诅咒问题。 |
| [^44] | [Fast UCB-type algorithms for stochastic bandits with heavy and super heavy symmetric noise](https://arxiv.org/abs/2402.07062) | 本研究提出了一种基于凸优化方法和不精确预测模型的新UCB类型算法，用于解决具有重和超重对称噪声的随机赌博机问题。通过理论和实验结果表明，在奖励中存在对称噪声的情况下，该算法能够达到更好的性能，相比于一般下界能够获得更小的遗憾界。即使奖励分布没有期望，该算法仍然有效。 |
| [^45] | [Understanding the Training Speedup from Sampling with Approximate Losses](https://arxiv.org/abs/2402.07052) | 本文研究利用近似损失进行样本采样的训练加速方法，通过贪婪策略选择具有大约损失的样本，减少选择的开销，并证明其收敛速度优于随机选择。同时开发了使用中间层表示获取近似损失的SIFT方法，并在训练BERT模型上取得了显著的提升。 |
| [^46] | [Logistic-beta processes for modeling dependent random probabilities with beta marginals](https://arxiv.org/abs/2402.07048) | 本文提出了一种新颖的logistic-beta过程用于建模具有beta边际分布的相关随机概率。该过程具有灵活的相关结构和计算优势，并通过非参数二分类回归模拟研究进行了验证。 |
| [^47] | [Generalization Error of Graph Neural Networks in the Mean-field Regime](https://arxiv.org/abs/2402.07025) | 该论文在均场极限下提供了一个理论框架，评估了图神经网络在过参数化情况下的泛化误差，通过推导出收敛速度为$O(1/n)$的上界，为我们对网络在未见数据上的性能提供了理论保证。 |
| [^48] | [Tree Ensembles for Contextual Bandits](https://arxiv.org/abs/2402.06963) | 本论文提出了一种基于树集成的情境多臂老虎机新框架，通过整合两种广泛使用的老虎机方法，在标准和组合设置中实现了优于基于神经网络的方法的性能，在减少后悔和计算时间方面表现出更出色的性能。 |
| [^49] | [Efficient Incremental Belief Updates Using Weighted Virtual Observations](https://arxiv.org/abs/2402.06940) | 本文介绍了在贝叶斯统计模型中使用加权虚拟观测进行增量信念更新的算法解决方案，该方案通过构建一组加权观测来调节模型，实现与原始后验相同的推断结果。 |
| [^50] | [CochCeps-Augment: A Novel Self-Supervised Contrastive Learning Using Cochlear Cepstrum-based Masking for Speech Emotion Recognition](https://arxiv.org/abs/2402.06923) | 提出了一种名为CochCeps-Augment的方法，利用基于Cochlear Cepstrum的掩蔽增强任务进行自监督对比学习，提高了语音情感识别的性能和噪声鲁棒性。 |
| [^51] | [Principled Penalty-based Methods for Bilevel Reinforcement Learning and RLHF](https://arxiv.org/abs/2402.06886) | 本文提出了一种基于惩罚的方法来解决Bilevel强化学习和RLHF问题，这是首个有原则的算法框架。通过理论分析和实验证明了算法的有效性。 |
| [^52] | [Low-Rank Approximation of Structural Redundancy for Self-Supervised Learning](https://arxiv.org/abs/2402.06884) | 本文研究结构冗余的低秩逼近在自监督学习中的应用，提出了一个逼近冗余组件的新方法，并通过分析过量风险来支持理论。 |
| [^53] | [Scalable Kernel Logistic Regression with Nystr\"om Approximation: Theoretical Analysis and Application to Discrete Choice Modelling](https://arxiv.org/abs/2402.06763) | 本文介绍了使用Nystr\"om近似方法解决大规模数据集上核逻辑回归的可扩展性问题。研究提供了理论分析并验证了不同的地标选择方法的性能。 |
| [^54] | [Convergence of Gradient Descent with Small Initialization for Unregularized Matrix Completion](https://arxiv.org/abs/2402.06756) | 本文分析了对称矩阵完成问题中梯度下降算法的收敛性。研究结果表明，在非正则化的情况下，使用小初始化的梯度下降算法可以收敛到真实的矩阵解，即使在过度参数化的情况下也成立。在过度参数化的情况下，几乎线性的收敛速度可以在获得足够多的观测条目后得到保证。 |
| [^55] | [Anatomically-Controllable Medical Image Generation with Segmentation-Guided Diffusion Models](https://arxiv.org/abs/2402.05210) | 这篇论文提出了一种采用分割引导扩散模型的解剖可控医学图像生成方法，通过随机掩模消融训练算法实现对解剖约束的条件化，同时提高了网络对解剖真实性的学习能力。 |
| [^56] | [Denoising Diffusion Probabilistic Models in Six Simple Steps](https://arxiv.org/abs/2402.04384) | 本论文提供了一个简单、全面、干净且清晰的介绍去噪扩散概率模型（DDPM）的方法，强调了从连续时间极限的视角出发，以提供更好的理解和实际性能。 |
| [^57] | [Fast sampling from constrained spaces using the Metropolis-adjusted Mirror Langevin algorithm](https://arxiv.org/abs/2312.08823) | 该论文提出了一种名为Metropolis-adjusted Mirror Langevin算法的方法，用于从约束空间中进行快速采样。这种算法是对Mirror Langevin算法的改进，通过添加接受-拒绝过滤器来消除渐近偏差，并具有指数优化依赖。 |
| [^58] | [Is Inverse Reinforcement Learning Harder than Standard Reinforcement Learning? A Theoretical Perspective](https://arxiv.org/abs/2312.00054) | 逆向强化学习是从专家策略示范中学习奖励函数的问题，本文提出了在标准离线和在线设置下用多项式样本和运行时间进行高效逆向强化学习的结果线索，并提供了几乎最优的样本复杂性的下界。 |
| [^59] | [Efficient Reinforcement Learning from Partial Observability](https://arxiv.org/abs/2311.12244) | 该论文提出了一种基于表示的方法，用于从部分观测中进行有效的强化学习。该方法能够处理部分可观测性带来的计算和统计挑战，并在各种基准测试中展现出优于先进算法的性能。 |
| [^60] | [Kernel-, mean- and noise-marginalised Gaussian processes for exoplanet transits and $H_0$ inference](https://arxiv.org/abs/2311.04153) | 该论文提出了一种基于贝叶斯方法的核、均值和噪声边缘化高斯过程，用于系外行星凌星和H0推断。通过核选择和核超参数的边缘化以及贝叶斯模型比较，可以实现核选择和推断。 |
| [^61] | [Rethinking the Expressive Power of GNNs via Graph Biconnectivity](https://arxiv.org/abs/2301.09505) | 本文从根本上不同的角度重新思考了图神经网络（GNN）的表达能力，通过引入一类新的表达度量方法，即图的双连通性，并强调了它们在理论和实践中的重要性。令人惊讶的是，在对以前的GNN架构进行彻底审查后，发现大多数架构都没有对这些度量具有表达能力。唯一的例外是ESAN框架。 |
| [^62] | [When is Momentum Extragradient Optimal? A Polynomial-Based Analysis](https://arxiv.org/abs/2211.04659) | 本论文通过多项式分析，对动量外移梯度方法在不同情景下的加速收敛进行研究，包括特征值存在于实轴、位于实轴上的共轭复数或仅存在共轭复数的情况。同时，我们还得出了实现最快收敛的超参数。 |
| [^63] | [Dynamic Latent Separation for Deep Learning](https://arxiv.org/abs/2210.03728) | 本研究提出了动态潜变量分离的方法，可以在复杂数据中学习表达性强的潜变量，提升输出的多样性。该方法受原子物理学启发，通过学习每个数据样本的结构来解释各个子组件的重要性。实验证明该方法在不同分类和生成问题中提升了模型的性能。 |
| [^64] | [Differentially Private Graph Learning via Sensitivity-Bounded Personalized PageRank](https://arxiv.org/abs/2207.06944) | 本论文提出了一种敏感性有界的个性化PageRank算法，能够保护用户隐私。该算法在保持准确性的同时，实现了差分隐私图学习的几种工具。 |
| [^65] | [Computationally Efficient High-Dimensional Bayesian Optimization via Variable Selection](https://arxiv.org/abs/2109.09264) | 本论文提出了一种变量选择的计算高效高维贝叶斯优化方法，能够自动学习子空间来优化高维域函数，同时减少了传统方法中的耗时问题，并在实验证明了方法的有效性。 |
| [^66] | [Machine Collaboration](https://arxiv.org/abs/2105.02569) | 本文提出了一种新的监督学习集成框架——机器协作（MaC），通过循环和交互的学习方式，使基础机器能够循环传递信息并相应地更新结构和参数。实验证明，MaC在大多数情况下表现优于其他先进方法。 |
| [^67] | [Sparse NMF with Archetypal Regularization: Computational and Robustness Properties](https://arxiv.org/abs/2104.03527) | 本文研究了使用典型正则化的稀疏非负矩阵分解问题，提出了强鲁棒性和弱鲁棒性的概念，并给出了理论保证和数值实验来加强这些概念的洞察力。 |
| [^68] | [Scalable network reconstruction in subquadratic time.](http://arxiv.org/abs/2401.01404) | 这篇论文提出了一个可扩展的网络重建算法，能够在次二次时间内实现结果，通过随机的二阶邻居搜索产生最佳的边候选。 |
| [^69] | [Robust Angular Synchronization via Directed Graph Neural Networks.](http://arxiv.org/abs/2310.05842) | 本论文提出了一个名为GNNSync的基于有向图神经网络的鲁棒角度同步解决方案，解决了角度同步问题在高噪声环境下的挑战，并提出了新的损失函数以更好地编码同步约束。 |
| [^70] | [Bayesian deep learning for cosmic volumes with modified gravity.](http://arxiv.org/abs/2309.00612) | 该研究利用贝叶斯深度学习的方法，从修正引力模拟中提取宇宙学参数，并对不确定性进行了评估。 |
| [^71] | [Adaptive Proximal Gradient Method for Convex Optimization.](http://arxiv.org/abs/2308.02261) | 本文提出了自适应版本的梯度下降（GD）和近端梯度方法（ProxGD），通过利用局部曲率信息完全自适应。所提出的方法具有收敛性，且允许使用更大的步长。 |
| [^72] | [Understanding quantum machine learning also requires rethinking generalization.](http://arxiv.org/abs/2306.13461) | 本文通过实验认为，传统方法无法解释量子机器学习模型在只使用少量数据训练的情况下表现出成功的泛化性能，该模型可以准确拟合随机状态及随机标记的训练数据，这种记忆随机数据的能力违反了当前小泛化误差的概念，我们通过理论构建补充实证结果，表明量子神经网络可将任意标记拟合到量子状态上，暗示了它们的记忆能力，这些结果排除了单单基于经典复杂度度量的所有可能保证。 |
| [^73] | [Implicit Compressibility of Overparametrized Neural Networks Trained with Heavy-Tailed SGD.](http://arxiv.org/abs/2306.08125) | 本研究提出了一种简单的SGD修改方法，使训练出的神经网络输出可被证明为可压缩，而不需要任何非平凡假设。 |
| [^74] | [MESSY Estimation: Maximum-Entropy based Stochastic and Symbolic densitY Estimation.](http://arxiv.org/abs/2306.04120) | MESSY估计方法是一种基于最大熵的随机和符号密度估计方法，通过构建基于梯度的漂移扩散过程来高效地找到最大熵分布的参数，支持高维问题，并具有优于现有最新方法的有效性和普适性。 |
| [^75] | [Initial Guessing Bias: How Untrained Networks Favor Some Classes.](http://arxiv.org/abs/2306.00809) | 本文提出了“初始猜测偏差”现象，即在未经过训练的神经网络中，由于架构选择的影响，模型往往会将所有预测指向同一个类别。该现象对架构选择和初始化有实际指导意义，并具有理论后果，例如节点置换对称性的崩溃和深度带来的非平凡差异。 |
| [^76] | [Constructing Semantics-Aware Adversarial Examples with Probabilistic Perspective.](http://arxiv.org/abs/2306.00353) | 本研究提出了一个基于概率视角的对抗样本构建方法，可以生成语义感知的对抗性样本，并可以有效规避传统对抗性攻击的强化对抗训练方法。 |
| [^77] | [Shuffle SGD is Always Better than SGD: Improved Analysis of SGD with Arbitrary Data Orders.](http://arxiv.org/abs/2305.19259) | 本论文研究了一种允许任意数据排序的普通SGD算法,并表明在非凸函数情况下，随机和单次洗牌的SGD比经典替换的SGD更快或至少与其一样好，无论迭代次数如何。 |
| [^78] | [Error Bounds for Flow Matching Methods.](http://arxiv.org/abs/2305.16860) | 本文提出了基于ODE的流匹配方法的误差界限，适用于完全确定性抽样，需要满足$L^2$近似误差范围的规律性条件和数据分布。 |

# 详细

[^1]: 线性二次控制中策略梯度的隐性偏差：对未见初始状态的外推

    Implicit Bias of Policy Gradient in Linear Quadratic Control: Extrapolation to Unseen Initial States

    [https://arxiv.org/abs/2402.07875](https://arxiv.org/abs/2402.07875)

    本文研究了策略梯度在线性二次调节控制中对未见初始状态的外推问题，发现外推程度取决于训练中系统的探索程度。

    

    在现代机器学习中，模型可以以多种方式拟合训练数据，其中一些在未见（测试）数据上表现良好，而其他一些则不然。有趣的是，在这种情况下，梯度下降经常展现出一种隐性偏差，导致在未见数据上表现出色。这种隐性偏差在监督学习中已经得到了广泛研究，但在最优控制（强化学习）中却了解得较少。在那里，通过梯度下降学习应用于系统的控制器被称为策略梯度，并且一个非常重要的问题是学习的控制器在对未见初始状态的外推程度。本文在理论上研究了策略梯度在对未见初始状态的外推方面的隐性偏差。我们以基本的线性二次调节器（LQR）问题为重点，确立了外推程度取决于训练中系统在初始状态下引起的探索程度。

    In modern machine learning, models can often fit training data in numerous ways, some of which perform well on unseen (test) data, while others do not. Remarkably, in such cases gradient descent frequently exhibits an implicit bias that leads to excellent performance on unseen data. This implicit bias was extensively studied in supervised learning, but is far less understood in optimal control (reinforcement learning). There, learning a controller applied to a system via gradient descent is known as policy gradient, and a question of prime importance is the extent to which a learned controller extrapolates to unseen initial states. This paper theoretically studies the implicit bias of policy gradient in terms of extrapolation to unseen initial states. Focusing on the fundamental Linear Quadratic Regulator (LQR) problem, we establish that the extent of extrapolation depends on the degree of exploration induced by the system when commencing from initial states included in training. Exper
    
[^2]: 利用在赋值流形上的E-测地线流匹配生成离散联合分布的模型

    Generative Modeling of Discrete Joint Distributions by E-Geodesic Flow Matching on Assignment Manifolds

    [https://arxiv.org/abs/2402.07846](https://arxiv.org/abs/2402.07846)

    本文提出了一种基于连续归一化流的生成模型，该模型可以逐步分配类别，避免了离散化潜在连续模型时的舍入和样本截断等问题。通过匹配分解离散分布的测地线流，可以高效地训练该模型，并且适用于表示复杂统计依赖关系的非分解离散分布。

    

    本文介绍了一种基于连续归一化流在分解离散度量子流形上的生成模型，该模型逐步对类别进行分配，避免了离散化潜在连续模型时的舍入、样本截断等问题。将子流形嵌入到所有联合离散分布和数据驱动平均的元单纯形中，可以近似表示能够表示结构化离散数据的复杂统计依赖关系的一般非分解离散分布。通过匹配分解离散分布的测地线流，演示了该生成模型的高效训练。各种实验突出了该方法的广泛适用性。

    This paper introduces a novel generative model for discrete distributions based on continuous normalizing flows on the submanifold of factorizing discrete measures. Integration of the flow gradually assigns categories and avoids issues of discretizing the latent continuous model like rounding, sample truncation etc. General non-factorizing discrete distributions capable of representing complex statistical dependencies of structured discrete data, can be approximated by embedding the submanifold into a the meta-simplex of all joint discrete distributions and data-driven averaging. Efficient training of the generative model is demonstrated by matching the flow of geodesics of factorizing discrete distributions. Various experiments underline the approach's broad applicability.
    
[^3]: 论计算有效的多类别校准问题

    On Computationally Efficient Multi-Class Calibration

    [https://arxiv.org/abs/2402.07821](https://arxiv.org/abs/2402.07821)

    提出了一种在多类别预测问题中多样化的投影平滑校准概念，并且给出了多项式时间复杂度的重新校准算法，从而实现了计算效率和强大的预测保证之间的权衡。

    

    考虑一个多类别标记问题，其中标记可以在[1,k]范围内取值，而预测器预测的是标记的分布。在这项工作中，我们研究了以下基础问题：是否存在多类别校准的概念，可以给出对有意义的预测的强大保证，并且可以在多项式时间和样本复杂度下实现？先前的校准概念在计算效率和表达能力之间存在着权衡：它们要么在k的样本复杂度上呈指数级增长，要么需要求解计算难题，要么给出的保证相当弱。我们的主要贡献是提出了一种能够实现所有这些期望的校准概念：我们在多类别预测中制定了一个稳健的投影平滑校准概念，并给出了新的重新校准算法，以在这个定义下以多项式时间复杂度校准预测器。投影平滑校准为多类别预测提供了强大的保证。

    Consider a multi-class labelling problem, where the labels can take values in $[k]$, and a predictor predicts a distribution over the labels. In this work, we study the following foundational question: Are there notions of multi-class calibration that give strong guarantees of meaningful predictions and can be achieved in time and sample complexities polynomial in $k$? Prior notions of calibration exhibit a tradeoff between computational efficiency and expressivity: they either suffer from having sample complexity exponential in $k$, or needing to solve computationally intractable problems, or give rather weak guarantees.   Our main contribution is a notion of calibration that achieves all these desiderata: we formulate a robust notion of projected smooth calibration for multi-class predictions, and give new recalibration algorithms for efficiently calibrating predictors under this definition with complexity polynomial in $k$. Projected smooth calibration gives strong guarantees for al
    
[^4]: 面向扩散模型的一种一致性训练数学理论的探索

    Towards a mathematical theory for consistency training in diffusion models

    [https://arxiv.org/abs/2402.07802](https://arxiv.org/abs/2402.07802)

    本文探索了面向扩散模型中一致性训练的理论基础，证明了在一致性学习中，步骤数量需要超过$d^{5/2}/\varepsilon$的阶数，能够生成与目标分布接近的样本。

    

    一致性模型被提出来减少扩散模型采样阶段的高计算开销，实现了单步采样并达到了最先进的实证性能。一致性模型在训练阶段被整合进来，试图训练一系列的一致性函数，能够将扩散过程中的任何时间步骤的任何点映射回其起始点。尽管在实证上取得了成功，但关于一致性训练的全面理论理解还是很难得到的。本文对一致性模型的理论基础进行了初步的探索。我们证明，为了在分布中生成与目标在$\varepsilon$接近的样本（通过某种Wasserstein度量衡量），一致性学习中的步骤数量需要超过$d^{5/2}/\varepsilon$的阶数，其中$d$是数据维度。我们的理论为一致性模型的有效性和有效性提供了严格的洞察。

    Consistency models, which were proposed to mitigate the high computational overhead during the sampling phase of diffusion models, facilitate single-step sampling while attaining state-of-the-art empirical performance. When integrated into the training phase, consistency models attempt to train a sequence of consistency functions capable of mapping any point at any time step of the diffusion process to its starting point. Despite the empirical success, a comprehensive theoretical understanding of consistency training remains elusive. This paper takes a first step towards establishing theoretical underpinnings for consistency models. We demonstrate that, in order to generate samples within $\varepsilon$ proximity to the target in distribution (measured by some Wasserstein metric), it suffices for the number of steps in consistency learning to exceed the order of $d^{5/2}/\varepsilon$, with $d$ the data dimension. Our theory offers rigorous insights into the validity and efficacy of cons
    
[^5]: 无调参的随机优化

    Tuning-Free Stochastic Optimization

    [https://arxiv.org/abs/2402.07793](https://arxiv.org/abs/2402.07793)

    本文提出了一种无调参的随机优化算法，能够在只给出问题参数的粗略提示的情况下，与最优调参优化算法的性能相匹配。并且在有界的优化领域中证明了此算法的可行性，并探讨了在无界域中的条件。

    

    大规模机器学习问题使得调参的成本越来越高昂。这导致了需要能够即时自我调整的算法的需求。我们将“无调参”算法的概念形式化，即只给出问题参数的粗略提示即可与最优调参优化算法的性能相匹配，误差为对数多项式因子。我们特别考虑能够与最优调参的随机梯度下降(SGD)相匹配的算法。当优化的域是有界的时候，我们证明了调参自由与SGD的匹配是可能的，并且通过几个现有算法实现了这一点。我们证明了当优化的域是无界的时候，对于最小化凸平滑或者Lipschitz函数的任务，无调参优化是不可能的。我们讨论了在无界域中，何种情况下可以实现无调参优化。特别地，我们展示了最近提出的 DoG 和 DoWG 算法在噪声分布足够时是无调参的。

    Large-scale machine learning problems make the cost of hyperparameter tuning ever more prohibitive. This creates a need for algorithms that can tune themselves on-the-fly. We formalize the notion of "tuning-free" algorithms that can match the performance of optimally-tuned optimization algorithms up to polylogarithmic factors given only loose hints on the relevant problem parameters. We consider in particular algorithms that can match optimally-tuned Stochastic Gradient Descent (SGD). When the domain of optimization is bounded, we show tuning-free matching of SGD is possible and achieved by several existing algorithms. We prove that for the task of minimizing a convex and smooth or Lipschitz function over an unbounded domain, tuning-free optimization is impossible. We discuss conditions under which tuning-free optimization is possible even over unbounded domains. In particular, we show that the recently proposed DoG and DoWG algorithms are tuning-free when the noise distribution is suf
    
[^6]: 可扩展的稀疏特定背景下因果系统的结构学习

    Scalable Structure Learning for Sparse Context-Specific Causal Systems

    [https://arxiv.org/abs/2402.07762](https://arxiv.org/abs/2402.07762)

    提出了一种可扩展的混合算法，用于学习特定背景模型，通过结合基于顺序的MCMC算法和稀疏性假设实现可扩展学习，该方法在准确性和可扩展性方面表现良好。

    

    已经提出了几种表示共同分布分类变量之间特定背景下关系的方法，并且提出了结构学习算法。然而，由于大量特定背景模型的存在，现有的基于优化的方法在可扩展性方面受到限制，而基于约束的方法比约束DAG学习算法更容易出错，因为必须测试更多关系。我们提出了一种混合算法来学习特定背景模型，能够扩展到数百个变量，并且测试的约束不多于标准DAG学习算法。通过结合基于顺序的MCMC算法和类似于DAG模型常用的稀疏性假设，实现了可扩展的学习。为了实现这种方法，我们解决了Alon和Balogh最近提出的一个开放问题的特殊情况。经过在合成数据和真实世界示例上的实验证明，该方法在准确性和可扩展性方面表现良好。

    Several approaches to graphically representing context-specific relations among jointly distributed categorical variables have been proposed, along with structure learning algorithms. While existing optimization-based methods have limited scalability due to the large number of context-specific models, the constraint-based methods are more prone to error than even constraint-based DAG learning algorithms since more relations must be tested. We present a hybrid algorithm for learning context-specific models that scales to hundreds of variables while testing no more constraints than standard DAG learning algorithms. Scalable learning is achieved through a combination of an order-based MCMC algorithm and sparsity assumptions analogous to those typically invoked for DAG models. To implement the method, we solve a special case of an open problem recently posed by Alon and Balogh. The method is shown to perform well on synthetic data and real world examples, in terms of both accuracy and scal
    
[^7]: 通过经验贝叶斯平滑进行最优分数估计

    Optimal score estimation via empirical Bayes smoothing

    [https://arxiv.org/abs/2402.07747](https://arxiv.org/abs/2402.07747)

    该论文研究了通过经验贝叶斯平滑在高维数据中估计未知概率分布的分数函数的问题，提出了一种基于高斯核的正则化分数估计器，在score matching损失函数下达到了最优速率，并揭示了维度增长对样本复杂性的指数级影响。

    

    我们研究了从$d$维独立同分布观测中估计未知概率分布$\rho^*$的分数函数的问题。在假设$\rho^*$是亚高斯的并且具有Lipschitz连续的分数函数$s^*$的情况下，我们在score matching文献中常用的损失函数$\|\hat s - s^*\|^2_{L^2(\rho^*)}$下建立了该估计问题的最优速率为$\tilde \Theta(n^{-\frac{2}{d+4}})$，强调了维度$d$的增长对于准确分数估计的样本复杂性呈指数级增长的困境。借助经验贝叶斯理论的关键见解以及平滑经验分布在Hellinger距离下的新收敛速率，我们展示了基于高斯核的正则化分数估计器能够达到该速率，并通过匹配最小值下界证明了其最优性。我们还讨论了我们理论对于基于分数的生成模型的样本复杂性的影响。

    We study the problem of estimating the score function of an unknown probability distribution $\rho^*$ from $n$ independent and identically distributed observations in $d$ dimensions. Assuming that $\rho^*$ is subgaussian and has a Lipschitz-continuous score function $s^*$, we establish the optimal rate of $\tilde \Theta(n^{-\frac{2}{d+4}})$ for this estimation problem under the loss function $\|\hat s - s^*\|^2_{L^2(\rho^*)}$ that is commonly used in the score matching literature, highlighting the curse of dimensionality where sample complexity for accurate score estimation grows exponentially with the dimension $d$. Leveraging key insights in empirical Bayes theory as well as a new convergence rate of smoothed empirical distribution in Hellinger distance, we show that a regularized score estimator based on a Gaussian kernel attains this rate, shown optimal by a matching minimax lower bound. We also discuss the implication of our theory on the sample complexity of score-based generativ
    
[^8]: 用BAM进行图结构推断：引入双线性注意机制

    Graph Structure Inference with BAM: Introducing the Bilinear Attention Mechanism

    [https://arxiv.org/abs/2402.07735](https://arxiv.org/abs/2402.07735)

    本论文提出了一种利用BAM进行图结构推断的方法。通过神经网络模型，通过变形的耦合模拟输入数据进行训练，仅需通过一次前向传递即可进行推断。通过利用结构方程模型和随机生成的多变量切比雪夫多项式来模拟训练数据，方法能够泛化到线性和各种非线性依赖关系。引入了双线性注意机制（BAM）来处理依赖关系，该机制在转换数据的协方差矩阵水平上运行，并尊重对称正定矩阵流形的几何特性。实证评估证明了方法的有效性和性能。

    

    在统计学和机器学习中，检测数据集中的依赖关系是一个核心挑战。我们提出了一种新颖的神经网络模型，用于监督图结构学习，即学习观测数据和它们的基本依赖结构之间的映射。该模型通过变形的耦合模拟输入数据进行训练，并且仅需通过训练网络进行一次前向传递即可进行推断。通过利用结构方程模型，并通过随机生成的多变量切比雪夫多项式来模拟训练数据，我们的方法展示了在线性和各种非线性依赖关系之间的强大泛化能力。我们引入了一种新的双线性注意机制（BAM），用于显式处理依赖信息，该机制在转换数据的协方差矩阵水平上运行，并尊重对称正定矩阵流形的几何特性。实证评估展示了方法的有效性和性能。

    In statistics and machine learning, detecting dependencies in datasets is a central challenge. We propose a novel neural network model for supervised graph structure learning, i.e., the process of learning a mapping between observational data and their underlying dependence structure. The model is trained with variably shaped and coupled simulated input data and requires only a single forward pass through the trained network for inference. By leveraging structural equation models and employing randomly generated multivariate Chebyshev polynomials for the simulation of training data, our method demonstrates robust generalizability across both linear and various types of non-linear dependencies. We introduce a novel bilinear attention mechanism (BAM) for explicit processing of dependency information, which operates on the level of covariance matrices of transformed data and respects the geometry of the manifold of symmetric positive definite matrices. Empirical evaluation demonstrates th
    
[^9]: 通过分数阻尼库仑方程证明重尾SDEs的泛化界限

    Generalization Bounds for Heavy-Tailed SDEs through the Fractional Fokker-Planck Equation

    [https://arxiv.org/abs/2402.07723](https://arxiv.org/abs/2402.07723)

    本论文通过分数阻尼库仑方程证明了重尾SDE的高概率泛化界限，并且相对于参数维度，界限的依赖性要好于p。

    

    过去几年来，理解重尾随机优化算法的泛化性能引起了越来越多的关注。在利用重尾随机微分方程作为代理来阐明随机优化器的有趣方面时，先前的工作要么提供预期的泛化界限，要么引入了不可计算的信息论术语。为了解决这些缺点，在本文中，我们证明了重尾SDE的高概率泛化界限，这些界限不含任何非平凡的信息论术语。为了实现这个目标，我们基于估计与所谓的分数阻尼库仑方程相关联的熵流，开发了新的证明技术（这是一种控制相应重尾SDE分布演化的偏微分方程）。除了获得高概率界限之外，我们还展示了我们的界限相对于参数维度的依赖性要好于p。

    Understanding the generalization properties of heavy-tailed stochastic optimization algorithms has attracted increasing attention over the past years. While illuminating interesting aspects of stochastic optimizers by using heavy-tailed stochastic differential equations as proxies, prior works either provided expected generalization bounds, or introduced non-computable information theoretic terms. Addressing these drawbacks, in this work, we prove high-probability generalization bounds for heavy-tailed SDEs which do not contain any nontrivial information theoretic terms. To achieve this goal, we develop new proof techniques based on estimating the entropy flows associated with the so-called fractional Fokker-Planck equation (a partial differential equation that governs the evolution of the distribution of the corresponding heavy-tailed SDE). In addition to obtaining high-probability bounds, we show that our bounds have a better dependence on the dimension of parameters as compared to p
    
[^10]: 一些统计模型之间的高效归约

    Efficient reductions between some statistical models

    [https://arxiv.org/abs/2402.07717](https://arxiv.org/abs/2402.07717)

    本研究提出了一种在不知道源统计模型参数的情况下，高效地将样本从源模型转换为目标模型的方法，并构造了几个归约方法。这些归约方法能适应不同的问题，例如专家混合模型、相位恢复和信号降噪等，并且可以处理缺失数据。此外，该研究还指出了一个潜在的应用，即将一个差分隐私机制转换为另一个机制。

    

    我们研究了在不知道源模型参数的情况下，近似地将来自源统计模型的样本转换为目标统计模型的样本的问题，并构造了几个计算上高效的这种统计实验之间的归约。具体而言，我们提供了计算上高效的程序，可以近似将均匀分布、Erlang分布和拉普拉斯分布的位置模型归约到一般的目标族。我们通过建立一些经典的高维问题之间的非渐近归约来说明我们的方法，包括专家混合模型、相位恢复和信号降噪等。值得注意的是，这些归约保持了结构，并可以适应缺失数据。我们还指出了将一个差分隐私机制转换为另一个机制的可能应用。

    We study the problem of approximately transforming a sample from a source statistical model to a sample from a target statistical model without knowing the parameters of the source model, and construct several computationally efficient such reductions between statistical experiments. In particular, we provide computationally efficient procedures that approximately reduce uniform, Erlang, and Laplace location models to general target families. We illustrate our methodology by establishing nonasymptotic reductions between some canonical high-dimensional problems, spanning mixtures of experts, phase retrieval, and signal denoising. Notably, the reductions are structure preserving and can accommodate missing data. We also point to a possible application in transforming one differentially private mechanism to another.
    
[^11]: 模型崩溃解密：回归案例研究

    Model Collapse Demystified: The Case of Regression

    [https://arxiv.org/abs/2402.07712](https://arxiv.org/abs/2402.07712)

    本研究在核回归的简化环境中解析了模型崩溃现象，并发现了模型能够处理虚假数据与性能完全崩溃之间的交叉点。通过提出基于自适应正则化的策略，成功缓解了模型崩溃问题。这些发现通过实验证实。

    

    在像ChatGPT这样的大型语言模型的时代，"模型崩溃"现象指的是模型在递归地训练自身上一代又一代生成的数据时，其性能逐渐降低，最终变得完全无用，即模型崩溃。在这项工作中，我们在核回归的简化环境中研究了这一现象，并获得了结果，显示模型能够处理虚假数据与模型性能完全崩溃之间存在明显的交叉点。在多项式衰减的光谱和源条件下，我们获得了修改后的缩放定律，展示了从快速到缓慢速率的新交叉现象。我们还提出了基于自适应正则化的简单策略来缓解模型崩溃。我们的理论结果通过实验证实。

    In the era of large language models like ChatGPT, the phenomenon of "model collapse" refers to the situation whereby as a model is trained recursively on data generated from previous generations of itself over time, its performance degrades until the model eventually becomes completely useless, i.e the model collapses. In this work, we study this phenomenon in the simplified setting of kernel regression and obtain results which show a clear crossover between where the model can cope with fake data, and a regime where the model's performance completely collapses. Under polynomial decaying spectral and source conditions, we obtain modified scaling laws which exhibit new crossover phenomena from fast to slow rates. We also propose a simple strategy based on adaptive regularization to mitigate model collapse. Our theoretical results are validated with experiments.
    
[^12]: 随机梯度流动力学中的测试风险及其弱特征的精确解

    Stochastic Gradient Flow Dynamics of Test Risk and its Exact Solution for Weak Features

    [https://arxiv.org/abs/2402.07626](https://arxiv.org/abs/2402.07626)

    本研究通过路径积分方法探索了连续时间随机梯度流动力学中的测试风险，并在小学习率情况下给出了计算纯梯度流动和随机梯度流动的测试风险曲线之间差异的一般公式。通过应用于一个弱特征模型，我们分析了随机项对动力学的修正效果，并与离散时间随机梯度下降的模拟结果进行了比较，结果显示出一致性。

    

    本研究探讨了学习理论中连续时间随机梯度流动力学的测试风险。利用路径积分公式，在小学习率的情况下，提供了计算纯梯度流动和随机梯度流动的测试风险曲线之间差异的一般公式。我们将这一通用理论应用到一个简单的弱特征模型中，该模型展示了双峰现象，并明确计算了动力学中增加的随机项随时间和模型参数的修正。分析结果与离散时间随机梯度下降的模拟进行了比较，显示出良好的一致性。

    We investigate the test risk of continuous-time stochastic gradient flow dynamics in learning theory. Using a path integral formulation we provide, in the regime of a small learning rate, a general formula for computing the difference between test risk curves of pure gradient and stochastic gradient flows. We apply the general theory to a simple model of weak features, which displays the double descent phenomenon, and explicitly compute the corrections brought about by the added stochastic term in the dynamics, as a function of time and model parameters. The analytical results are compared to simulations of discrete-time stochastic gradient descent and show good agreement.
    
[^13]: 在可接受的对称约束条件下的全局最优性

    Global optimality under amenable symmetry constraints

    [https://arxiv.org/abs/2402.07613](https://arxiv.org/abs/2402.07613)

    该论文研究了在可接受的对称约束条件下的全局最优性问题，提出了一种满足对称性质的函数或度量，并通过引入轨道凸体和coycle等工具解决了这一问题。具体应用包括不变核均值嵌入和基于对称约束的运输方案最优性。这些结果与不变性检验的Hunt-Stein定理相关。

    

    我们研究是否存在一种满足可接受变换群指定的对称性质的函数或度量，即同时满足以下两个条件：（1）最小化给定的凸性泛函或风险，（2）满足可容忍对称约束。这种对称性质的例子包括不变性、可变性或准不变性。我们的结果依赖于Stein和Le Cam的老思想，以及在可接受群的遍历定理中出现的近似群平均值。在凸分析中，一类称为轨道凸体的凸集显得至关重要，我们在非参数设置中确定了这类轨道凸体的性质。我们还展示了一个称为coycle的简单装置如何将不同形式的对称性转化为一个问题。作为应用，我们得出了关于不变核均值嵌入和在对称约束下运输方案最优性的Monge-Kantorovich定理的结果。我们还解释了与不变性检验的Hunt-Stein定理的联系。

    We ask whether there exists a function or measure that (1) minimizes a given convex functional or risk and (2) satisfies a symmetry property specified by an amenable group of transformations. Examples of such symmetry properties are invariance, equivariance, or quasi-invariance. Our results draw on old ideas of Stein and Le Cam and on approximate group averages that appear in ergodic theorems for amenable groups. A class of convex sets known as orbitopes in convex analysis emerges as crucial, and we establish properties of such orbitopes in nonparametric settings. We also show how a simple device called a cocycle can be used to reduce different forms of symmetry to a single problem. As applications, we obtain results on invariant kernel mean embeddings and a Monge-Kantorovich theorem on optimality of transport plans under symmetry constraints. We also explain connections to the Hunt-Stein theorem on invariant tests.
    
[^14]: 基于生成模型的近最小极大分布式强化学习算法

    Near-Minimax-Optimal Distributional Reinforcement Learning with a Generative Model

    [https://arxiv.org/abs/2402.07598](https://arxiv.org/abs/2402.07598)

    本论文提出了一种基于生成模型的近最小极大分布式强化学习算法，该算法在使用生成模型近似回报分布方面具有极小极大优势，解决了一个开放问题，并提供了实验研究结果。

    

    我们提出了一种新的基于模型的分布式强化学习算法，并证明了在使用生成模型近似回报分布方面，它是近似最小极大的（在对数因子上），从而解决了Zhang等人（2023）的一个开放问题。我们的分析为分布式强化学习中的分类方法提供了新的理论结果，并引入了一种新的分布式Bellman方程，即随机分类累积分布函数Bellman方程，我们认为这个方程也具有独立的研究意义。我们还进行了实验研究，比较了几种基于模型的分布式强化学习算法，并得出了对实践者有意义的几个结论。

    We propose a new algorithm for model-based distributional reinforcement learning (RL), and prove that it is minimax-optimal for approximating return distributions with a generative model (up to logarithmic factors), resolving an open question of Zhang et al. (2023). Our analysis provides new theoretical results on categorical approaches to distributional RL, and also introduces a new distributional Bellman equation, the stochastic categorical CDF Bellman equation, which we expect to be of independent interest. We also provide an experimental study comparing several model-based distributional RL algorithms, with several takeaways for practitioners.
    
[^15]: 重新思考战略环境中学习的比例定律

    Rethinking Scaling Laws for Learning in Strategic Environments

    [https://arxiv.org/abs/2402.07588](https://arxiv.org/abs/2402.07588)

    本文重新思考了在战略环境中学习的比例定律，发现战略互动可以打破传统的观点，即模型越大或表达能力越强并不一定会随之提高性能。通过几个战略环境的例子，我们展示了这种现象的影响。

    

    越来越大的机器学习模型的部署反映出一个共识：模型越有表达能力，越拥有大量数据，就能改善性能。随着模型在各种真实场景中的部署，它们不可避免地面临着战略环境。本文考虑了模型与战略互动对比例定律的相互作用对性能的影响这个自然问题。我们发现战略互动可以打破传统的比例定律观点，即性能并不一定随着模型的扩大和/或表达能力的增强（即使有无限数据）而单调提高。我们通过战略回归、战略分类和多智能体强化学习的例子展示了这一现象的影响，这些例子展示了战略环境中的限制模型或策略类的表达能力即可。

    The deployment of ever-larger machine learning models reflects a growing consensus that the more expressive the model$\unicode{x2013}$and the more data one has access to$\unicode{x2013}$the more one can improve performance. As models get deployed in a variety of real world scenarios, they inevitably face strategic environments. In this work, we consider the natural question of how the interplay of models and strategic interactions affects scaling laws. We find that strategic interactions can break the conventional view of scaling laws$\unicode{x2013}$meaning that performance does not necessarily monotonically improve as models get larger and/ or more expressive (even with infinite data). We show the implications of this phenomenon in several contexts including strategic regression, strategic classification, and multi-agent reinforcement learning through examples of strategic environments in which$\unicode{x2013}$by simply restricting the expressivity of one's model or policy class$\uni
    
[^16]: Weisfeiler-Leman在边缘条件下的更高表达力的重要性

    Weisfeiler-Leman at the margin: When more expressivity matters

    [https://arxiv.org/abs/2402.07568](https://arxiv.org/abs/2402.07568)

    研究探讨了1-WL算法在图同构问题中的表达能力和泛化性能之间的关系，发现增强的表达能力对提高泛化性能并不总是有效。此外，通过引入子图信息和经典的边缘理论，探索了更高表达力与改进泛化性能的条件。梯度流也被证明可以促进模型学习更丰富的表达能力。

    

    Weisfeiler-Leman算法（1-WL）是一个被广泛研究的用于图同构问题的启发式算法。最近，该算法在理解传递消息的图神经网络（MPNNs）的表达能力以及作为图核函数方面发挥了重要作用。尽管取得了成功，但1-WL在区分非同构图方面面临挑战，从而导致了更具表达力的MPNN和核架构的发展。然而，增强的表达能力和改进的泛化性能之间的关系仍不清楚。在本文中，我们展示了当通过图同构来观察时，架构的表达能力在解释其泛化性能方面具有有限的洞察力。此外，我们着重在1-WL和MPNN中引入子图信息，并运用经典的边缘理论来研究架构的增强表达能力与改进的泛化性能之间的条件。此外，我们还展示了梯度流如何推动模型学习更丰富的表达能力。

    The Weisfeiler-Leman algorithm ($1$-WL) is a well-studied heuristic for the graph isomorphism problem. Recently, the algorithm has played a prominent role in understanding the expressive power of message-passing graph neural networks (MPNNs) and being effective as a graph kernel. Despite its success, $1$-WL faces challenges in distinguishing non-isomorphic graphs, leading to the development of more expressive MPNN and kernel architectures. However, the relationship between enhanced expressivity and improved generalization performance remains unclear. Here, we show that an architecture's expressivity offers limited insights into its generalization performance when viewed through graph isomorphism. Moreover, we focus on augmenting $1$-WL and MPNNs with subgraph information and employ classical margin theory to investigate the conditions under which an architecture's increased expressivity aligns with improved generalization performance. In addition, we show that gradient flow pushes the 
    
[^17]: 机器学习与小区域估计的整合步骤

    A step towards the integration of machine learning and small area estimation

    [https://arxiv.org/abs/2402.07521](https://arxiv.org/abs/2402.07521)

    本文提出了一个基于机器学习算法的预测模型，可以根据横断面和纵向数据预测任何人群或子人群的特征，并分析了在实际生活中更重要的背景下的性能。

    

    机器学习技术的应用已经在许多研究领域得到了发展。目前，在统计学中，包括正式统计学在内，也广泛应用于数据收集（如卫星图像、网络爬取和文本挖掘、数据清洗、集成和插补）以及数据分析。然而，在调查抽样包括小区域估计方面，这些方法的使用仍然非常有限。因此，我们提出一个由这些算法支持的预测模型，可以根据横断面和纵向数据预测任何人群或子人群的特征。机器学习方法已经显示出在识别和建模变量之间复杂和非线性关系方面非常强大，这意味着在强烈偏离经典假设的情况下，它们具有非常好的性能。因此，我们分析了我们的模型在一种不同的背景下的表现，这个背景在我们看来在实际生活中更重要。

    The use of machine-learning techniques has grown in numerous research areas. Currently, it is also widely used in statistics, including the official statistics for data collection (e.g. satellite imagery, web scraping and text mining, data cleaning, integration and imputation) but also for data analysis. However, the usage of these methods in survey sampling including small area estimation is still very limited. Therefore, we propose a predictor supported by these algorithms which can be used to predict any population or subpopulation characteristics based on cross-sectional and longitudinal data. Machine learning methods have already been shown to be very powerful in identifying and modelling complex and nonlinear relationships between the variables, which means that they have very good properties in case of strong departures from the classic assumptions. Therefore, we analyse the performance of our proposal under a different set-up, in our opinion of greater importance in real-life s
    
[^18]: 基于得分的物理信息神经网络用于高维福克-普朗克方程

    Score-Based Physics-Informed Neural Networks for High-Dimensional Fokker-Planck Equations

    [https://arxiv.org/abs/2402.07465](https://arxiv.org/abs/2402.07465)

    这项研究提出了一种基于得分函数的求解器来解决高维福克-普朗克方程中的维数灾难问题。与蒙特卡洛和普通PINN相比，该方法能够更准确地处理与布朗运动相关的概率密度函数，并提供快速采样。

    

    福克-普朗克（FP）方程是随机过程中的基础偏微分方程（PDE）。然而，当处理高维FP PDE时，维数灾难（CoD）会带来挑战。尽管蒙特卡洛和普通物理信息神经网络（PINN）已经显示出应对CoD的潜力，但在处理与布朗运动相关的概率密度函数（PDF）时，两种方法都在高维度上显示出数值误差。点值PDF随着维度增加呈指数级下降，超过了数值模拟的精度，导致了相当大的误差。此外，由于其大规模采样，蒙特卡洛无法提供快速采样。通过对普通PINNs模拟对数似然（LL），将FP方程转化为一个困难的HJB方程，其误差随维数增长迅速。为此，我们提出了一种新方法，利用基于得分的求解器来拟合SDE中的得分函数。得分函数定义为概率密度函数的梯度。

    The Fokker-Planck (FP) equation is a foundational PDE in stochastic processes. However, curse of dimensionality (CoD) poses challenge when dealing with high-dimensional FP PDEs. Although Monte Carlo and vanilla Physics-Informed Neural Networks (PINNs) have shown the potential to tackle CoD, both methods exhibit numerical errors in high dimensions when dealing with the probability density function (PDF) associated with Brownian motion. The point-wise PDF values tend to decrease exponentially as dimension increases, surpassing the precision of numerical simulations and resulting in substantial errors. Moreover, due to its massive sampling, Monte Carlo fails to offer fast sampling. Modeling the logarithm likelihood (LL) via vanilla PINNs transforms the FP equation into a difficult HJB equation, whose error grows rapidly with dimension. To this end, we propose a novel approach utilizing a score-based solver to fit the score function in SDEs. The score function, defined as the gradient of t
    
[^19]: 关于顺序预测中的标定距离研究

    On the Distance from Calibration in Sequential Prediction

    [https://arxiv.org/abs/2402.07458](https://arxiv.org/abs/2402.07458)

    本论文研究了顺序预测中的标定距离，证明了存在一种预测算法可以在敌人选择的二进制序列上实现$O(\sqrt{T})$的标定距离，通过较低的标定距离进行准确近似。

    

    我们研究了一种顺序二进制预测场景，在这种场景中，预测器的评估是以标定距离为基准的，标定距离定义为预测值与事后完全标定的预测集之间的$L_1$距离。这类似于最近由B{\l}asiok、Gopalan、Hu和Nakkiran（STOC 2023）提出的离线场景中的标定度量。标定距离是一种自然且直观的偏离完美标定的度量，并且满足不同于许多常见的标定度量（如$L_1$标定误差及其变种）的Lipschitz连续性属性。我们证明了存在一种预测算法，可以在对敌人选择的长度为$T$的二进制序列上，以期望$O(\sqrt{T})$的标定距离实现。在这个上界的核心是一个结构性结果，证明了标定距离可以通过较低的标定距离进行准确近似。

    We study a sequential binary prediction setting where the forecaster is evaluated in terms of the calibration distance, which is defined as the $L_1$ distance between the predicted values and the set of predictions that are perfectly calibrated in hindsight. This is analogous to a calibration measure recently proposed by B{\l}asiok, Gopalan, Hu and Nakkiran (STOC 2023) for the offline setting. The calibration distance is a natural and intuitive measure of deviation from perfect calibration, and satisfies a Lipschitz continuity property which does not hold for many popular calibration measures, such as the $L_1$ calibration error and its variants.   We prove that there is a forecasting algorithm that achieves an $O(\sqrt{T})$ calibration distance in expectation on an adversarially chosen sequence of $T$ binary outcomes. At the core of this upper bound is a structural result showing that the calibration distance is accurately approximated by the lower calibration distance, which is a con
    
[^20]: Bandit-Feedback在线多类分类：变体和权衡

    Bandit-Feedback Online Multiclass Classification: Variants and Tradeoffs

    [https://arxiv.org/abs/2402.07453](https://arxiv.org/abs/2402.07453)

    该论文研究了在对抗在线环境中多类分类中依赖于强盗反馈的代价，自适应对手和随机学习者与无视对手和确定性学习者之间的损失差距。

    

    在对抗在线环境中考虑多类分类领域。与提供完全信息相比，依赖于强盗反馈的代价是多少？自适应对手与无视对手相比，可以增加损失的程度有多大？随机学习者与确定性学习者相比，可以降低损失的程度有多大？我们在错误边界模型中研究了这些问题，并提供了几乎紧确的答案。

    Consider the domain of multiclass classification within the adversarial online setting. What is the price of relying on bandit feedback as opposed to full information? To what extent can an adaptive adversary amplify the loss compared to an oblivious one? To what extent can a randomized learner reduce the loss compared to a deterministic one? We study these questions in the mistake bound model and provide nearly tight answers.   We demonstrate that the optimal mistake bound under bandit feedback is at most $O(k)$ times higher than the optimal mistake bound in the full information case, where $k$ represents the number of labels. This bound is tight and provides an answer to an open question previously posed and studied by Daniely and Helbertal ['13] and by Long ['17, '20], who focused on deterministic learners.   Moreover, we present nearly optimal bounds of $\tilde{\Theta}(k)$ on the gap between randomized and deterministic learners, as well as between adaptive and oblivious adversarie
    
[^21]: 具有单调对手的Top-K排名问题

    Top-$K$ ranking with a monotone adversary

    [https://arxiv.org/abs/2402.07445](https://arxiv.org/abs/2402.07445)

    本文针对具有单调对手的Top-K排名问题，提出了一种加权最大似然估计器(MLE)，在样本复杂度方面接近最优。算法创新包括了对加权MLE的精确且紧密的$\ell_\infty$误差分析，并与加权比较图的谱特性相关联。

    

    本文解决了具有单调对手的Top-K排名问题。我们考虑了一个比较图被随机生成且对手可以添加任意边的情况。统计学家的目标是根据从这个半随机比较图导出的两两比较准确地识别出Top-K的首选项。本文的主要贡献是开发出一种加权最大似然估计器(MLE)，它在样本复杂度方面达到了近似最优，最多差一个$log^2(n)$的因子，其中n表示比较项的数量。这得益于分析和算法创新的结合。在分析方面，我们提供了一种更明确、更紧密的加权MLE的$\ell_\infty$误差分析，它与加权比较图的谱特性相关。受此启发，我们的算法创新涉及到了

    In this paper, we address the top-$K$ ranking problem with a monotone adversary. We consider the scenario where a comparison graph is randomly generated and the adversary is allowed to add arbitrary edges. The statistician's goal is then to accurately identify the top-$K$ preferred items based on pairwise comparisons derived from this semi-random comparison graph. The main contribution of this paper is to develop a weighted maximum likelihood estimator (MLE) that achieves near-optimal sample complexity, up to a $\log^2(n)$ factor, where n denotes the number of items under comparison. This is made possible through a combination of analytical and algorithmic innovations. On the analytical front, we provide a refined $\ell_\infty$ error analysis of the weighted MLE that is more explicit and tighter than existing analyses. It relates the $\ell_\infty$ error with the spectral properties of the weighted comparison graph. Motivated by this, our algorithmic innovation involves the development 
    
[^22]: 条件生成模型足以从任何因果效应测度中采样

    Conditional Generative Models are Sufficient to Sample from Any Causal Effect Estimand

    [https://arxiv.org/abs/2402.07419](https://arxiv.org/abs/2402.07419)

    本文展示了通过条件生成模型的推进计算可以计算任何可辨识的因果效应，并提出了基于扩散的方法用于从图像的任何（条件）干预分布中进行采样。

    

    最近，从观测数据进行因果推断在机器学习中得到了广泛应用。虽然存在计算因果效应的可靠且完备的算法，但其中许多算法需要显式访问观测分布上的条件似然，而在高维场景中（例如图像），估计这些似然是困难的。为了解决这个问题，研究人员通过使用神经模型模拟因果关系，并取得了令人印象深刻的结果。然而，这些现有方法中没有一个可以应用于通用场景，例如具有潜在混淆因素的图像数据的因果图，或者获得条件干预样本。在本文中，我们展示了在任意因果图下，通过条件生成模型的推进计算可以计算任何可辨识的因果效应。基于此结果，我们设计了一个基于扩散的方法，可以从任何（条件）干预分布中采样图像。

    Causal inference from observational data has recently found many applications in machine learning. While sound and complete algorithms exist to compute causal effects, many of these algorithms require explicit access to conditional likelihoods over the observational distribution, which is difficult to estimate in the high-dimensional regime, such as with images. To alleviate this issue, researchers have approached the problem by simulating causal relations with neural models and obtained impressive results. However, none of these existing approaches can be applied to generic scenarios such as causal graphs on image data with latent confounders, or obtain conditional interventional samples. In this paper, we show that any identifiable causal effect given an arbitrary causal graph can be computed through push-forward computations of conditional generative models. Based on this result, we devise a diffusion-based approach to sample from any (conditional) interventional distribution on ima
    
[^23]: 可容许预测规划用于机遇受限优化

    Conformal Predictive Programming for Chance Constrained Optimization

    [https://arxiv.org/abs/2402.07407](https://arxiv.org/abs/2402.07407)

    可容许预测规划（CPP）是一种解决受任意随机参数影响的优化问题的方法，通过利用样本和量子引理将机遇受限优化（CCO）问题转化为确定性优化问题，并具备边际概率可行性保证。

    

    在对预测规划（CP）的进展的激励下，我们提出了可容许预测规划（CPP），一种解决机遇受限优化（CCO）问题的方法，即受任意随机参数影响的非线性约束函数的优化问题。CPP利用这些随机参数的样本以及量子引理（CP的核心）将CCO问题转化为确定性优化问题。然后，我们通过：（1）将量子表示为线性规划以及其KKT条件（CPP-KKT）；（2）使用混合整数规划（CPP-MIP）来呈现CPP的两种易于处理的改进。CPP具备对CCO问题进行边际概率可行性保证，这与现有方法（例如样本逼近和场景方法）在概念上有所不同。尽管我们探讨了与样本逼近方法的算法相似之处，但我们强调CPP的优势在于易于扩展。

    Motivated by the advances in conformal prediction (CP), we propose conformal predictive programming (CPP), an approach to solve chance constrained optimization (CCO) problems, i.e., optimization problems with nonlinear constraint functions affected by arbitrary random parameters. CPP utilizes samples from these random parameters along with the quantile lemma -- which is central to CP -- to transform the CCO problem into a deterministic optimization problem. We then present two tractable reformulations of CPP by: (1) writing the quantile as a linear program along with its KKT conditions (CPP-KKT), and (2) using mixed integer programming (CPP-MIP). CPP comes with marginal probabilistic feasibility guarantees for the CCO problem that are conceptually different from existing approaches, e.g., the sample approximation and the scenario approach. While we explore algorithmic similarities with the sample approximation approach, we emphasize that the strength of CPP is that it can easily be ext
    
[^24]: 在多臂赌博机中，可复制性渐进自由

    Replicability is Asymptotically Free in Multi-armed Bandits

    [https://arxiv.org/abs/2402.07391](https://arxiv.org/abs/2402.07391)

    本论文研究在多臂赌博机问题中，通过引入探索-再确定算法和连续淘汰算法，以及谨慎选择置信区间的幅度，实现了可复制性，并证明了当时间界足够大时，可复制算法的额外代价是不必要的。

    

    本研究受可复制的机器学习需求的推动，研究了随机多臂赌博机问题。特别地，我们考虑了一个可复制算法，确保算法的操作序列不受数据集固有随机性的影响。我们观察到，现有算法所需的遗憾值比不可复制算法多$O(1/\rho^2)$倍，其中$\rho$是非复制程度。然而，我们证明了当给定的$\rho$下时间界$T$足够大时，此额外代价是不必要的，前提是谨慎选择置信区间的幅度。我们引入了一个先探索后决策的算法，在决策之前均匀选择动作。此外，我们还研究了一个连续淘汰算法，在每个阶段结束时淘汰次优动作。为了确保这些算法的可复制性，我们将随机性引入决策制定中。

    This work is motivated by the growing demand for reproducible machine learning. We study the stochastic multi-armed bandit problem. In particular, we consider a replicable algorithm that ensures, with high probability, that the algorithm's sequence of actions is not affected by the randomness inherent in the dataset. We observe that existing algorithms require $O(1/\rho^2)$ times more regret than nonreplicable algorithms, where $\rho$ is the level of nonreplication. However, we demonstrate that this additional cost is unnecessary when the time horizon $T$ is sufficiently large for a given $\rho$, provided that the magnitude of the confidence bounds is chosen carefully. We introduce an explore-then-commit algorithm that draws arms uniformly before committing to a single arm. Additionally, we examine a successive elimination algorithm that eliminates suboptimal arms at the end of each phase. To ensure the replicability of these algorithms, we incorporate randomness into their decision-ma
    
[^25]: 无假设测试算法性能的限制

    The Limits of Assumption-free Tests for Algorithm Performance

    [https://arxiv.org/abs/2402.07388](https://arxiv.org/abs/2402.07388)

    这项研究探讨了使用有限数据量回答算法性能问题的基本限制，证明了黑盒测试方法无法准确回答算法在不同训练集上的整体性能和特定模型的性能问题。

    

    算法评价和比较是机器学习和统计学中基本的问题，一个算法在给定的建模任务中表现如何，哪个算法表现最佳？许多方法已经开发出来评估算法性能，通常基于交叉验证策略，将感兴趣的算法在不同的数据子集上重新训练，并评估其在留出数据点上的性能。尽管广泛使用这些程序，但对于这些方法的理论性质尚未完全理解。在这项工作中，我们探讨了在有限的数据量下回答这些问题的一些基本限制。特别地，我们区分了两个问题: 算法$A$在大小为$n$的训练集上学习问题有多好，以及在特定大小为$n$的训练数据集上运行$A$所产生的特定拟合模型有多好？我们的主要结果证明，对于任何将算法视为黑盒的测试方法，无法准确地回答这两个问题。

    Algorithm evaluation and comparison are fundamental questions in machine learning and statistics -- how well does an algorithm perform at a given modeling task, and which algorithm performs best? Many methods have been developed to assess algorithm performance, often based around cross-validation type strategies, retraining the algorithm of interest on different subsets of the data and assessing its performance on the held-out data points. Despite the broad use of such procedures, the theoretical properties of these methods are not yet fully understood. In this work, we explore some fundamental limits for answering these questions with limited amounts of data. In particular, we make a distinction between two questions: how good is an algorithm $A$ at the problem of learning from a training set of size $n$, versus, how good is a particular fitted model produced by running $A$ on a particular training data set of size $n$?   Our main results prove that, for any test that treats the algor
    
[^26]: 回归树用于快速和自适应的预测区间

    Regression Trees for Fast and Adaptive Prediction Intervals

    [https://arxiv.org/abs/2402.07357](https://arxiv.org/abs/2402.07357)

    该论文提出了一种新的、与模型无关的方法族，用于校准具有局部覆盖保证的回归问题的预测区间。这种方法利用回归树和随机森林训练来创建最粗糙的特征空间划分，以近似条件覆盖，提供了准确、快速和自适应的预测区间。

    

    预测模型会犯错，因此需要量化与其预测相关的不确定性。符合性推断已经成为一种强大的工具，可以在点预测周围创建统计上有效的预测区域，但是它在回归问题上的朴素应用会产生非自适应的区域。新的符合性得分，通常依赖于分位数回归器或条件密度估计器，旨在解决这个限制。虽然它们在创建预测带方面很有用，但这些得分与量化任意预测模型周围的不确定性的原始目标脱节。本文提出了一种新的、与模型无关的方法族，用于校准具有局部覆盖保证的回归问题的预测区间。我们的方法是基于追求最粗糙的特征空间划分来近似条件覆盖。我们通过对符合性得分进行回归树和随机森林的训练来创建这个划分。我们的提议将回归树和随机森林应用于符合性推断的新领域，以提供准确、快速和自适应的预测区间。

    Predictive models make mistakes. Hence, there is a need to quantify the uncertainty associated with their predictions. Conformal inference has emerged as a powerful tool to create statistically valid prediction regions around point predictions, but its naive application to regression problems yields non-adaptive regions. New conformal scores, often relying upon quantile regressors or conditional density estimators, aim to address this limitation. Although they are useful for creating prediction bands, these scores are detached from the original goal of quantifying the uncertainty around an arbitrary predictive model. This paper presents a new, model-agnostic family of methods to calibrate prediction intervals for regression problems with local coverage guarantees. Our approach is based on pursuing the coarsest partition of the feature space that approximates conditional coverage. We create this partition by training regression trees and Random Forests on conformity scores. Our proposal
    
[^27]: 一个新的高斯最小最大定理及其应用

    A Novel Gaussian Min-Max Theorem and its Applications

    [https://arxiv.org/abs/2402.07356](https://arxiv.org/abs/2402.07356)

    本文介绍了一个新的高斯最小最大定理，扩展了经典定理对于独立但非恒定分布的情况。此外，该定理在高维统计学、机器学习、非光滑优化和信号处理等领域有广泛的应用。

    

    Gordon的一个著名结果允许比较两个高斯过程的最小最大行为，如果满足某些不等式条件。这个结果的结果包括高斯最小最大（GMT）和凸高斯最小最大（CGMT）定理，这些定理在高维统计学、机器学习、非光滑优化和信号处理方面有广泛的应用。目前为止，没有发现满足这些不等式的其他一对高斯过程。在本文中，我们确定了这样一对新的高斯过程。由此得到的定理将经典的GMT定理和CGMT定理从基本过程中的底层高斯矩阵具有iid行的情况扩展到具有独立但非恒定分布的情况。新的CGMT定理应用于多源高斯回归问题，以及属于的二元分类问题。

    A celebrated result by Gordon allows one to compare the min-max behavior of two Gaussian processes if certain inequality conditions are met. The consequences of this result include the Gaussian min-max (GMT) and convex Gaussian min-max (CGMT) theorems which have had far-reaching implications in high-dimensional statistics, machine learning, non-smooth optimization, and signal processing. Both theorems rely on a pair of Gaussian processes, first identified by Slepian, that satisfy Gordon's comparison inequalities. To date, no other pair of Gaussian processes satisfying these inequalities has been discovered. In this paper, we identify such a new pair. The resulting theorems extend the classical GMT and CGMT Theorems from the case where the underlying Gaussian matrix in the primary process has iid rows to where it has independent but non-identically-distributed ones. The new CGMT is applied to the problems of multi-source Gaussian regression, as well as to binary classification of genera
    
[^28]: 从均场稳态分布中采样

    Sampling from the Mean-Field Stationary Distribution

    [https://arxiv.org/abs/2402.07355](https://arxiv.org/abs/2402.07355)

    本文研究了从均场随机微分方程 (SDE) 的稳态分布中采样的复杂性，并提出了一种解耦的方法。该方法能够在多种情况下提供改进的保证，包括在均场区域优化某些双层神经网络的更好保证。

    

    我们研究了从均场随机微分方程 (SDE) 的稳态分布中采样的复杂性，或者等价地，即包含交互项的概率测度空间上的最小化函数的复杂性。我们的主要洞察是将这个问题的两个关键方面解耦：(1) 通过有限粒子系统逼近均场SDE，通过时间均匀传播混沌，和(2) 通过标准对数凹抽样器从有限粒子稳态分布中采样。我们的方法在概念上更简单，其灵活性允许结合用于算法和理论的最新技术。这导致在许多设置中提供了改进的保证，包括在均场区域优化某些双层神经网络的更好保证。

    We study the complexity of sampling from the stationary distribution of a mean-field SDE, or equivalently, the complexity of minimizing a functional over the space of probability measures which includes an interaction term.   Our main insight is to decouple the two key aspects of this problem: (1) approximation of the mean-field SDE via a finite-particle system, via uniform-in-time propagation of chaos, and (2) sampling from the finite-particle stationary distribution, via standard log-concave samplers. Our approach is conceptually simpler and its flexibility allows for incorporating the state-of-the-art for both algorithms and theory. This leads to improved guarantees in numerous settings, including better guarantees for optimizing certain two-layer neural networks in the mean-field regime.
    
[^29]: 对线性强化学习领域的噪声自适应置信区间及其在贝叶斯优化中的应用

    Noise-Adaptive Confidence Sets for Linear Bandits and Application to Bayesian Optimization

    [https://arxiv.org/abs/2402.07341](https://arxiv.org/abs/2402.07341)

    这项研究提出了一种对线性强化学习领域中未知噪声水平的自适应置信区间，与已有方法相比，在维度较大时具有显著的改进。此外，针对有界奖励，还提出了一种方差自适应置信区间，具有更好的数值性能。

    

    在序贯决策中，适应未知噪声水平是一个非常重要但具有挑战性的问题，因为有效的探索通常需要对噪声水平有一定的了解，而噪声水平通常只能粗略地指定。我们在线性强化学习领域取得了显著进展，主要有两方面。首先，我们提出了一种新颖的置信区间，该置信区间在未知的亚高斯参数σ_*^2上是“半自适应”的，意味着（归一化的）置信宽度与√（dσ_*^2 + σ_0^2）成正比，其中d为维度，σ_0^2为指定的（已知）亚高斯参数，其值可能比σ_*^2大得多。相比于Abbasi-Yadkori等人（2011）的标准置信区间的√（dσ_0^2），这是一个显著的改进，特别是当d较大时。我们证明了这导致了线性强化学习中改进的后悔边界。其次，对于有界奖励，我们提出了一种新颖的方差自适应置信区间，具有更好的数值性能。

    Adapting to a priori unknown noise level is a very important but challenging problem in sequential decision-making as efficient exploration typically requires knowledge of the noise level, which is often loosely specified. We report significant progress in addressing this issue in linear bandits in two respects. First, we propose a novel confidence set that is `semi-adaptive' to the unknown sub-Gaussian parameter $\sigma_*^2$ in the sense that the (normalized) confidence width scales with $\sqrt{d\sigma_*^2 + \sigma_0^2}$ where $d$ is the dimension and $\sigma_0^2$ is the specified sub-Gaussian parameter (known) that can be much larger than $\sigma_*^2$. This is a significant improvement over $\sqrt{d\sigma_0^2}$ of the standard confidence set of Abbasi-Yadkori et al. (2011), especially when $d$ is large. We show that this leads to an improved regret bound in linear bandits. Second, for bounded rewards, we propose a novel variance-adaptive confidence set that has a much improved numeri
    
[^30]: 用图神经网络对随机几何图进行对齐

    Random Geometric Graph Alignment with Graph Neural Networks

    [https://arxiv.org/abs/2402.07340](https://arxiv.org/abs/2402.07340)

    本文研究了在图对齐问题中，通过图神经网络可以高概率恢复正确的顶点对齐。通过特定的特征稀疏性和噪声水平条件，我们证明了图神经网络的有效性，并与直接匹配方法进行了比较。

    

    我们研究了在顶点特征信息存在的情况下，图神经网络在图对齐问题中的性能。具体而言，给定两个独立扰动的单个随机几何图以及噪声稀疏特征的情况下，任务是恢复两个图的顶点之间的未知一对一映射关系。我们证明在特征向量的稀疏性和噪声水平满足一定条件的情况下，经过精心设计的单层图神经网络可以在很高的概率下通过图结构来恢复正确的顶点对齐。我们还证明了噪声水平的条件上界，仅存在对数因子差距。最后，我们将图神经网络的性能与直接在噪声顶点特征上求解分配问题进行了比较。我们证明了当噪声水平至少为常数时，这种直接匹配会导致恢复不完全，而图神经网络可以容忍n

    We characterize the performance of graph neural networks for graph alignment problems in the presence of vertex feature information. More specifically, given two graphs that are independent perturbations of a single random geometric graph with noisy sparse features, the task is to recover an unknown one-to-one mapping between the vertices of the two graphs. We show under certain conditions on the sparsity and noise level of the feature vectors, a carefully designed one-layer graph neural network can with high probability recover the correct alignment between the vertices with the help of the graph structure. We also prove that our conditions on the noise level are tight up to logarithmic factors. Finally we compare the performance of the graph neural network to directly solving an assignment problem on the noisy vertex features. We demonstrate that when the noise level is at least constant this direct matching fails to have perfect recovery while the graph neural network can tolerate n
    
[^31]: 一种关于一般KL正则化偏好下纳什学习从人类反馈中的理论分析

    A Theoretical Analysis of Nash Learning from Human Feedback under General KL-Regularized Preference

    [https://arxiv.org/abs/2402.07314](https://arxiv.org/abs/2402.07314)

    本论文从理论层面分析了一种关于一般偏好下纳什学习从人类反馈中的方法，通过对两个竞争的LLM进行博弈来找到一种一致生成响应的策略。

    

    来自人类反馈的强化学习（RLHF）从一个概率偏好模型提供的偏好信号中学习，该模型以一个提示和两个响应作为输入，并产生一个分数，表示对一个响应相对于另一个响应的偏好程度。迄今为止，最流行的RLHF范式是基于奖励的，它从奖励建模的初始步骤开始，然后使用构建的奖励为后续的奖励优化阶段提供奖励信号。然而，奖励函数的存在是一个强假设，基于奖励的RLHF在表达能力上有局限性，不能捕捉到真实世界中复杂的人类偏好。在这项工作中，我们为最近提出的学习范式Nash学习从人类反馈（NLHF）提供了理论洞察力，该学习范式考虑了一个一般的偏好模型，并将对齐过程定义为两个竞争的LLM之间的博弈。学习目标是找到一个一致生成响应的策略。

    Reinforcement Learning from Human Feedback (RLHF) learns from the preference signal provided by a probabilistic preference model, which takes a prompt and two responses as input, and produces a score indicating the preference of one response against another. So far, the most popular RLHF paradigm is reward-based, which starts with an initial step of reward modeling, and the constructed reward is then used to provide a reward signal for the subsequent reward optimization stage. However, the existence of a reward function is a strong assumption and the reward-based RLHF is limited in expressivity and cannot capture the real-world complicated human preference.   In this work, we provide theoretical insights for a recently proposed learning paradigm, Nash learning from human feedback (NLHF), which considered a general preference model and formulated the alignment process as a game between two competitive LLMs. The learning objective is to find a policy that consistently generates responses
    
[^32]: HyperBERT:将混合超图感知层与语言模型用于文本属性超图上的节点分类

    HyperBERT: Mixing Hypergraph-Aware Layers with Language Models for Node Classification on Text-Attributed Hypergraphs

    [https://arxiv.org/abs/2402.07309](https://arxiv.org/abs/2402.07309)

    本文提出了HyperBERT模型，通过在预训练的BERT模型中引入超图感知层，克服了现有方法在节点分类任务上难以捕捉超图结构信息和文本属性的局限性，提高了模型的效果和泛化能力。

    

    超图通过复杂的拓扑结构标记，表达多个实体之间的高阶相互作用，其中超边扮演重要角色。最近，基于超图的深度学习方法在学习文本属性超图上的节点分类问题中引起了越来越多的研究关注。然而，现有方法往往难以同时捕捉超图结构信息的全部内容和节点属性中的丰富语言属性，这在很大程度上影响了它们的效果和泛化能力。为了克服这些挑战，我们探索了如何通过为节点分类任务进一步增强预训练的BERT模型，引入专门的超图感知层。这些层将高阶结构归纳偏差引入语言模型中，从而提高模型利用超图结构中的高阶上下文信息和文本中的语义信息的能力。

    Hypergraphs are marked by complex topology, expressing higher-order interactions among multiple entities with hyperedges. Lately, hypergraph-based deep learning methods to learn informative data representations for the problem of node classification on text-attributed hypergraphs have garnered increasing research attention. However, existing methods struggle to simultaneously capture the full extent of hypergraph structural information and the rich linguistic attributes inherent in the nodes attributes, which largely hampers their effectiveness and generalizability. To overcome these challenges, we explore ways to further augment a pretrained BERT model with specialized hypergraph-aware layers for the task of node classification. Such layers introduce higher-order structural inductive bias into the language model, thus improving the model's capacity to harness both higher-order context information from the hypergraph structure and semantic information present in text. In this paper, we
    
[^33]: 自洽的符合预测

    Self-Consistent Conformal Prediction

    [https://arxiv.org/abs/2402.07307](https://arxiv.org/abs/2402.07307)

    自洽的符合预测方法能够提供既符合校准的预测又符合以模型预测的动作为条件的预测区间，为决策者提供了严格的、针对具体动作的决策保证。

    

    在机器学习指导下的决策中，决策者通常在具有相同预测结果的情境中采取相同的行动。符合预测帮助决策者量化动作的结果不确定性，从而实现更好的风险管理。受这种观点的启发，我们引入了自洽的符合预测，它产生了既符合Venn-Abers校准的预测，又符合以模型预测引发的动作为条件的符合预测区间。我们的方法可以后验地应用于任何黑盒预测器，提供严格的、针对具体动作的决策保证。数值实验表明，我们的方法在区间的效率和条件的有效性之间达到了平衡。

    In decision-making guided by machine learning, decision-makers often take identical actions in contexts with identical predicted outcomes. Conformal prediction helps decision-makers quantify outcome uncertainty for actions, allowing for better risk management. Inspired by this perspective, we introduce self-consistent conformal prediction, which yields both Venn-Abers calibrated predictions and conformal prediction intervals that are valid conditional on actions prompted by model predictions. Our procedure can be applied post-hoc to any black-box predictor to provide rigorous, action-specific decision-making guarantees. Numerical experiments show our approach strikes a balance between interval efficiency and conditional validity.
    
[^34]: 估计几何遗传马尔可夫过程的混合系数

    Estimating the Mixing Coefficients of Geometrically Ergodic Markov Processes

    [https://arxiv.org/abs/2402.07296](https://arxiv.org/abs/2402.07296)

    该论文提出了一种方法来估计几何遗传马尔可夫过程的混合系数，我们通过在满足特定条件和无需密度假设的情况下，得到了估计器的预期误差收敛速度和高概率界限。

    

    我们提出了一种方法来估计实值几何遗传马尔可夫过程的单个β-混合系数从一个单一的样本路径X0，X1，...，Xn。在对密度的标准光滑条件下，即对于每个m，对$(X_0,X_m)$对的联合密度都属于某个已知$s>0$的 Besov 空间$B^s_{1,\infty}(\mathbb R^2)$，我们得到了我们在这种情况下的估计器的预期误差的收敛速度为$\mathcal{O}(\log(n) n^{-[s]/(2[s]+2)})$ 的收敛速度。我们通过对估计误差的高概率界限进行了补充，并在状态空间有限的情况下获得了这些界限的类比。在这种情况下不需要密度的假设；预期误差率显示为$\mathcal O(\log(

    We propose methods to estimate the individual $\beta$-mixing coefficients of a real-valued geometrically ergodic Markov process from a single sample-path $X_0,X_1, \dots,X_n$. Under standard smoothness conditions on the densities, namely, that the joint density of the pair $(X_0,X_m)$ for each $m$ lies in a Besov space $B^s_{1,\infty}(\mathbb R^2)$ for some known $s>0$, we obtain a rate of convergence of order $\mathcal{O}(\log(n) n^{-[s]/(2[s]+2)})$ for the expected error of our estimator in this case\footnote{We use $[s]$ to denote the integer part of the decomposition $s=[s]+\{s\}$ of $s \in (0,\infty)$ into an integer term and a {\em strictly positive} remainder term $\{s\} \in (0,1]$.}. We complement this result with a high-probability bound on the estimation error, and further obtain analogues of these bounds in the case where the state-space is finite. Naturally no density assumptions are required in this setting; the expected error rate is shown to be of order $\mathcal O(\log(
    
[^35]: 神经网络中的深度分离：将维度与准确度分离

    Depth Separations in Neural Networks: Separating the Dimension from the Accuracy

    [https://arxiv.org/abs/2402.07248](https://arxiv.org/abs/2402.07248)

    通过研究深度2和深度3神经网络在逼近Lipschitz目标函数时的分离性质，证明了维度诅咒也会在深度2逼近中存在，即使目标函数可以使用深度3高效表示。这为以前确定深度要求的下界提供了新的观点，并且适用于多种激活函数。

    

    我们证明了深度2和深度3神经网络在逼近一个$\mathcal{O}(1)$-Lipschitz目标函数至常数精度时的指数分离，对于支持在$[0,1]^{d}$上的分布，假设权重指数有界。这解决了在\citet{safran2019depth}中提出的一个问题，并证明了维度诅咒在深度2逼近中的存在，即使在目标函数可以使用深度3高效表示的情况下也是如此。以前，将深度2和深度3分离的下界要求至少有一个Lipschitz参数、目标准确度或逼近域的大小（某种度量）与输入维度多项式地缩放，而我们保持前两者不变，并将我们的域限制在单位超立方体上。我们的下界适用于各种激活函数，并基于一种新的平均情况到最坏情况的随机自约化论证的应用，以减少

    We prove an exponential separation between depth 2 and depth 3 neural networks, when approximating an $\mathcal{O}(1)$-Lipschitz target function to constant accuracy, with respect to a distribution with support in $[0,1]^{d}$, assuming exponentially bounded weights. This addresses an open problem posed in \citet{safran2019depth}, and proves that the curse of dimensionality manifests in depth 2 approximation, even in cases where the target function can be represented efficiently using depth 3. Previously, lower bounds that were used to separate depth 2 from depth 3 required that at least one of the Lipschitz parameter, target accuracy or (some measure of) the size of the domain of approximation scale polynomially with the input dimension, whereas we fix the former two and restrict our domain to the unit hypercube. Our lower bound holds for a wide variety of activation functions, and is based on a novel application of an average- to worst-case random self-reducibility argument, to reduce
    
[^36]: 面向扩散生成模型的快速随机采样方法

    Towards Fast Stochastic Sampling in Diffusion Generative Models

    [https://arxiv.org/abs/2402.07211](https://arxiv.org/abs/2402.07211)

    本文提出了一种在扩散生成模型中进行快速随机采样的方法，通过对分裂积分器进行原则性修改，实现了更高的采样效率。在CIFAR-10数据集上进行实验，100次网络函数评估下的FID分数为2.36。

    

    扩散模型在推理时生成样本的速度较慢。尽管最近有一些努力在改善扩散模型的随机采样效率，但仍然有待改进。我们提出了基于分裂积分器的预训练扩散模型的快速随机采样方法。分裂积分器通常在分子动力学中使用，通过巧妙地在涉及数据、辅助或噪声变量的数值更新之间交替来提高采样效率。然而，我们发现对于快速采样，简单应用分裂积分器是次优的。因此，我们提出了几种原则上修改了简单分裂采样器以提高采样效率的方法，并将得到的采样器称为减小分裂积分器。在CIFAR-10数据集上使用相空间朗之万扩散 (PSLD) [Pandey \& Mandt, 2023] 的背景下，我们的随机采样器在仅进行100次网络函数评估后，实现了2.36的FID分数。

    Diffusion models suffer from slow sample generation at inference time. Despite recent efforts, improving the sampling efficiency of stochastic samplers for diffusion models remains a promising direction. We propose Splitting Integrators for fast stochastic sampling in pre-trained diffusion models in augmented spaces. Commonly used in molecular dynamics, splitting-based integrators attempt to improve sampling efficiency by cleverly alternating between numerical updates involving the data, auxiliary, or noise variables. However, we show that a naive application of splitting integrators is sub-optimal for fast sampling. Consequently, we propose several principled modifications to naive splitting samplers for improving sampling efficiency and denote the resulting samplers as Reduced Splitting Integrators. In the context of Phase Space Langevin Diffusion (PSLD) [Pandey \& Mandt, 2023] on CIFAR-10, our stochastic sampler achieves an FID score of 2.36 in only 100 network function evaluations 
    
[^37]: 梯度噪声的隐性偏见：从对称性角度来看

    The Implicit Bias of Gradient Noise: A Symmetry Perspective

    [https://arxiv.org/abs/2402.07193](https://arxiv.org/abs/2402.07193)

    本研究通过对对称性的存在进行分析，揭示了梯度噪声在随机梯度下降中的隐性偏见。我们发现不同类型的对称性会导致不同的学习动态，其中一类对称性可以自然收敛，而另一类对称性几乎总是发散。此外，我们的研究结果适用于没有对称性的损失函数，对于理解训练动态和解释相关实际问题具有普适性。

    

    我们对随机梯度下降（SGD）在损失函数存在连续对称性时的学习动态进行了表征，说明了SGD和梯度下降之间的分歧是多么巨大。我们展示了根据对称性对学习动态的影响方式，我们可以将一族对称性分为两类。对于一类对称性，SGD自然地收敛到具有平衡和对齐梯度噪声的解。对于另一类对称性，SGD几乎总是发散的。然后，我们展示了即使损失函数中没有对称性，我们的结果依然适用并可以帮助我们理解训练动态。我们的主要结果是普遍的，它只依赖于对称性的存在，而与损失函数的细节无关。我们证明了所提出的理论对于逐步变形和平坦化提供了解释，并可以应用于常见的实际问题，如表示正则化。

    We characterize the learning dynamics of stochastic gradient descent (SGD) when continuous symmetry exists in the loss function, where the divergence between SGD and gradient descent is dramatic. We show that depending on how the symmetry affects the learning dynamics, we can divide a family of symmetry into two classes. For one class of symmetry, SGD naturally converges to solutions that have a balanced and aligned gradient noise. For the other class of symmetry, SGD will almost always diverge. Then, we show that our result remains applicable and can help us understand the training dynamics even when the symmetry is not present in the loss function. Our main result is universal in the sense that it only depends on the existence of the symmetry and is independent of the details of the loss function. We demonstrate that the proposed theory offers an explanation of progressive sharpening and flattening and can be applied to common practical problems such as representation normalization, 
    
[^38]: 通过张量化随机投影改进局部敏感哈希LSH

    Improving LSH via Tensorized Random Projection

    [https://arxiv.org/abs/2402.07189](https://arxiv.org/abs/2402.07189)

    本文提出了CP-E2LSH和TT-E2LSH两种方法，用于改进局部敏感哈希算法LSH，在处理张量数据的欧几里得距离和余弦相似度时能够提供更快和更空间有效的结果。

    

    局部敏感哈希(LSH)是数据科学家用于近似最近邻搜索问题的基本算法工具，已在许多大规模数据处理应用中广泛使用，如近似重复检测、最近邻搜索、聚类等。在本文中，我们旨在提出更快和空间更有效的局部敏感哈希函数，用于张量数据的欧几里得距离和余弦相似度。通常，对于张量数据获得LSH的朴素方法涉及将张量重塑为向量，然后应用现有的向量数据LSH方法(E2LSH和SRP)。然而，对于高阶张量，这种方法变得不切实际，因为重塑向量的大小在张量的阶数中呈指数增长。因此，LSH参数的大小呈指数增加。为解决这个问题，我们提出了两种欧几里得距离和余弦相似度的LSH方法，分别是CP-E2LSH和TT-E2LSH。

    Locality sensitive hashing (LSH) is a fundamental algorithmic toolkit used by data scientists for approximate nearest neighbour search problems that have been used extensively in many large scale data processing applications such as near duplicate detection, nearest neighbour search, clustering, etc. In this work, we aim to propose faster and space efficient locality sensitive hash functions for Euclidean distance and cosine similarity for tensor data. Typically, the naive approach for obtaining LSH for tensor data involves first reshaping the tensor into vectors, followed by applying existing LSH methods for vector data $E2LSH$ and $SRP$. However, this approach becomes impractical for higher order tensors because the size of the reshaped vector becomes exponential in the order of the tensor. Consequently, the size of LSH parameters increases exponentially. To address this problem, we suggest two methods for LSH for Euclidean distance and cosine similarity, namely $CP-E2LSH$, $TT-E2LSH
    
[^39]: PASOA-基于粒子的贝叶斯最优自适应设计

    PASOA- PArticle baSed Bayesian Optimal Adaptive design

    [https://arxiv.org/abs/2402.07160](https://arxiv.org/abs/2402.07160)

    PASOA是一种新的贝叶斯实验设计程序，通过提供连续的后验分布的准确估计，同时执行顺序设计优化和参数推断。该方法使用 stochastic optimization 和 tempered SMC 来最大化期望信息增益，并提供了一致性的最优设计估计。

    

    我们提出了一种名为PASOA的新程序，用于贝叶斯实验设计，通过同时提供连续的后验分布的准确估计来执行顺序设计优化。顺序设计过程通过对比估计原则进行，使用随机优化和顺序蒙特卡罗（SMC）采样器来最大化期望信息增益（EIG）。由于连续后验分布之间的距离越大，获得的信息增益越大，因此这个EIG目标可能会恶化经典SMC的性能。为了解决这个问题，提出了温度调节，既可以获得大的信息增益，又可以获得准确的SMC采样，我们证明这对性能来说是至关重要的。这种随机优化和温度调节的新颖组合允许同时处理设计优化和参数推断。我们证明了所得到的最优设计估计量具有一致性。数值实验表明，我们的方法在相同计算预算下比其他方法更好地优化了设计。

    We propose a new procedure named PASOA, for Bayesian experimental design, that performs sequential design optimization by simultaneously providing accurate estimates of successive posterior distributions for parameter inference. The sequential design process is carried out via a contrastive estimation principle, using stochastic optimization and Sequential Monte Carlo (SMC) samplers to maximise the Expected Information Gain (EIG). As larger information gains are obtained for larger distances between successive posterior distributions, this EIG objective may worsen classical SMC performance. To handle this issue, tempering is proposed to have both a large information gain and an accurate SMC sampling, that we show is crucial for performance. This novel combination of stochastic optimization and tempered SMC allows to jointly handle design optimization and parameter inference. We provide a proof that the obtained optimal design estimators benefit from some consistency property. Numerical
    
[^40]: 针对私有统计推断的重采样方法

    Resampling methods for Private Statistical Inference

    [https://arxiv.org/abs/2402.07131](https://arxiv.org/abs/2402.07131)

    这项研究提出了两种私有变体的非参数bootstrap方法，用于在差分隐私的情况下构建置信区间。方法在计算效率和置信区间长度上相比现有方法有显著改进。

    

    我们考虑使用差分隐私构建置信区间的任务。我们提出了两种私有变体的非参数bootstrap方法，该方法在数据的分区上私下计算多个“小”bootstrap的结果的中位数，并给出了得到的置信区间的渐进覆盖误差上界。对于固定的差分隐私参数ε，我们的方法在样本大小n上的误差率与非私有bootstrap相当，只是在对数因子内。我们使用真实数据和合成数据在均值估计、中位数估计和逻辑回归方面对我们的方法进行了经验验证。我们的方法在提供类似的覆盖精度的同时，比以前的方法提供了显著缩短（大约10倍）的置信区间。

    We consider the task of constructing confidence intervals with differential privacy. We propose two private variants of the non-parametric bootstrap, which privately compute the median of the results of multiple ``little'' bootstraps run on partitions of the data and give asymptotic bounds on the coverage error of the resulting confidence intervals. For a fixed differential privacy parameter $\epsilon$, our methods enjoy the same error rates as that of the non-private bootstrap to within logarithmic factors in the sample size $n$. We empirically validate the performance of our methods for mean estimation, median estimation, and logistic regression with both real and synthetic data. Our methods achieve similar coverage accuracy to existing methods (and non-private baselines) while providing notably shorter ($\gtrsim 10$ times) confidence intervals than previous approaches.
    
[^41]: 对Adam的预条件效应进行量化

    Towards Quantifying the Preconditioning Effect of Adam

    [https://arxiv.org/abs/2402.07114](https://arxiv.org/abs/2402.07114)

    本论文量化了Adam的预条件效应，结果表明Adam能够减轻病态条件的影响，但会受到维度的限制。

    

    本文对Adam的预条件效应进行了详细分析，并量化了Adam在减轻病态条件（困扰梯度下降法）上的作用程度。我们的关键发现是，Adam在病态条件上能够减少依赖于Hessian矩阵条件数的程度，但代价是会受到与维度有关的因素影响。具体来说，对于一个具有对角Hessian矩阵、条件数为κ的d维二次函数，我们证明了在没有动量的Adam中，控制迭代复杂度的有效条件数类似量为O(min(d, κ))。对于一个对角占优的Hessian矩阵，我们获得相应量的上界为O(min(d√(dκ), κ))。因此，当d < O(κ^p)，其中p = 1适用于对角Hessian矩阵时，我们可以得到这种量的界限。

    There is a notable dearth of results characterizing the preconditioning effect of Adam and showing how it may alleviate the curse of ill-conditioning -- an issue plaguing gradient descent (GD). In this work, we perform a detailed analysis of Adam's preconditioning effect for quadratic functions and quantify to what extent Adam can mitigate the dependence on the condition number of the Hessian. Our key finding is that Adam can suffer less from the condition number but at the expense of suffering a dimension-dependent quantity. Specifically, for a $d$-dimensional quadratic with a diagonal Hessian having condition number $\kappa$, we show that the effective condition number-like quantity controlling the iteration complexity of Adam without momentum is $\mathcal{O}(\min(d, \kappa))$. For a diagonally dominant Hessian, we obtain a bound of $\mathcal{O}(\min(d \sqrt{d \kappa}, \kappa))$ for the corresponding quantity. Thus, when $d < \mathcal{O}(\kappa^p)$ where $p = 1$ for a diagonal Hessia
    
[^42]: 自我纠正自我消耗循环用于生成模型训练

    Self-Correcting Self-Consuming Loops for Generative Model Training

    [https://arxiv.org/abs/2402.07087](https://arxiv.org/abs/2402.07087)

    本论文研究了使用合成数据进行生成模型训练时可能出现的自我消耗循环问题，并提出了一种通过引入理想的修正函数来稳定训练的方法。同时，我们还提出了自我修正函数来近似理想的修正函数，并通过实验证实了其有效性。

    

    随着合成数据在互联网上的质量越来越高以及数量不断增加，机器学习模型越来越多地在人工和机器生成的数据的混合上进行训练。尽管使用合成数据进行表征学习的成功案例有很多，但是在生成模型训练中使用合成数据会产生"自我消耗循环"，这可能导致训练不稳定甚至崩溃，除非满足某些条件。我们的论文旨在稳定自我消耗的生成模型训练。我们的理论结果表明，通过引入一个理想的修正函数，将数据点映射为更有可能来自真实数据分布的样本，可以使自我消耗循环的稳定性呈指数增加。然后，我们提出了自我修正函数，它依赖于专家知识（例如，编程在模拟器中的物理定律），并且旨在自动且大规模地近似理想的修正函数。我们通过实验证实了自我纠正自我消耗循环在生成模型训练中的有效性。

    As synthetic data becomes higher quality and proliferates on the internet, machine learning models are increasingly trained on a mix of human- and machine-generated data. Despite the successful stories of using synthetic data for representation learning, using synthetic data for generative model training creates "self-consuming loops" which may lead to training instability or even collapse, unless certain conditions are met. Our paper aims to stabilize self-consuming generative model training. Our theoretical results demonstrate that by introducing an idealized correction function, which maps a data point to be more likely under the true data distribution, self-consuming loops can be made exponentially more stable. We then propose self-correction functions, which rely on expert knowledge (e.g. the laws of physics programmed in a simulator), and aim to approximate the idealized corrector automatically and at scale. We empirically validate the effectiveness of self-correcting self-consum
    
[^43]: 基于独立线性函数逼近的马尔科夫博弈的样本复杂度改进

    Refined Sample Complexity for Markov Games with Independent Linear Function Approximation

    [https://arxiv.org/abs/2402.07082](https://arxiv.org/abs/2402.07082)

    本文在独立线性函数逼近的马尔科夫博弈中，通过改进AVLPR框架，提出了基于数据依赖的悲观估计方法，解决了多智能体的诅咒问题。

    

    马尔科夫博弈（MG）是多智能体强化学习（MARL）中的重要模型。长期以来人们一直认为“多智能体的诅咒”（即算法性能随着智能体数量指数级下降）是不可避免的，直到最近几篇作品（Daskalakis等人，2023年；Cui等人，2023年；Wang等人，2023年）。这些作品确实解决了多智能体的诅咒，当状态空间极大且（线性）函数逼近被应用时，它们要么具有更慢的收敛速度$O(T^{-1/4})$，要么在行动数$A_{\max}$上带来多项式依赖——尽管在单智能体情况下即使损失函数可以随时间任意变化（Dai等人，2023年），也可避免这种依赖。本文首先通过Wang等人（2023年）的“AVLPR”框架精化，洞察了基于数据的（即随机的）悲观估计子优化差距，从而允许更广泛的插件算法选择。当专门应用于MGs时，这一方法能够处理独立的情况。

    Markov Games (MG) is an important model for Multi-Agent Reinforcement Learning (MARL). It was long believed that the "curse of multi-agents" (i.e., the algorithmic performance drops exponentially with the number of agents) is unavoidable until several recent works (Daskalakis et al., 2023; Cui et al., 2023; Wang et al., 2023. While these works did resolve the curse of multi-agents, when the state spaces are prohibitively large and (linear) function approximations are deployed, they either had a slower convergence rate of $O(T^{-1/4})$ or brought a polynomial dependency on the number of actions $A_{\max}$ -- which is avoidable in single-agent cases even when the loss functions can arbitrarily vary with time (Dai et al., 2023). This paper first refines the `AVLPR` framework by Wang et al. (2023), with an insight of *data-dependent* (i.e., stochastic) pessimistic estimation of the sub-optimality gap, allowing a broader choice of plug-in algorithms. When specialized to MGs with independent
    
[^44]: 快速UCB类型算法用于具有重和超重对称噪声的随机赌博机问题

    Fast UCB-type algorithms for stochastic bandits with heavy and super heavy symmetric noise

    [https://arxiv.org/abs/2402.07062](https://arxiv.org/abs/2402.07062)

    本研究提出了一种基于凸优化方法和不精确预测模型的新UCB类型算法，用于解决具有重和超重对称噪声的随机赌博机问题。通过理论和实验结果表明，在奖励中存在对称噪声的情况下，该算法能够达到更好的性能，相比于一般下界能够获得更小的遗憾界。即使奖励分布没有期望，该算法仍然有效。

    

    在本研究中，我们提出了一种基于一般凸优化方法和不精确的预测模型的UCB类型算法构建方法，并推导了与优化方法收敛速度相对应的遗憾界。我们提出了一种新的算法Clipped-SGD-UCB，并通过理论和经验结果表明，在奖励中存在对称噪声的情况下，可以达到$O(\log T\sqrt{KT\log T})$的遗憾界，而不是$O\left (T^{\frac{1}{1+\alpha}} K^{\frac{\alpha}{1+\alpha}} \right)$，该界是当奖励分布满足$\mathbb{E}_{X \in D}[|X|^{1+\alpha}] \leq \sigma^{1+\alpha}$（$\alpha \in (0, 1]$）时的一般下界。此外，即使奖励分布没有期望，即，当$\alpha<0$时，同样的界限也成立。

    In this study, we propose a new method for constructing UCB-type algorithms for stochastic multi-armed bandits based on general convex optimization methods with an inexact oracle. We derive the regret bounds corresponding to the convergence rates of the optimization methods. We propose a new algorithm Clipped-SGD-UCB and show, both theoretically and empirically, that in the case of symmetric noise in the reward, we can achieve an $O(\log T\sqrt{KT\log T})$ regret bound instead of $O\left (T^{\frac{1}{1+\alpha}} K^{\frac{\alpha}{1+\alpha}} \right)$ for the case when the reward distribution satisfies $\mathbb{E}_{X \in D}[|X|^{1+\alpha}] \leq \sigma^{1+\alpha}$ ($\alpha \in (0, 1])$, i.e. perform better than it is assumed by the general lower bound for bandits with heavy-tails. Moreover, the same bound holds even when the reward distribution does not have the expectation, that is, when $\alpha<0$.
    
[^45]: 理解通过使用近似损失进行采样的训练加速

    Understanding the Training Speedup from Sampling with Approximate Losses

    [https://arxiv.org/abs/2402.07052](https://arxiv.org/abs/2402.07052)

    本文研究利用近似损失进行样本采样的训练加速方法，通过贪婪策略选择具有大约损失的样本，减少选择的开销，并证明其收敛速度优于随机选择。同时开发了使用中间层表示获取近似损失的SIFT方法，并在训练BERT模型上取得了显著的提升。

    

    众所周知，选择具有较大损失/梯度的样本可以显著减少训练步骤的数量。然而，选择的开销往往过高，无法在总体训练时间方面获得有意义的提升。在本文中，我们专注于选择具有大约损失的样本的贪婪方法，而不是准确损失，以减少选择的开销。对于平滑凸损失，我们证明了这种贪婪策略可以在比随机选择更少的迭代次数内收敛到平均损失的最小值的常数因子。我们还理论上量化了近似水平的影响。然后，我们开发了使用中间层表示获取近似损失以进行样本选择的SIFT。我们评估了SIFT在训练一个具有1.1亿参数的12层BERT基础模型上的任务，并展示了显著的提升（以训练时间和反向传播步骤的数量衡量）。

    It is well known that selecting samples with large losses/gradients can significantly reduce the number of training steps. However, the selection overhead is often too high to yield any meaningful gains in terms of overall training time. In this work, we focus on the greedy approach of selecting samples with large \textit{approximate losses} instead of exact losses in order to reduce the selection overhead. For smooth convex losses, we show that such a greedy strategy can converge to a constant factor of the minimum value of the average loss in fewer iterations than the standard approach of random selection. We also theoretically quantify the effect of the approximation level. We then develop SIFT which uses early exiting to obtain approximate losses with an intermediate layer's representations for sample selection. We evaluate SIFT on the task of training a 110M parameter 12-layer BERT base model and show significant gains (in terms of training hours and number of backpropagation step
    
[^46]: 用于建模具有beta边际分布的相关随机概率的logistic-beta过程

    Logistic-beta processes for modeling dependent random probabilities with beta marginals

    [https://arxiv.org/abs/2402.07048](https://arxiv.org/abs/2402.07048)

    本文提出了一种新颖的logistic-beta过程用于建模具有beta边际分布的相关随机概率。该过程具有灵活的相关结构和计算优势，并通过非参数二分类回归模拟研究进行了验证。

    

    beta分布被广泛应用于概率建模，并在统计学和机器学习中被广泛使用，尤其在贝叶斯非参数领域。尽管其被广泛使用，但在建模相关随机概率的灵活和计算方便的随机过程扩展方面，相关工作有限。我们提出了一种新颖的随机过程，称为logistic-beta过程，其logistic变换生成具有常见beta边际分布的随机过程。类似于高斯过程，logistic-beta过程可以建模离散和连续域（例如空间或时间）上的相关性，并通过相关核函数具有高度灵活的相关结构。此外，它的正态方差-均值混合表示导致了高效的后验推理算法。通过非参数二分类回归模拟研究，展示了logistic-beta过程的灵活性和计算优势。

    The beta distribution serves as a canonical tool for modeling probabilities and is extensively used in statistics and machine learning, especially in the field of Bayesian nonparametrics. Despite its widespread use, there is limited work on flexible and computationally convenient stochastic process extensions for modeling dependent random probabilities. We propose a novel stochastic process called the logistic-beta process, whose logistic transformation yields a stochastic process with common beta marginals. Similar to the Gaussian process, the logistic-beta process can model dependence on both discrete and continuous domains, such as space or time, and has a highly flexible dependence structure through correlation kernels. Moreover, its normal variance-mean mixture representation leads to highly effective posterior inference algorithms. The flexibility and computational benefits of logistic-beta processes are demonstrated through nonparametric binary regression simulation studies. Fur
    
[^47]: 均场极限下图神经网络的泛化误差

    Generalization Error of Graph Neural Networks in the Mean-field Regime

    [https://arxiv.org/abs/2402.07025](https://arxiv.org/abs/2402.07025)

    该论文在均场极限下提供了一个理论框架，评估了图神经网络在过参数化情况下的泛化误差，通过推导出收敛速度为$O(1/n)$的上界，为我们对网络在未见数据上的性能提供了理论保证。

    

    该工作提供了一个理论框架，用于评估在过参数化的情况下通过图神经网络进行图分类任务的泛化误差，即参数数量超过数据点数量的情况。我们研究了两种广泛使用的图神经网络类型：图卷积神经网络和消息传递图神经网络。在本研究之前，关于过参数化情况下泛化误差的现有界限缺乏信息，限制了我们对过参数化网络性能的理解。我们的创新方法是在均场极限下推导出上界，以评估这些图神经网络的泛化误差。我们建立了以$O(1/n)$收敛速度的上界，其中$n$是图样本的数量。这些上界为在具有挑战性的过参数化情况下网络在未见数据上的性能提供了理论上的保证，从而对我们的理解做出了贡献。

    This work provides a theoretical framework for assessing the generalization error of graph classification tasks via graph neural networks in the over-parameterized regime, where the number of parameters surpasses the quantity of data points. We explore two widely utilized types of graph neural networks: graph convolutional neural networks and message passing graph neural networks. Prior to this study, existing bounds on the generalization error in the over-parametrized regime were uninformative, limiting our understanding of over-parameterized network performance. Our novel approach involves deriving upper bounds within the mean-field regime for evaluating the generalization error of these graph neural networks. We establish upper bounds with a convergence rate of $O(1/n)$, where $n$ is the number of graph samples. These upper bounds offer a theoretical assurance of the networks' performance on unseen data in the challenging over-parameterized regime and overall contribute to our under
    
[^48]: 基于树集成的情境多臂老虎机

    Tree Ensembles for Contextual Bandits

    [https://arxiv.org/abs/2402.06963](https://arxiv.org/abs/2402.06963)

    本论文提出了一种基于树集成的情境多臂老虎机新框架，通过整合两种广泛使用的老虎机方法，在标准和组合设置中实现了优于基于神经网络的方法的性能，在减少后悔和计算时间方面表现出更出色的性能。

    

    我们提出了一个基于树集成的情境多臂老虎机的新框架。我们的框架将两种广泛使用的老虎机方法，上信心界和汤普森抽样，整合到标准和组合设置中。通过使用流行的树集成方法XGBoost进行多次实验研究，我们展示了我们框架的有效性。当应用于基准数据集和道路网络导航的真实世界应用时，与基于神经网络的最先进方法相比，我们的方法在减少后悔和计算时间方面表现出更好的性能。

    We propose a novel framework for contextual multi-armed bandits based on tree ensembles. Our framework integrates two widely used bandit methods, Upper Confidence Bound and Thompson Sampling, for both standard and combinatorial settings. We demonstrate the effectiveness of our framework via several experimental studies, employing XGBoost, a popular tree ensemble method. Compared to state-of-the-art methods based on neural networks, our methods exhibit superior performance in terms of both regret minimization and computational runtime, when applied to benchmark datasets and the real-world application of navigation over road networks.
    
[^49]: 使用加权虚拟观测实现高效的增量信念更新

    Efficient Incremental Belief Updates Using Weighted Virtual Observations

    [https://arxiv.org/abs/2402.06940](https://arxiv.org/abs/2402.06940)

    本文介绍了在贝叶斯统计模型中使用加权虚拟观测进行增量信念更新的算法解决方案，该方案通过构建一组加权观测来调节模型，实现与原始后验相同的推断结果。

    

    我们提出了一个算法解决了在贝叶斯统计模型中蒙特卡洛推断环境下的增量信念更新问题，该模型由概率编程表示。给定一个模型和样本逼近的后验概率，我们的解决方案构建了一组加权观测来调节模型，从而推断结果与原始后验相同。该问题出现在多层建模、增量推断和数据隐私约束下的推断等情况。首先，选择一组虚拟观测值，然后通过高效的计算优化过程找到观测权重，使得重建的后验与原始后验一致或近似。我们对一些教学示例和案例研究实施并应用了该解决方案，展示了我们方法的效率和鲁棒性。所提供的参考实现不依赖于概率编程语言或推断算法。

    We present an algorithmic solution to the problem of incremental belief updating in the context of Monte Carlo inference in Bayesian statistical models represented by probabilistic programs. Given a model and a sample-approximated posterior, our solution constructs a set of weighted observations to condition the model such that inference would result in the same posterior. This problem arises e.g. in multi-level modelling, incremental inference, inference in presence of privacy constraints. First, a set of virtual observations is selected, then, observation weights are found through a computationally efficient optimization procedure such that the reconstructed posterior coincides with or closely approximates the original posterior. We implement and apply the solution to a number of didactic examples and case studies, showing efficiency and robustness of our approach. The provided reference implementation is agnostic to the probabilistic programming language or the inference algorithm, 
    
[^50]: CochCeps-Augment：一种使用基于Cochlear Cepstrum的掩蔽的自监督对比学习方法用于语音情感识别

    CochCeps-Augment: A Novel Self-Supervised Contrastive Learning Using Cochlear Cepstrum-based Masking for Speech Emotion Recognition

    [https://arxiv.org/abs/2402.06923](https://arxiv.org/abs/2402.06923)

    提出了一种名为CochCeps-Augment的方法，利用基于Cochlear Cepstrum的掩蔽增强任务进行自监督对比学习，提高了语音情感识别的性能和噪声鲁棒性。

    

    自监督学习（SSL）用于情感内容的自动语音识别可以被噪声干扰严重降低，影响对语音的复杂时域和频谱信息结构进行建模的效率。最近，大规模语音数据集上的SSL以及新的音频特定的SSL代理任务（如时域和频域掩蔽）已经出现，相比于传统的源自图像增强领域的方法，取得了更好的性能。我们提出的创新在于基于成功的范例引入CochCeps-Augment，这是一种用于自监督对比学习语音表示的新型生物启发掩蔽增强任务。具体来说，我们利用了新引入的生物启发式Cochlear cepstrogram（CCGRAM）来推导输入语音的噪声鲁棒表示，然后通过自监督学习方案进一步优化。后者利用SimCLR生成CCGRAM的对比视图，通过对比学习来产生。

    Self-supervised learning (SSL) for automated speech recognition in terms of its emotional content, can be heavily degraded by the presence noise, affecting the efficiency of modeling the intricate temporal and spectral informative structures of speech. Recently, SSL on large speech datasets, as well as new audio-specific SSL proxy tasks, such as, temporal and frequency masking, have emerged, yielding superior performance compared to classic approaches drawn from the image augmentation domain. Our proposed contribution builds upon this successful paradigm by introducing CochCeps-Augment, a novel bio-inspired masking augmentation task for self-supervised contrastive learning of speech representations. Specifically, we utilize the newly introduced bio-inspired cochlear cepstrogram (CCGRAM) to derive noise robust representations of input speech, that are then further refined through a self-supervised learning scheme. The latter employs SimCLR to generate contrastive views of a CCGRAM throu
    
[^51]: Bilevel强化学习和RLHF的有原则的基于惩罚的方法

    Principled Penalty-based Methods for Bilevel Reinforcement Learning and RLHF

    [https://arxiv.org/abs/2402.06886](https://arxiv.org/abs/2402.06886)

    本文提出了一种基于惩罚的方法来解决Bilevel强化学习和RLHF问题，这是首个有原则的算法框架。通过理论分析和实验证明了算法的有效性。

    

    最近，Bilevel优化已被应用于许多机器学习任务中。然而，它们的应用仅限于监督学习设置，其中考虑了具有良性结构的静态目标函数。但是，激励设计、反向强化学习(RL)和来自人类反馈的RLHF等Bilevel问题通常被建模为超越简单静态目标结构的动态目标函数，这给使用现有Bilevel解决方案带来了重大挑战。为了解决这一新的Bilevel问题类别，我们通过惩罚形式引入了解决Bilevel RL问题的第一个原则性算法框架。我们通过理论研究问题的景观及其基于惩罚的（策略）梯度算法进行了验证。我们通过在Stackelberg马尔可夫博弈、来自人类反馈的RL和激励设计中进行模拟来证明我们算法的有效性。

    Bilevel optimization has been recently applied to many machine learning tasks. However, their applications have been restricted to the supervised learning setting, where static objective functions with benign structures are considered. But bilevel problems such as incentive design, inverse reinforcement learning (RL), and RL from human feedback (RLHF) are often modeled as dynamic objective functions that go beyond the simple static objective structures, which pose significant challenges of using existing bilevel solutions. To tackle this new class of bilevel problems, we introduce the first principled algorithmic framework for solving bilevel RL problems through the lens of penalty formulation. We provide theoretical studies of the problem landscape and its penalty-based (policy) gradient algorithms. We demonstrate the effectiveness of our algorithms via simulations in the Stackelberg Markov game, RL from human feedback and incentive design.
    
[^52]: 结构冗余的低秩逼近用于自监督学习

    Low-Rank Approximation of Structural Redundancy for Self-Supervised Learning

    [https://arxiv.org/abs/2402.06884](https://arxiv.org/abs/2402.06884)

    本文研究结构冗余的低秩逼近在自监督学习中的应用，提出了一个逼近冗余组件的新方法，并通过分析过量风险来支持理论。

    

    我们研究重构型自监督学习的数据生成机制，以揭示其有效性。在拥有无限量的标记样本的情况下，我们提供了完美线性逼近的充分必要条件。该条件揭示了一个保留标签类别Y的满秩组件，以及一个冗余组件。受到该条件的启发，我们提出通过低秩分解逼近冗余组件，并通过引入一个由分解秩s参数化的新量$\epsilon_s$来衡量逼近质量。我们将$\epsilon_s$整合到线性回归和岭回归设置下的过量风险分析中，后一种正则化方法用于处理学习特征的维度远大于下游任务的标记样本数n的情况。我们设计了三个简化实验，以比较不同设置下的自监督学习和监督学习，以支持我们的理论。

    We study the data-generating mechanism for reconstructive SSL to shed light on its effectiveness. With an infinite amount of labeled samples, we provide a sufficient and necessary condition for perfect linear approximation. The condition reveals a full-rank component that preserves the label classes of Y, along with a redundant component. Motivated by the condition, we propose to approximate the redundant component by a low-rank factorization and measure the approximation quality by introducing a new quantity $\epsilon_s$, parameterized by the rank of factorization s. We incorporate $\epsilon_s$ into the excess risk analysis under both linear regression and ridge regression settings, where the latter regularization approach is to handle scenarios when the dimension of the learned features is much larger than the number of labeled samples n for downstream tasks. We design three stylized experiments to compare SSL with supervised learning under different settings to support our theoretic
    
[^53]: 使用Nystr\"om近似的可扩展核逻辑回归：理论分析和离散选择建模应用

    Scalable Kernel Logistic Regression with Nystr\"om Approximation: Theoretical Analysis and Application to Discrete Choice Modelling

    [https://arxiv.org/abs/2402.06763](https://arxiv.org/abs/2402.06763)

    本文介绍了使用Nystr\"om近似方法解决大规模数据集上核逻辑回归的可扩展性问题。研究提供了理论分析并验证了不同的地标选择方法的性能。

    

    将基于核的机器学习技术应用于使用大规模数据集的离散选择建模时，经常面临存储需求和模型中涉及的大量参数的挑战。这种复杂性影响了大规模模型的高效训练。本文通过引入Nystr\"om近似方法解决了可扩展性问题，用于大规模数据集上的核逻辑回归。研究首先进行了理论分析，其中：i) 对KLR解的集合进行了描述，ii) 给出了使用Nystr\"om近似的KLR解的上界，并最后描述了专门用于Nystr\"om KLR的优化算法的特化。之后，对Nystr\"om KLR进行了计算验证。测试了四种地标选择方法，包括基本均匀采样、k-means采样策略和基于杠杆得分的两种非均匀方法。这些策略的性能进行了评估。

    The application of kernel-based Machine Learning (ML) techniques to discrete choice modelling using large datasets often faces challenges due to memory requirements and the considerable number of parameters involved in these models. This complexity hampers the efficient training of large-scale models. This paper addresses these problems of scalability by introducing the Nystr\"om approximation for Kernel Logistic Regression (KLR) on large datasets. The study begins by presenting a theoretical analysis in which: i) the set of KLR solutions is characterised, ii) an upper bound to the solution of KLR with Nystr\"om approximation is provided, and finally iii) a specialisation of the optimisation algorithms to Nystr\"om KLR is described. After this, the Nystr\"om KLR is computationally validated. Four landmark selection methods are tested, including basic uniform sampling, a k-means sampling strategy, and two non-uniform methods grounded in leverage scores. The performance of these strategi
    
[^54]: 使用小初始化的梯度下降算法在非正则化矩阵完成中的收敛性分析

    Convergence of Gradient Descent with Small Initialization for Unregularized Matrix Completion

    [https://arxiv.org/abs/2402.06756](https://arxiv.org/abs/2402.06756)

    本文分析了对称矩阵完成问题中梯度下降算法的收敛性。研究结果表明，在非正则化的情况下，使用小初始化的梯度下降算法可以收敛到真实的矩阵解，即使在过度参数化的情况下也成立。在过度参数化的情况下，几乎线性的收敛速度可以在获得足够多的观测条目后得到保证。

    

    本文研究对称矩阵完成的问题，目标是从仅观测到的部分条目中重构一个正半定矩阵X*，其等价于参数化矩阵UU^T，其中X*的秩为r。我们证明，使用小的初始化的基本梯度下降（GD）算法可以收敛到真实的矩阵X*，而不需要显式的正则化。这个收敛结果适用于过度参数化的场景，其中真实秩r是未知的，并且被一个搜索秩r'保守估计，且r' >> r。现有的结果要么需要显式的正则化，或者需要足够准确的初始点，或者需要准确知道真实秩r。在过度参数化的情况下，即r' >= r，我们证明，在获得Ω(dr^9)的观测条目后，GD算法以初始点∥U_0∥ <= ε几乎线性收敛到X*的ε-邻域中。

    We study the problem of symmetric matrix completion, where the goal is to reconstruct a positive semidefinite matrix $\rm{X}^\star \in \mathbb{R}^{d\times d}$ of rank-$r$, parameterized by $\rm{U}\rm{U}^{\top}$, from only a subset of its observed entries. We show that the vanilla gradient descent (GD) with small initialization provably converges to the ground truth $\rm{X}^\star$ without requiring any explicit regularization. This convergence result holds true even in the over-parameterized scenario, where the true rank $r$ is unknown and conservatively over-estimated by a search rank $r'\gg r$. The existing results for this problem either require explicit regularization, a sufficiently accurate initial point, or exact knowledge of the true rank $r$.   In the over-parameterized regime where $r'\geq r$, we show that, with $\widetilde\Omega(dr^9)$ observations, GD with an initial point $\|\rm{U}_0\| \leq \epsilon$ converges near-linearly to an $\epsilon$-neighborhood of $\rm{X}^\star$. C
    
[^55]: 采用分割引导扩散模型的解剖可控医学图像生成

    Anatomically-Controllable Medical Image Generation with Segmentation-Guided Diffusion Models

    [https://arxiv.org/abs/2402.05210](https://arxiv.org/abs/2402.05210)

    这篇论文提出了一种采用分割引导扩散模型的解剖可控医学图像生成方法，通过随机掩模消融训练算法实现对解剖约束的条件化，同时提高了网络对解剖真实性的学习能力。

    

    扩散模型已经实现了非常高质量的医学图像生成，可以通过为小型或不平衡的数据集提供补充，从而帮助减轻获取和注释新图像的费用，同时还可以应用于其他方面。然而，这些模型在生成图像时面临着全局解剖真实性的挑战。因此，我们提出了一种解剖可控的医学图像生成模型。我们的模型在每个采样步骤中遵循多类解剖分割掩模，并采用随机掩模消融训练算法，以实现对所选解剖约束的条件化，同时允许其他解剖区域的灵活性。这也改善了网络在完全无条件（无约束生成）情况下对解剖真实性的学习。通过对乳腺MRI和腹部/颈部到盆腔CT数据集的比较评估，证明了我们模型在解剖真实性和输入掩模保真度方面具有优越性。

    Diffusion models have enabled remarkably high-quality medical image generation, which can help mitigate the expenses of acquiring and annotating new images by supplementing small or imbalanced datasets, along with other applications. However, these are hampered by the challenge of enforcing global anatomical realism in generated images. To this end, we propose a diffusion model for anatomically-controlled medical image generation. Our model follows a multi-class anatomical segmentation mask at each sampling step and incorporates a \textit{random mask ablation} training algorithm, to enable conditioning on a selected combination of anatomical constraints while allowing flexibility in other anatomical areas. This also improves the network's learning of anatomical realism for the completely unconditional (unconstrained generation) case. Comparative evaluation on breast MRI and abdominal/neck-to-pelvis CT datasets demonstrates superior anatomical realism and input mask faithfulness over st
    
[^56]: 在六个简单的步骤中去噪扩散概率模型

    Denoising Diffusion Probabilistic Models in Six Simple Steps

    [https://arxiv.org/abs/2402.04384](https://arxiv.org/abs/2402.04384)

    本论文提供了一个简单、全面、干净且清晰的介绍去噪扩散概率模型（DDPM）的方法，强调了从连续时间极限的视角出发，以提供更好的理解和实际性能。

    

    去噪扩散概率模型（DDPM）是一类非常流行的深度生成模型，已成功应用于包括图像和视频生成、蛋白质和材料合成、天气预测和偏微分方程的神经替代等多个问题。尽管其普及度很高，但很难找到一个简单、全面、干净且清晰的DDPM介绍。研究论文中必要的简洁解释无法阐明制定DDPM所采取的不同设计步骤以及省略了步骤的理由以节省空间。此外，这些论述通常从变分下界的视角出发，这是不必要且可能有害的，因为它混淆了方法奏效的原因并暗示了实践中表现不佳的泛化性质。另一方面，采用连续时间极限的视角是美丽且普遍的，但是...

    Denoising Diffusion Probabilistic Models (DDPMs) are a very popular class of deep generative model that have been successfully applied to a diverse range of problems including image and video generation, protein and material synthesis, weather forecasting, and neural surrogates of partial differential equations. Despite their ubiquity it is hard to find an introduction to DDPMs which is simple, comprehensive, clean and clear. The compact explanations necessary in research papers are not able to elucidate all of the different design steps taken to formulate the DDPM and the rationale of the steps that are presented is often omitted to save space. Moreover, the expositions are typically presented from the variational lower bound perspective which is unnecessary and arguably harmful as it obfuscates why the method is working and suggests generalisations that do not perform well in practice. On the other hand, perspectives that take the continuous time-limit are beautiful and general, but 
    
[^57]: 使用Metropolis-adjusted Mirror Langevin算法从约束空间中快速采样

    Fast sampling from constrained spaces using the Metropolis-adjusted Mirror Langevin algorithm

    [https://arxiv.org/abs/2312.08823](https://arxiv.org/abs/2312.08823)

    该论文提出了一种名为Metropolis-adjusted Mirror Langevin算法的方法，用于从约束空间中进行快速采样。这种算法是对Mirror Langevin算法的改进，通过添加接受-拒绝过滤器来消除渐近偏差，并具有指数优化依赖。

    

    我们提出了一种新的方法，称为Metropolis-adjusted Mirror Langevin算法，用于从其支持是紧凸集的分布中进行近似采样。该算法在Mirror Langevin算法（Zhang et al., 2020）的单步马尔科夫链中添加了一个接受-拒绝过滤器，Mirror Langevin算法是Mirror Langevin动力学的基本离散化。由于包含了这个过滤器，我们的方法相对于目标是无偏的，而已知的Mirror Langevin算法等Mirror Langevin动力学的离散化具有渐近偏差。对于该算法，我们还给出了混合到一个相对平滑、凸性好且与自共轭镜像函数相关的约束分布所需迭代次数的上界。由于包含Metropolis-Hastings过滤器导致的马尔科夫链是可逆的，我们得到了对误差的指数优化依赖。

    We propose a new method called the Metropolis-adjusted Mirror Langevin algorithm for approximate sampling from distributions whose support is a compact and convex set. This algorithm adds an accept-reject filter to the Markov chain induced by a single step of the Mirror Langevin algorithm (Zhang et al., 2020), which is a basic discretisation of the Mirror Langevin dynamics. Due to the inclusion of this filter, our method is unbiased relative to the target, while known discretisations of the Mirror Langevin dynamics including the Mirror Langevin algorithm have an asymptotic bias. For this algorithm, we also give upper bounds for the number of iterations taken to mix to a constrained distribution whose potential is relatively smooth, convex, and Lipschitz continuous with respect to a self-concordant mirror function. As a consequence of the reversibility of the Markov chain induced by the inclusion of the Metropolis-Hastings filter, we obtain an exponentially better dependence on the erro
    
[^58]: 逆向强化学习比标准强化学习更困难吗？一个理论的观点

    Is Inverse Reinforcement Learning Harder than Standard Reinforcement Learning? A Theoretical Perspective

    [https://arxiv.org/abs/2312.00054](https://arxiv.org/abs/2312.00054)

    逆向强化学习是从专家策略示范中学习奖励函数的问题，本文提出了在标准离线和在线设置下用多项式样本和运行时间进行高效逆向强化学习的结果线索，并提供了几乎最优的样本复杂性的下界。

    

    逆向强化学习（IRL）是从专家策略的示范中学习奖励函数的问题，在开发智能系统中起着关键作用。尽管在应用中广泛使用，但与标准强化学习相比，IRL的理论理解存在独特的挑战，且发展相对较少。本文首次提出了使用多项式样本和运行时间在标准离线和在线设置下进行高效IRL的结果线索。我们的算法和分析巧妙地采用了离线强化学习中常用的悲观原则，并在比现有工作中考虑的更强的度量标准下实现了IRL的保证。我们提供了下界，表明我们的样本复杂性几乎是最优的。

    Inverse Reinforcement Learning (IRL) -- the problem of learning reward functions from demonstrations of an \emph{expert policy} -- plays a critical role in developing intelligent systems. While widely used in applications, theoretical understandings of IRL present unique challenges and remain less developed compared with standard RL. For example, it remains open how to do IRL efficiently in standard \emph{offline} settings with pre-collected data, where states are obtained from a \emph{behavior policy} (which could be the expert policy itself), and actions are sampled from the expert policy.   This paper provides the first line of results for efficient IRL in vanilla offline and online settings using polynomial samples and runtime. Our algorithms and analyses seamlessly adapt the pessimism principle commonly used in offline RL, and achieve IRL guarantees in stronger metrics than considered in existing work. We provide lower bounds showing that our sample complexities are nearly optimal
    
[^59]: 高效强化学习在部分可观察性下的应用

    Efficient Reinforcement Learning from Partial Observability

    [https://arxiv.org/abs/2311.12244](https://arxiv.org/abs/2311.12244)

    该论文提出了一种基于表示的方法，用于从部分观测中进行有效的强化学习。该方法能够处理部分可观测性带来的计算和统计挑战，并在各种基准测试中展现出优于先进算法的性能。

    

    在大多数实际应用中，状态信息只能部分观测到，这破坏了马尔科夫决策过程的假设，导致将观测与状态相混淆的算法表现不佳。而部分可观测马尔科夫决策过程（POMDP）提供了一个允许在学习、探索和规划中考虑部分可观测性的通用框架，但也带来了显著的计算和统计挑战。为解决这些困难，我们提出了一个基于表示的视角，提供了一个统一的框架和可行的算法方法，用于从部分观测中进行实际的强化学习。我们提供了理论分析来证明所提出算法的统计效率，并经验性地证明了在各种基准测试中，所提出的算法在部分观测下能够超越最先进性能，推动了可靠的强化学习。

    In most real-world reinforcement learning applications, state information is only partially observable, which breaks the Markov decision process assumption and leads to inferior performance for algorithms that conflate observations with state. Partially Observable Markov Decision Processes (POMDPs), on the other hand, provide a general framework that allows for partial observability to be accounted for in learning, exploration and planning, but presents significant computational and statistical challenges. To address these difficulties, we develop a representation-based perspective that leads to a coherent framework and tractable algorithmic approach for practical reinforcement learning from partial observations. We provide a theoretical analysis for justifying the statistical efficiency of the proposed algorithm, and also empirically demonstrate the proposed algorithm can surpass state-of-the-art performance with partial observations across various benchmarks, advancing reliable reinf
    
[^60]: 用于系外行星凌星和H0推断的核、均值和噪声边缘化高斯过程

    Kernel-, mean- and noise-marginalised Gaussian processes for exoplanet transits and $H_0$ inference

    [https://arxiv.org/abs/2311.04153](https://arxiv.org/abs/2311.04153)

    该论文提出了一种基于贝叶斯方法的核、均值和噪声边缘化高斯过程，用于系外行星凌星和H0推断。通过核选择和核超参数的边缘化以及贝叶斯模型比较，可以实现核选择和推断。

    

    使用完全贝叶斯方法，将高斯过程回归扩展为包括核选择和核超参数的边缘化。此外，通过证据进行贝叶斯模型比较，可以直接比较核选择。通过在高维空间中嵌入离散核选择和超参数，使用嵌套抽样进行联合后验计算。在系外行星凌星光变曲线模拟的合成数据上探索了核恢复和均值函数推断。随后，将该方法扩展到均值函数和噪声模型的边缘化，并应用于从实际的红移相关哈勃参数测量中推断当今哈勃参数H0，这些参数来自于宇宙学模型独立的宇宙计时器和ΛCDM依赖的声学谐振。

    Using a fully Bayesian approach, Gaussian Process regression is extended to include marginalisation over the kernel choice and kernel hyperparameters. In addition, Bayesian model comparison via the evidence enables direct kernel comparison. The calculation of the joint posterior was implemented with a transdimensional sampler which simultaneously samples over the discrete kernel choice and their hyperparameters by embedding these in a higher-dimensional space, from which samples are taken using nested sampling. Kernel recovery and mean function inference were explored on synthetic data from exoplanet transit light curve simulations. Subsequently, the method was extended to marginalisation over mean functions and noise models and applied to the inference of the present-day Hubble parameter, $H_0$, from real measurements of the Hubble parameter as a function of redshift, derived from the cosmologically model-independent cosmic chronometer and $\Lambda$CDM-dependent baryon acoustic oscill
    
[^61]: 通过图的双连通性重新思考GNN的表达能力

    Rethinking the Expressive Power of GNNs via Graph Biconnectivity

    [https://arxiv.org/abs/2301.09505](https://arxiv.org/abs/2301.09505)

    本文从根本上不同的角度重新思考了图神经网络（GNN）的表达能力，通过引入一类新的表达度量方法，即图的双连通性，并强调了它们在理论和实践中的重要性。令人惊讶的是，在对以前的GNN架构进行彻底审查后，发现大多数架构都没有对这些度量具有表达能力。唯一的例外是ESAN框架。

    

    设计具有表达能力的图神经网络(GNNs)是学习图结构数据的一个核心主题。尽管已经提出了很多方法来改进GNNs在Weisfeiler-Lehman (WL)测试方面的表现，但是普遍还存在对它们能够系统和可证明地获得的额外能力的缺乏深入了解。本文从根本上不同的角度来研究GNNs的表达能力，超越了WL测试。具体地，我们引入了一类新的通过图的双连通性的表达度量，并强调它们在理论和实践中的重要性。由于双连通性可以使用简单的算法进行计算，并且具有线性的计算成本，因此很自然地可以期望流行的GNNs也可以很容易地进行学习。然而，经过对以前的GNN架构的彻底审查，我们惊讶地发现大多数架构对于任何这些度量都不具有表达能力。唯一的例外是ESAN框架，对于该框架，我们给出了一个理论的解释。

    Designing expressive Graph Neural Networks (GNNs) is a central topic in learning graph-structured data. While numerous approaches have been proposed to improve GNNs in terms of the Weisfeiler-Lehman (WL) test, generally there is still a lack of deep understanding of what additional power they can systematically and provably gain. In this paper, we take a fundamentally different perspective to study the expressive power of GNNs beyond the WL test. Specifically, we introduce a novel class of expressivity metrics via graph biconnectivity and highlight their importance in both theory and practice. As biconnectivity can be easily calculated using simple algorithms that have linear computational costs, it is natural to expect that popular GNNs can learn it easily as well. However, after a thorough review of prior GNN architectures, we surprisingly find that most of them are not expressive for any of these metrics. The only exception is the ESAN framework, for which we give a theoretical just
    
[^62]: 动量外移梯度何时能达到最佳？基于多项式的分析

    When is Momentum Extragradient Optimal? A Polynomial-Based Analysis

    [https://arxiv.org/abs/2211.04659](https://arxiv.org/abs/2211.04659)

    本论文通过多项式分析，对动量外移梯度方法在不同情景下的加速收敛进行研究，包括特征值存在于实轴、位于实轴上的共轭复数或仅存在共轭复数的情况。同时，我们还得出了实现最快收敛的超参数。

    

    外移梯度方法由于其在可微分博弈中的稳健收敛性而受到青睐。与单目标优化不同，博弈动力学涉及到复杂的相互作用，这种相互作用通过博弈向量场的雅可比矩阵的特征值散布在复平面上。这种复杂性会导致简单的梯度方法发散，即使对于双线性博弈也是如此，而外移梯度方法却能实现收敛。在最近证明的基础上，即动量外移梯度方法在双线性博弈中实现加速收敛\citep{azizian2020accelerating}，我们使用基于多项式的分析来确定该方法出现进一步加速收敛的三种不同情景。这些情景包括特征值存在于（正）实轴上、位于实轴上的共轭复数以及仅存在共轭复数的情况。此外，我们还推导出每个情景的超参数，以实现最快的收敛。

    The extragradient method has gained popularity due to its robust convergence properties for differentiable games. Unlike single-objective optimization, game dynamics involve complex interactions reflected by the eigenvalues of the game vector field's Jacobian scattered across the complex plane. This complexity can cause the simple gradient method to diverge, even for bilinear games, while the extragradient method achieves convergence. Building on the recently proven accelerated convergence of the momentum extragradient method for bilinear games \citep{azizian2020accelerating}, we use a polynomial-based analysis to identify three distinct scenarios where this method exhibits further accelerated convergence. These scenarios encompass situations where the eigenvalues reside on the (positive) real line, lie on the real line alongside complex conjugates, or exist solely as complex conjugates. Furthermore, we derive the hyperparameters for each scenario that achieve the fastest convergence r
    
[^63]: 深度学习的动态潜变量分离

    Dynamic Latent Separation for Deep Learning

    [https://arxiv.org/abs/2210.03728](https://arxiv.org/abs/2210.03728)

    本研究提出了动态潜变量分离的方法，可以在复杂数据中学习表达性强的潜变量，提升输出的多样性。该方法受原子物理学启发，通过学习每个数据样本的结构来解释各个子组件的重要性。实验证明该方法在不同分类和生成问题中提升了模型的性能。

    

    机器学习中的一个核心问题是以灵活和可解释的方式学习用于复杂数据模型预测的表达性潜变量，这些数据包含多个子组件。我们开发了一种方法，改进了表达性，提供了部分解释，并且不限于特定的应用。关键思想是在潜空间中动态地分离数据样本，从而增强输出的多样性。我们的动态潜变量分离方法受到原子物理学的启发，依赖于每个数据样本共同学习的结构，这也揭示出了每个子组件在区分数据样本中的重要性。这种方法，原子建模，不需要对潜空间进行监督，并且允许我们学习额外的部分可解释表示，除了模型的原始目标。实验证明，该算法还提高了各种分类和生成问题中小到大规模模型的性能。

    A core problem in machine learning is to learn expressive latent variables for model prediction on complex data that involves multiple sub-components in a flexible and interpretable fashion. Here, we develop an approach that improves expressiveness, provides partial interpretation, and is not restricted to specific applications. The key idea is to dynamically distance data samples in the latent space and thus enhance the output diversity. Our dynamic latent separation method, inspired by atomic physics, relies on the jointly learned structures of each data sample, which also reveal the importance of each sub-component for distinguishing data samples. This approach, atom modeling, requires no supervision of the latent space and allows us to learn extra partially interpretable representations besides the original goal of a model. We empirically demonstrate that the algorithm also enhances the performance of small to larger-scale models in various classification and generation problems.
    
[^64]: 差分隐私图学习的敏感性有界个性化PageRank算法

    Differentially Private Graph Learning via Sensitivity-Bounded Personalized PageRank

    [https://arxiv.org/abs/2207.06944](https://arxiv.org/abs/2207.06944)

    本论文提出了一种敏感性有界的个性化PageRank算法，能够保护用户隐私。该算法在保持准确性的同时，实现了差分隐私图学习的几种工具。

    

    个性化PageRank(PPR)是一种基本工具，用于无监督学习图表示，如节点排序、标注和图嵌入。然而，随着数据隐私成为最近最重要的关注点之一，现有的PPR算法并未设计用于保护用户隐私。PPR对输入图的边非常敏感：仅差一个边的差异可能会导致PPR向量发生巨大改变，从而可能泄漏用户私密数据。在这篇论文中，我们提出了一种算法，该算法输出近似PPR，并对输入边具有可证明的敏感性边界。此外，我们证明了当输入图具有大度数时，我们的算法达到与非私密算法相似的准确性。我们敏感性有界PPR直接意味着图学习的几种私密算法，如差分隐私(DP)PPR排序、DP节点分类和DP节点嵌入。为了补充我们的理论分析，我们还通过实验证明了算法的实际性能。

    Personalized PageRank (PPR) is a fundamental tool in unsupervised learning of graph representations such as node ranking, labeling, and graph embedding. However, while data privacy is one of the most important recent concerns, existing PPR algorithms are not designed to protect user privacy. PPR is highly sensitive to the input graph edges: the difference of only one edge may cause a big change in the PPR vector, potentially leaking private user data.   In this work, we propose an algorithm which outputs an approximate PPR and has provably bounded sensitivity to input edges. In addition, we prove that our algorithm achieves similar accuracy to non-private algorithms when the input graph has large degrees. Our sensitivity-bounded PPR directly implies private algorithms for several tools of graph learning, such as, differentially private (DP) PPR ranking, DP node classification, and DP node embedding. To complement our theoretical analysis, we also empirically verify the practical perfor
    
[^65]: 变量选择的计算高效高维贝叶斯优化方法

    Computationally Efficient High-Dimensional Bayesian Optimization via Variable Selection

    [https://arxiv.org/abs/2109.09264](https://arxiv.org/abs/2109.09264)

    本论文提出了一种变量选择的计算高效高维贝叶斯优化方法，能够自动学习子空间来优化高维域函数，同时减少了传统方法中的耗时问题，并在实验证明了方法的有效性。

    

    贝叶斯优化（BO）是一种用于全局优化黑盒函数的方法。虽然BO已成功应用于许多场景，但是开发能够适用于高维域函数的有效BO算法仍然是一个挑战。通过普通的BO优化此类函数非常耗时。基于将高维空间嵌入到低维空间的思想的高维BO的替代策略对嵌入维度的选择非常敏感，需要预先指定。我们开发了一种新的计算高效的高维BO方法，利用了变量选择。我们的方法能够自动学习轴对齐的子空间，即包含选定变量的空间，而无需任何预先指定的超参数。我们从理论上分析了算法的计算复杂性并得出了遗憾界限。我们在几个合成和真实数据上实验证明了我们方法的有效性。

    Bayesian Optimization (BO) is a method for globally optimizing black-box functions. While BO has been successfully applied to many scenarios, developing effective BO algorithms that scale to functions with high-dimensional domains is still a challenge. Optimizing such functions by vanilla BO is extremely time-consuming. Alternative strategies for high-dimensional BO that are based on the idea of embedding the high-dimensional space to the one with low dimension are sensitive to the choice of the embedding dimension, which needs to be pre-specified. We develop a new computationally efficient high-dimensional BO method that exploits variable selection. Our method is able to automatically learn axis-aligned sub-spaces, i.e. spaces containing selected variables, without the demand of any pre-specified hyperparameters. We theoretically analyze the computational complexity of our algorithm and derive the regret bound. We empirically show the efficacy of our method on several synthetic and re
    
[^66]: 机器协作

    Machine Collaboration

    [https://arxiv.org/abs/2105.02569](https://arxiv.org/abs/2105.02569)

    本文提出了一种新的监督学习集成框架——机器协作（MaC），通过循环和交互的学习方式，使基础机器能够循环传递信息并相应地更新结构和参数。实验证明，MaC在大多数情况下表现优于其他先进方法。

    

    我们提出了一种新的监督学习集成框架，称为机器协作（MaC），利用一组基础机器进行预测任务。与并行且独立的bagging/stacking框架和顺序且自上而下的boosting框架不同，MaC是一种循环和交互学习框架。循环和交互特性帮助基础机器循环传递信息并相应地更新其结构和参数。对于从MaC得出的估计器的风险界的理论结果表明，循环和交互特性可以帮助MaC通过简洁的集成减少风险。我们在模拟数据和119个基准真实数据集上进行了大量实验证明，在大多数情况下，MaC的性能显著优于包括分类回归树、神经网络、堆叠和提升在内的其他几种最先进的方法。

    We propose a new ensemble framework for supervised learning, called machine collaboration (MaC), using a collection of base machines for prediction tasks. Unlike bagging/stacking (a parallel & independent framework) and boosting (a sequential & top-down framework), MaC is a type of circular & interactive learning framework. The circular & interactive feature helps the base machines to transfer information circularly and update their structures and parameters accordingly. The theoretical result on the risk bound of the estimator from MaC reveals that the circular & interactive feature can help MaC reduce risk via a parsimonious ensemble. We conduct extensive experiments on MaC using both simulated data and 119 benchmark real datasets. The results demonstrate that in most cases, MaC performs significantly better than several other state-of-the-art methods, including classification and regression trees, neural networks, stacking, and boosting.
    
[^67]: 稀疏NMF与典型正则化：计算和鲁棒性性质

    Sparse NMF with Archetypal Regularization: Computational and Robustness Properties

    [https://arxiv.org/abs/2104.03527](https://arxiv.org/abs/2104.03527)

    本文研究了使用典型正则化的稀疏非负矩阵分解问题，提出了强鲁棒性和弱鲁棒性的概念，并给出了理论保证和数值实验来加强这些概念的洞察力。

    

    我们考虑使用典型正则化的稀疏非负矩阵分解（NMF）问题。目标是将一组数据点表示为少数非负稀疏因子的非负线性组合，这些因子具有吸引人的几何特性，来自于使用典型正则化。我们将在Javadi和Montanari（2019）中研究的鲁棒性概念（无稀疏性）推广为（a）强鲁棒性，即每个估计的典型都接近真实的典型，以及（b）弱鲁棒性，即至少存在一个恢复的典型接近真实的典型。我们的理论结果对于基础数据的假设较为简化，并适用于基于不需要稀疏性的典型的情况。我们提出了新的算法来优化我们的问题。

    We consider the problem of sparse nonnegative matrix factorization (NMF) using archetypal regularization. The goal is to represent a collection of data points as nonnegative linear combinations of a few nonnegative sparse factors with appealing geometric properties, arising from the use of archetypal regularization. We generalize the notion of robustness studied in Javadi and Montanari (2019) (without sparsity) to the notions of (a) strong robustness that implies each estimated archetype is close to the underlying archetypes and (b) weak robustness that implies there exists at least one recovered archetype that is close to the underlying archetypes. Our theoretical results on robustness guarantees hold under minimal assumptions on the underlying data, and applies to settings where the underlying archetypes need not be sparse. We present theoretical results and illustrative examples to strengthen the insights underlying the notions of robustness. We propose new algorithms for our optimi
    
[^68]: 可扩展的子二次时间网络重建

    Scalable network reconstruction in subquadratic time. (arXiv:2401.01404v1 [cs.DS])

    [http://arxiv.org/abs/2401.01404](http://arxiv.org/abs/2401.01404)

    这篇论文提出了一个可扩展的网络重建算法，能够在次二次时间内实现结果，通过随机的二阶邻居搜索产生最佳的边候选。

    

    网络重建是指在只有关于条件偶联的观测数据，例如时间序列或图模型的独立样本的情况下，确定N个节点之间未观测到的成对耦合。针对这个问题提出的算法的可扩展性的主要障碍是似乎无法避免的二次复杂度O(N^2)，即要考虑每种可能的成对耦合至少一次，尽管大多数感兴趣的网络都是稀疏的，非零耦合的数量只有O(N)。在这里，我们提出了一个适用于广泛重建问题的通用算法，其在子二次时间内实现结果，其数据相关复杂度宽松上界为O(N^(3/2)logN)，但具有更典型的对数线性复杂度O(Nlog^2 N)。我们的算法依赖于一个随机的二阶邻居搜索，产生了最佳的边候选。

    Network reconstruction consists in determining the unobserved pairwise couplings between $N$ nodes given only observational data on the resulting behavior that is conditioned on those couplings -- typically a time-series or independent samples from a graphical model. A major obstacle to the scalability of algorithms proposed for this problem is a seemingly unavoidable quadratic complexity of $O(N^2)$, corresponding to the requirement of each possible pairwise coupling being contemplated at least once, despite the fact that most networks of interest are sparse, with a number of non-zero couplings that is only $O(N)$. Here we present a general algorithm applicable to a broad range of reconstruction problems that achieves its result in subquadratic time, with a data-dependent complexity loosely upper bounded by $O(N^{3/2}\log N)$, but with a more typical log-linear complexity of $O(N\log^2N)$. Our algorithm relies on a stochastic second neighbor search that produces the best edge candidat
    
[^69]: 鲁棒的角度同步问题的有向图神经网络解决方案

    Robust Angular Synchronization via Directed Graph Neural Networks. (arXiv:2310.05842v1 [cs.LG])

    [http://arxiv.org/abs/2310.05842](http://arxiv.org/abs/2310.05842)

    本论文提出了一个名为GNNSync的基于有向图神经网络的鲁棒角度同步解决方案，解决了角度同步问题在高噪声环境下的挑战，并提出了新的损失函数以更好地编码同步约束。

    

    角度同步问题旨在通过$m$个偏移量$\theta_i-\theta_j \;\mbox{mod} \; 2\pi$的噪声测量准确估计（最多一个常数相位偏移）一组未知角度$\theta_1, \dots, \theta_n\in[0, 2\pi)$. 应用包括传感器网络定位、相位恢复和分布式时钟同步。该问题的异构扩展（称为$k$-同步）是同时估计$k$组角度，给定每个组的未知组分配的噪声观察值。现有的角度同步方法在高噪声环境下通常表现不佳，而这在应用中很常见。在本文中，我们利用神经网络解决角度同步问题及其异构扩展，提出了GNNSync，这是一个理论支撑的端到端可训练框架，使用有向图神经网络。此外，我们设计了新的损失函数来编码角度同步的约束。

    The angular synchronization problem aims to accurately estimate (up to a constant additive phase) a set of unknown angles $\theta_1, \dots, \theta_n\in[0, 2\pi)$ from $m$ noisy measurements of their offsets $\theta_i-\theta_j \;\mbox{mod} \; 2\pi.$ Applications include, for example, sensor network localization, phase retrieval, and distributed clock synchronization. An extension of the problem to the heterogeneous setting (dubbed $k$-synchronization) is to estimate $k$ groups of angles simultaneously, given noisy observations (with unknown group assignment) from each group. Existing methods for angular synchronization usually perform poorly in high-noise regimes, which are common in applications. In this paper, we leverage neural networks for the angular synchronization problem, and its heterogeneous extension, by proposing GNNSync, a theoretically-grounded end-to-end trainable framework using directed graph neural networks. In addition, new loss functions are devised to encode synchro
    
[^70]: 基于贝叶斯深度学习的宇宙尺度中的修正引力研究

    Bayesian deep learning for cosmic volumes with modified gravity. (arXiv:2309.00612v1 [astro-ph.CO])

    [http://arxiv.org/abs/2309.00612](http://arxiv.org/abs/2309.00612)

    该研究利用贝叶斯深度学习的方法，从修正引力模拟中提取宇宙学参数，并对不确定性进行了评估。

    

    新一代的星系调查将提供前所未有的数据，使我们能够在宇宙尺度上测试引力。对大尺度结构的健壮宇宙学分析需要利用编码在宇宙网中的非线性信息。机器学习技术提供了这样的工具，然而却不能提供先验的不确定性评估。本研究旨在通过具有不确定性估计的深度神经网络从修正引力（MG）模拟中提取宇宙学参数。我们使用贝叶斯神经网络（BNNs）实现了一个丰富的近似后验分布，分别考虑了一个带有单一贝叶斯最后一层（BLL）的情况，和一个在所有层面上都具有贝叶斯层（FullB）的情况。我们使用实空间密度场和一套2000个仅包含暗物质粒子网格$ N $-体模拟的功率谱对这两个BNN进行训练，这些模拟包括基于MG-PICOLA的修正引力模型，覆盖了边长为256 $h^{-1}$ Mpc的立方体体积，其中包含128$。

    The new generation of galaxy surveys will provide unprecedented data allowing us to test gravity at cosmological scales. A robust cosmological analysis of the large-scale structure demands exploiting the nonlinear information encoded in the cosmic web. Machine Learning techniques provide such tools, however, do not provide a priori assessment of uncertainties. This study aims at extracting cosmological parameters from modified gravity (MG) simulations through deep neural networks endowed with uncertainty estimations. We implement Bayesian neural networks (BNNs) with an enriched approximate posterior distribution considering two cases: one with a single Bayesian last layer (BLL), and another one with Bayesian layers at all levels (FullB). We train both BNNs with real-space density fields and power-spectra from a suite of 2000 dark matter only particle mesh $N$-body simulations including modified gravity models relying on MG-PICOLA covering 256 $h^{-1}$ Mpc side cubical volumes with 128$
    
[^71]: 自适应近端梯度方法的凸优化

    Adaptive Proximal Gradient Method for Convex Optimization. (arXiv:2308.02261v1 [math.OC])

    [http://arxiv.org/abs/2308.02261](http://arxiv.org/abs/2308.02261)

    本文提出了自适应版本的梯度下降（GD）和近端梯度方法（ProxGD），通过利用局部曲率信息完全自适应。所提出的方法具有收敛性，且允许使用更大的步长。

    

    在本文中，我们探讨了凸优化中的两个基本一阶算法，即梯度下降（GD）和近端梯度方法（ProxGD）。我们的重点是通过利用平滑函数的局部曲率信息，使这些算法完全自适应。我们提出了基于观察到的梯度差异的自适应版本的GD和ProxGD，因此不会增加计算成本。此外，我们在仅假设梯度的局部Lipschitz性的情况下，证明了我们方法的收敛性。另外，所提出的版本允许使用比[MM20]最初建议的更大的步长。

    In this paper, we explore two fundamental first-order algorithms in convex optimization, namely, gradient descent (GD) and proximal gradient method (ProxGD). Our focus is on making these algorithms entirely adaptive by leveraging local curvature information of smooth functions. We propose adaptive versions of GD and ProxGD that are based on observed gradient differences and, thus, have no added computational costs. Moreover, we prove convergence of our methods assuming only local Lipschitzness of the gradient. In addition, the proposed versions allow for even larger stepsizes than those initially suggested in [MM20].
    
[^72]: 理解量子机器学习需要重新思考泛化问题

    Understanding quantum machine learning also requires rethinking generalization. (arXiv:2306.13461v1 [quant-ph])

    [http://arxiv.org/abs/2306.13461](http://arxiv.org/abs/2306.13461)

    本文通过实验认为，传统方法无法解释量子机器学习模型在只使用少量数据训练的情况下表现出成功的泛化性能，该模型可以准确拟合随机状态及随机标记的训练数据，这种记忆随机数据的能力违反了当前小泛化误差的概念，我们通过理论构建补充实证结果，表明量子神经网络可将任意标记拟合到量子状态上，暗示了它们的记忆能力，这些结果排除了单单基于经典复杂度度量的所有可能保证。

    

    量子机器学习模型在只用少量数据训练的情况下也能表现出成功的泛化性能。本文通过系统的随机化实验，展示传统的理解泛化的方法无法解释这些量子模型的行为。我们的实验揭示了最先进的量子神经网络能够准确地拟合随机状态和随机训练数据的标记。这种记忆随机数据的能力违反了当前小泛化误差的概念，使得建立在VC维、Rademacher复杂度和所有均匀相关性度量基础上的方法有些棘手。我们还通过理论构建补充了我们的实证结果，表明量子神经网络能够将任意标记拟合到量子状态上，暗示了它们的记忆能力。我们的结果并不排除只用少量训练数据就能获得良好泛化的可能性，但是排除了单单基于经典复杂度度量的所有可能保证。

    Quantum machine learning models have shown successful generalization performance even when trained with few data. In this work, through systematic randomization experiments, we show that traditional approaches to understanding generalization fail to explain the behavior of such quantum models. Our experiments reveal that state-of-the-art quantum neural networks accurately fit random states and random labeling of training data. This ability to memorize random data defies current notions of small generalization error, problematizing approaches that build on complexity measures such as the VC dimension, the Rademacher complexity, and all their uniform relatives. We complement our empirical results with a theoretical construction showing that quantum neural networks can fit arbitrary labels to quantum states, hinting at their memorization ability. Our results do not preclude the possibility of good generalization with few training data but rather rule out any possible guarantees based only
    
[^73]: 带有沉重尾部SGD训练的过参数化神经网络的隐式可压缩性

    Implicit Compressibility of Overparametrized Neural Networks Trained with Heavy-Tailed SGD. (arXiv:2306.08125v1 [stat.ML])

    [http://arxiv.org/abs/2306.08125](http://arxiv.org/abs/2306.08125)

    本研究提出了一种简单的SGD修改方法，使训练出的神经网络输出可被证明为可压缩，而不需要任何非平凡假设。

    

    由于减少计算需求和压缩与泛化误差之间的显式关系，神经网络压缩成为越来越重要的研究对象。最近的研究表明，随机梯度下降(SGD)的超参数选择可以影响学习参数向量的压缩性。虽然这些结果揭示了训练动态对压缩性的影响，但是它们依赖于不可验证的假设，由于隐含性质，得出的理论并没有提供实用的指导方针。在本研究中，我们提出了一种简单的SGD修改方法，使得算法的输出能够被证明是可压缩的，而不需要任何非平凡假设。我们考虑了一个使用SGD训练的单隐藏层神经网络，并在每次迭代中注入附加的沉重尾部噪声。

    Neural network compression has been an increasingly important subject, due to its practical implications in terms of reducing the computational requirements and its theoretical implications, as there is an explicit connection between compressibility and the generalization error. Recent studies have shown that the choice of the hyperparameters of stochastic gradient descent (SGD) can have an effect on the compressibility of the learned parameter vector. Even though these results have shed some light on the role of the training dynamics over compressibility, they relied on unverifiable assumptions and the resulting theory does not provide a practical guideline due to its implicitness. In this study, we propose a simple modification for SGD, such that the outputs of the algorithm will be provably compressible without making any nontrivial assumptions. We consider a one-hidden-layer neural network trained with SGD and we inject additive heavy-tailed noise to the iterates at each iteration.
    
[^74]: MESSY估计：基于最大熵的随机和符号密度估计

    MESSY Estimation: Maximum-Entropy based Stochastic and Symbolic densitY Estimation. (arXiv:2306.04120v1 [cs.LG])

    [http://arxiv.org/abs/2306.04120](http://arxiv.org/abs/2306.04120)

    MESSY估计方法是一种基于最大熵的随机和符号密度估计方法，通过构建基于梯度的漂移扩散过程来高效地找到最大熵分布的参数，支持高维问题，并具有优于现有最新方法的有效性和普适性。

    

    我们引入了基于最大熵的随机和符号密度估计方法MESSY。所提出的方法使用梯度流的矩将概率密度函数从样本中恢复为符号表达式，并将ansatz作为驱动力。特别地，我们构建了一个基于梯度的漂移扩散过程，将未知分布函数的样本与猜测的符号表达式相连。然后，我们展示出当猜测分布具有最大熵形式时，可以通过使用提供的样本的矩构建的线性方程组高效地找到该分布的参数。此外，我们使用符号回归来探索平滑函数的空间，并找到导致最大熵泛函指数的最优基函数，以获得良好条件。该方法在随机搜索的每次迭代中的成本与样本数量呈线性关系，与变量数量呈二次关系，使其可扩展到高维问题。数值实验显示出所提出方法的有效性和普适性，与现有的最新方法相比。

    We introduce MESSY estimation, a Maximum-Entropy based Stochastic and Symbolic densitY estimation method. The proposed approach recovers probability density functions symbolically from samples using moments of a Gradient flow in which the ansatz serves as the driving force. In particular, we construct a gradient-based drift-diffusion process that connects samples of the unknown distribution function to a guess symbolic expression. We then show that when the guess distribution has the maximum entropy form, the parameters of this distribution can be found efficiently by solving a linear system of equations constructed using the moments of the provided samples. Furthermore, we use Symbolic regression to explore the space of smooth functions and find optimal basis functions for the exponent of the maximum entropy functional leading to good conditioning. The cost of the proposed method in each iteration of the random search is linear with the number of samples and quadratic with the number 
    
[^75]: 初始猜测偏差：未经过训练的神经网络倾向于某些类别

    Initial Guessing Bias: How Untrained Networks Favor Some Classes. (arXiv:2306.00809v1 [cs.LG])

    [http://arxiv.org/abs/2306.00809](http://arxiv.org/abs/2306.00809)

    本文提出了“初始猜测偏差”现象，即在未经过训练的神经网络中，由于架构选择的影响，模型往往会将所有预测指向同一个类别。该现象对架构选择和初始化有实际指导意义，并具有理论后果，例如节点置换对称性的崩溃和深度带来的非平凡差异。

    

    神经网络的初始状态在调节后续的训练过程中扮演重要角色。在分类问题的背景下，我们提供了理论分析，证明神经网络的结构可以在训练之前，甚至在不存在显式偏差的情况下，使模型将所有预测都指向同一个类别。我们展示了这种现象的存在，称为“初始猜测偏差”（Initial Guessing Bias，IGB），这取决于架构选择，例如激活函数、最大池化层和网络深度。我们对IGB进行的分析具有实际意义，可以指导架构的选择和初始化。我们还强调理论后果，例如节点置换对称性的崩溃、自平均的破坏、某些均场近似的有效性以及深度带来的非平凡差异。

    The initial state of neural networks plays a central role in conditioning the subsequent training dynamics. In the context of classification problems, we provide a theoretical analysis demonstrating that the structure of a neural network can condition the model to assign all predictions to the same class, even before the beginning of training, and in the absence of explicit biases. We show that the presence of this phenomenon, which we call "Initial Guessing Bias" (IGB), depends on architectural choices such as activation functions, max-pooling layers, and network depth. Our analysis of IGB has practical consequences, in that it guides architecture selection and initialization. We also highlight theoretical consequences, such as the breakdown of node-permutation symmetry, the violation of self-averaging, the validity of some mean-field approximations, and the non-trivial differences arising with depth.
    
[^76]: 从概率角度构建语义感知的对抗样本

    Constructing Semantics-Aware Adversarial Examples with Probabilistic Perspective. (arXiv:2306.00353v1 [stat.ML])

    [http://arxiv.org/abs/2306.00353](http://arxiv.org/abs/2306.00353)

    本研究提出了一个基于概率视角的对抗样本构建方法，可以生成语义感知的对抗性样本，并可以有效规避传统对抗性攻击的强化对抗训练方法。

    

    本研究提出了一种新颖的概率视角对抗样本构建方法——箱约束 Langevin Monte Carlo（LMC）。从这个角度出发，我们开发了一种创新性的方法，以原则性的方式生成语义感知的对抗性样本。这种方法超越了几何距离所施加的限制，选择了语义约束。我们的方法赋予了个体将其对语义的理解融入到模型中的能力。通过人类评估，我们验证了我们的语义感知的对抗样本保持其固有的含义。在 MNIST 和 SVHN 数据集上的实验结果表明，我们的语义感知的对抗样本可以有效地规避针对传统对抗性攻击的强健性对抗训练方法。

    In this study, we introduce a novel, probabilistic viewpoint on adversarial examples, achieved through box-constrained Langevin Monte Carlo (LMC). Proceeding from this perspective, we develop an innovative approach for generating semantics-aware adversarial examples in a principled manner. This methodology transcends the restriction imposed by geometric distance, instead opting for semantic constraints. Our approach empowers individuals to incorporate their personal comprehension of semantics into the model. Through human evaluation, we validate that our semantics-aware adversarial examples maintain their inherent meaning. Experimental findings on the MNIST and SVHN datasets demonstrate that our semantics-aware adversarial examples can effectively circumvent robust adversarial training methods tailored for traditional adversarial attacks.
    
[^77]: Shuffle SGD总是比SGD更好：对具有任意数据顺序的SGD进行改进分析

    Shuffle SGD is Always Better than SGD: Improved Analysis of SGD with Arbitrary Data Orders. (arXiv:2305.19259v1 [cs.LG])

    [http://arxiv.org/abs/2305.19259](http://arxiv.org/abs/2305.19259)

    本论文研究了一种允许任意数据排序的普通SGD算法,并表明在非凸函数情况下，随机和单次洗牌的SGD比经典替换的SGD更快或至少与其一样好，无论迭代次数如何。

    

    随机梯度下降（SGD）算法被广泛用于优化神经网络，随机重排（RR）和单次洗牌（SS）是通过循环遍历训练数据的随机或单个排列的常见选择，然而这些算法在非凸情况下的收敛性质尚未完全理解。现有结果表明，在实际的训练场景中，当时代的数量小于训练集大小时，RR可能表现不如SGD。本文分析了一种允许任意数据排序的普通SGD算法，并展示了在非凸函数情况下的改进收敛速度。具体而言，我们的分析表明，随机和单次洗牌的SGD比经典替换的SGD更快或至少与其一样好，无论迭代次数如何。总的来说，我们的研究凸显了使用随机/单次洗牌的SGD的好处，并为其非凸收敛性质提供了新的见解。

    Stochastic Gradient Descent (SGD) algorithms are widely used in optimizing neural networks, with Random Reshuffling (RR) and Single Shuffle (SS) being popular choices for cycling through random or single permutations of the training data. However, the convergence properties of these algorithms in the non-convex case are not fully understood. Existing results suggest that, in realistic training scenarios where the number of epochs is smaller than the training set size, RR may perform worse than SGD.  In this paper, we analyze a general SGD algorithm that allows for arbitrary data orderings and show improved convergence rates for non-convex functions. Specifically, our analysis reveals that SGD with random and single shuffling is always faster or at least as good as classical SGD with replacement, regardless of the number of iterations. Overall, our study highlights the benefits of using SGD with random/single shuffling and provides new insights into its convergence properties for non-co
    
[^78]: 流匹配方法的误差界限

    Error Bounds for Flow Matching Methods. (arXiv:2305.16860v1 [stat.ML])

    [http://arxiv.org/abs/2305.16860](http://arxiv.org/abs/2305.16860)

    本文提出了基于ODE的流匹配方法的误差界限，适用于完全确定性抽样，需要满足$L^2$近似误差范围的规律性条件和数据分布。

    

    基于分数的生成模型是一类依赖于随机微分方程（SDE）的流行生成建模技术。自从它们诞生以来，就已经意识到可以使用普通微分方程（ODE）而不是SDE进行生成。这导致介绍了概率流ODE方法和去噪扩散隐式模型。流匹配方法最近进一步扩展了这些基于ODE的方法，并近似于两个任意概率分布之间的流。以前的工作针对随机抽样模式下的扩散模型推导了近似误差的边界，假设$L^2$损失具有某些限制。我们在完全确定性抽样的情况下提供了流匹配过程的误差界限，假设$L^2$近似误差范围有一定的规律性条件和数据分布。

    Score-based generative models are a popular class of generative modelling techniques relying on stochastic differential equations (SDE). From their inception, it was realized that it was also possible to perform generation using ordinary differential equations (ODE) rather than SDE. This led to the introduction of the probability flow ODE approach and denoising diffusion implicit models. Flow matching methods have recently further extended these ODE-based approaches and approximate a flow between two arbitrary probability distributions. Previous work derived bounds on the approximation error of diffusion models under the stochastic sampling regime, given assumptions on the $L^2$ loss. We present error bounds for the flow matching procedure using fully deterministic sampling, assuming an $L^2$ bound on the approximation error and a certain regularity condition on the data distributions.
    


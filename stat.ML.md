# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multivariate Probabilistic Time Series Forecasting with Correlated Errors](https://rss.arxiv.org/abs/2402.01000) | 本文提出了一种方法，基于低秩加对角线参数化协方差矩阵，可以有效地刻画时间序列预测中误差的自相关性，并具有复杂度低、校准预测准确性高等优点。 |
| [^2] | [The Power of Few: Accelerating and Enhancing Data Reweighting with Coreset Selection](https://arxiv.org/abs/2403.12166) | 提出一种利用核心子集选择进行数据重新加权的方法，有效优化了计算时间和模型性能，突显其作为模型训练的可扩展和精确解决方案的潜力。 |
| [^3] | [Large-scale variational Gaussian state-space models](https://arxiv.org/abs/2403.01371) | 该论文介绍了一种针对具有高斯噪声驱动非线性动力学的状态空间模型的大规模变分算法和结构化逼近方法，可以有效评估ELBO和获取低方差的随机梯度估计，通过利用低秩蒙特卡罗逼近和推断网络的精度矩阵更新，将近似平滑问题转化为近似滤波问题。 |
| [^4] | [Non-asymptotic Convergence of Discrete-time Diffusion Models: New Approach and Improved Rate](https://arxiv.org/abs/2402.13901) | 本文提出了离散时间扩散模型的新方法，改进了对更大类的分布的收敛保证，并提高了具有有界支撑的分布的收敛速率。 |
| [^5] | [Best Arm Identification for Prompt Learning under a Limited Budget](https://arxiv.org/abs/2402.09723) | 这项工作提出了一种在提示学习中考虑有限预算约束的方法，通过建立提示学习和多臂赌博机中固定预算最佳臂识别之间的联系，提出了一个通用框架TRIPLE，通过利用聚类和嵌入思想实现了两个增强方法。 |
| [^6] | [Cross-Temporal Forecast Reconciliation at Digital Platforms with Machine Learning](https://arxiv.org/abs/2402.09033) | 本论文介绍了一种使用机器学习方法在数字平台上进行跨时预测协调的非线性分层预测协调方法，该方法能够直接且自动化地生成跨时预测协调的预测，通过对来自按需交付平台的大规模流式数据集进行实证测试。 |
| [^7] | [An Accelerated Gradient Method for Simple Bilevel Optimization with Convex Lower-level Problem](https://arxiv.org/abs/2402.08097) | 本文提出了一种加速梯度方法来解决具有凸下层问题的简单双层优化问题，通过局部逼近下层问题的解集和加速梯度更新方法，在有限次迭代内找到一个具有一定精度的最优解。 |
| [^8] | [Resampling methods for Private Statistical Inference](https://arxiv.org/abs/2402.07131) | 这项研究提出了两种私有变体的非参数bootstrap方法，用于在差分隐私的情况下构建置信区间。方法在计算效率和置信区间长度上相比现有方法有显著改进。 |
| [^9] | [Identification and Estimation of Conditional Average Partial Causal Effects via Instrumental Variable.](http://arxiv.org/abs/2401.11130) | 本文提出了一种通过工具变量法识别和估计连续处理的因果效应异质性的方法，并开发了三类相应的估计器，并对其进行了统计性质分析。 |
| [^10] | [High-dimensional robust regression under heavy-tailed data: Asymptotics and Universality.](http://arxiv.org/abs/2309.16476) | 本文研究了在高维度和重尾干扰条件下的鲁棒回归估计器的性质，通过研究一类椭圆协变量和噪声数据分布的M-估计器，我们发现在存在重尾噪声的情况下，Huber损失需要进一步正则化才能达到最优性能。同时，我们还推导出了岭回归的超额风险的衰减速率。 |
| [^11] | [Byzantine-Resilient Federated PCA and Low Rank Matrix Recovery.](http://arxiv.org/abs/2309.14512) | 这项工作提出了一个拜占庭鲁棒、通信高效和私密的算法(Subspace-Median)来解决在联邦环境中估计对称矩阵主子空间的问题，同时还研究了联邦主成分分析（PCA）和水平联邦低秩列感知（LRCCS）的特殊情况，并展示了Subspace-Median算法的优势。 |
| [^12] | [BayOTIDE: Bayesian Online Multivariate Time series Imputation with functional decomposition.](http://arxiv.org/abs/2308.14906) | BayOTIDE是一种基于贝叶斯方法的在线多元时间序列插补与函数分解模型，通过将多元时间序列视为低秩时序因子组的加权组合来进行插补，同时解决了全局趋势和周期性模式的忽略以及不规则采样时间序列的处理问题。 |
| [^13] | [From CNNs to Shift-Invariant Twin Models Based on Complex Wavelets.](http://arxiv.org/abs/2212.00394) | 论文提出了一种消除OCV（Aliasing）的方法，该方法基于复数卷积，同时采用Gabor样式的卷积核，可以提高卷积神经网络的分类准确性。 |

# 详细

[^1]: 多元概率时间序列预测与相关误差

    Multivariate Probabilistic Time Series Forecasting with Correlated Errors

    [https://rss.arxiv.org/abs/2402.01000](https://rss.arxiv.org/abs/2402.01000)

    本文提出了一种方法，基于低秩加对角线参数化协方差矩阵，可以有效地刻画时间序列预测中误差的自相关性，并具有复杂度低、校准预测准确性高等优点。

    

    建模误差之间的相关性与模型能够准确量化概率时间序列预测中的预测不确定性密切相关。最近的多元模型在考虑误差之间的同时相关性方面取得了显著进展，然而，对于统计简化的目的，对这些误差的常见假设是它们在时间上是独立的。然而，实际观测往往偏离了这个假设，因为误差通常由于各种因素（如排除时间相关的协变量）而表现出显著的自相关性。在这项工作中，我们提出了一种基于低秩加对角线参数化协方差矩阵的高效方法，可以有效地刻画误差的自相关性。所提出的方法具有几个可取的特性：复杂度不随时间序列数目增加，得到的协方差可以用于校准预测，且具有较好的性能。

    Modeling the correlations among errors is closely associated with how accurately the model can quantify predictive uncertainty in probabilistic time series forecasting. Recent multivariate models have made significant progress in accounting for contemporaneous correlations among errors, while a common assumption on these errors is that they are temporally independent for the sake of statistical simplicity. However, real-world observations often deviate from this assumption, since errors usually exhibit substantial autocorrelation due to various factors such as the exclusion of temporally correlated covariates. In this work, we propose an efficient method, based on a low-rank-plus-diagonal parameterization of the covariance matrix, which can effectively characterize the autocorrelation of errors. The proposed method possesses several desirable properties: the complexity does not scale with the number of time series, the resulting covariance can be used for calibrating predictions, and i
    
[^2]: 少数个体的力量：利用核心子集选择加速和优化数据重新加权

    The Power of Few: Accelerating and Enhancing Data Reweighting with Coreset Selection

    [https://arxiv.org/abs/2403.12166](https://arxiv.org/abs/2403.12166)

    提出一种利用核心子集选择进行数据重新加权的方法，有效优化了计算时间和模型性能，突显其作为模型训练的可扩展和精确解决方案的潜力。

    

    随着机器学习任务不断发展，趋势是收集更大的数据集并训练规模越来越大的模型。虽然这提高了准确性，但也将计算成本提高到不可持续的水平。针对这一问题，我们的工作旨在在计算效率和模型准确性之间取得微妙的平衡，这是该领域中一直存在的挑战。我们引入了一种利用核心子集选择进行重新加权的新方法，有效优化了计算时间和模型性能。通过专注于 strategically selected coreset，我们的方法提供了一个稳健的表示，因为它有效地最小化了异常值的影响。然后，重新校准的权重被映射回并传播到整个数据集。我们的实验结果证实了这种方法的有效性，突显了它作为模型训练的可扩展和精确解决方案的潜力。

    arXiv:2403.12166v1 Announce Type: new  Abstract: As machine learning tasks continue to evolve, the trend has been to gather larger datasets and train increasingly larger models. While this has led to advancements in accuracy, it has also escalated computational costs to unsustainable levels. Addressing this, our work aims to strike a delicate balance between computational efficiency and model accuracy, a persisting challenge in the field. We introduce a novel method that employs core subset selection for reweighting, effectively optimizing both computational time and model performance. By focusing on a strategically selected coreset, our approach offers a robust representation, as it efficiently minimizes the influence of outliers. The re-calibrated weights are then mapped back to and propagated across the entire dataset. Our experimental results substantiate the effectiveness of this approach, underscoring its potential as a scalable and precise solution for model training.
    
[^3]: 大规模变分高斯状态空间模型

    Large-scale variational Gaussian state-space models

    [https://arxiv.org/abs/2403.01371](https://arxiv.org/abs/2403.01371)

    该论文介绍了一种针对具有高斯噪声驱动非线性动力学的状态空间模型的大规模变分算法和结构化逼近方法，可以有效评估ELBO和获取低方差的随机梯度估计，通过利用低秩蒙特卡罗逼近和推断网络的精度矩阵更新，将近似平滑问题转化为近似滤波问题。

    

    我们介绍了一种用于状态空间模型的嵌套变分推断算法和结构化变分逼近方法，其中非线性动力学由高斯噪声驱动。值得注意的是，所提出的框架允许在没有采用对角高斯逼近的情况下有效地评估ELBO和低方差随机梯度估计，通过利用（i）通过动力学对隐状态进行边缘化的蒙特卡罗逼近的低秩结构，（ii）一个推断网络，该网络通过低秩精度矩阵更新来近似更新步骤，（iii）将当前和未来观测编码为伪观测--将近似平滑问题转换为（更简单的）近似滤波问题。整体而言，必要的统计信息和ELBO可以在$O（TL（Sr+S^2+r^2））$时间内计算，其中$T$是系列长度，$L$是状态空间维数，$S$是用于逼近的样本数量。

    arXiv:2403.01371v1 Announce Type: cross  Abstract: We introduce an amortized variational inference algorithm and structured variational approximation for state-space models with nonlinear dynamics driven by Gaussian noise. Importantly, the proposed framework allows for efficient evaluation of the ELBO and low-variance stochastic gradient estimates without resorting to diagonal Gaussian approximations by exploiting (i) the low-rank structure of Monte-Carlo approximations to marginalize the latent state through the dynamics (ii) an inference network that approximates the update step with low-rank precision matrix updates (iii) encoding current and future observations into pseudo observations -- transforming the approximate smoothing problem into an (easier) approximate filtering problem. Overall, the necessary statistics and ELBO can be computed in $O(TL(Sr + S^2 + r^2))$ time where $T$ is the series length, $L$ is the state-space dimensionality, $S$ are the number of samples used to app
    
[^4]: 离散时间扩散模型的非渐近收敛：新方法和改进速率

    Non-asymptotic Convergence of Discrete-time Diffusion Models: New Approach and Improved Rate

    [https://arxiv.org/abs/2402.13901](https://arxiv.org/abs/2402.13901)

    本文提出了离散时间扩散模型的新方法，改进了对更大类的分布的收敛保证，并提高了具有有界支撑的分布的收敛速率。

    

    最近，去噪扩散模型作为一种强大的生成技术出现，将噪声转化为数据。理论上主要研究了连续时间扩散模型的收敛性保证，并且仅在文献中对具有有界支撑的分布的离散时间扩散模型进行了获得。本文为更大类的分布建立了离散时间扩散模型的收敛性保证，并进一步改进了对具有有界支撑的分布的收敛速率。特别地，首先为具有有限二阶矩的平滑和一般（可能非光滑）分布建立了收敛速率。然后将结果专门应用于一些有明确参数依赖关系的有趣分布类别，包括具有Lipschitz分数、高斯混合分布和具有有界支撑的分布。

    arXiv:2402.13901v1 Announce Type: new  Abstract: The denoising diffusion model emerges recently as a powerful generative technique that converts noise into data. Theoretical convergence guarantee has been mainly studied for continuous-time diffusion models, and has been obtained for discrete-time diffusion models only for distributions with bounded support in the literature. In this paper, we establish the convergence guarantee for substantially larger classes of distributions under discrete-time diffusion models and further improve the convergence rate for distributions with bounded support. In particular, we first establish the convergence rates for both smooth and general (possibly non-smooth) distributions having finite second moment. We then specialize our results to a number of interesting classes of distributions with explicit parameter dependencies, including distributions with Lipschitz scores, Gaussian mixture distributions, and distributions with bounded support. We further 
    
[^5]: 有限预算下的迅速学习最佳臂识别

    Best Arm Identification for Prompt Learning under a Limited Budget

    [https://arxiv.org/abs/2402.09723](https://arxiv.org/abs/2402.09723)

    这项工作提出了一种在提示学习中考虑有限预算约束的方法，通过建立提示学习和多臂赌博机中固定预算最佳臂识别之间的联系，提出了一个通用框架TRIPLE，通过利用聚类和嵌入思想实现了两个增强方法。

    

    大型语言模型（LLMs）的显著指令跟随能力引发了对自动学习合适提示的兴趣。然而，虽然提出了许多有效的方法，但在学习过程中产生的成本（例如访问LLM和评估响应）尚未得到考虑。为克服这个限制，本工作在提示学习中明确引入了有限预算约束。为了开发有原则的解决方案，本研究在提示学习和多臂赌博机的固定预算最佳臂识别（BAI-FB）之间建立了一种新的联系。基于这种联系，提出了一个通用框架TRIPLE（用于提示学习的最佳臂识别），以系统地利用BAI-FB在提示学习中的力量。提示学习的独特特点进一步通过利用聚类和嵌入思想提出了TRIPLE的两个基于嵌入的增强方法。

    arXiv:2402.09723v1 Announce Type: cross  Abstract: The remarkable instruction-following capability of large language models (LLMs) has sparked a growing interest in automatically learning suitable prompts. However, while many effective methods have been proposed, the cost incurred during the learning process (e.g., accessing LLM and evaluating the responses) has not been considered. To overcome this limitation, this work explicitly incorporates a finite budget constraint into prompt learning. Towards developing principled solutions, a novel connection is established between prompt learning and fixed-budget best arm identification (BAI-FB) in multi-armed bandits (MAB). Based on this connection, a general framework TRIPLE (besT aRm Identification for Prompt LEarning) is proposed to harness the power of BAI-FB in prompt learning systematically. Unique characteristics of prompt learning further lead to two embedding-based enhancements of TRIPLE by exploiting the ideas of clustering and fun
    
[^6]: 使用机器学习在数字平台上进行跨时预测协调

    Cross-Temporal Forecast Reconciliation at Digital Platforms with Machine Learning

    [https://arxiv.org/abs/2402.09033](https://arxiv.org/abs/2402.09033)

    本论文介绍了一种使用机器学习方法在数字平台上进行跨时预测协调的非线性分层预测协调方法，该方法能够直接且自动化地生成跨时预测协调的预测，通过对来自按需交付平台的大规模流式数据集进行实证测试。

    

    平台业务在数字核心上运作，其决策需要不同层次（例如地理区域）和时间聚合（例如分钟到天）的高维准确预测流。为了确保不同规划单元（如定价、产品、控制和战略）之间的决策一致，也需要在层次结构的所有级别上进行协调预测。鉴于平台数据流具有复杂的特征和相互依赖关系，我们引入了一种非线性分层预测协调方法，通过使用流行的机器学习方法，以直接和自动化的方式生成跨时预测协调的预测。该方法足够快，可以满足平台所需的基于预测的高频决策。我们使用来自领先的按需交付平台的独特大规模流式数据集对我们的框架进行了实证测试。

    arXiv:2402.09033v1 Announce Type: new Abstract: Platform businesses operate on a digital core and their decision making requires high-dimensional accurate forecast streams at different levels of cross-sectional (e.g., geographical regions) and temporal aggregation (e.g., minutes to days). It also necessitates coherent forecasts across all levels of the hierarchy to ensure aligned decision making across different planning units such as pricing, product, controlling and strategy. Given that platform data streams feature complex characteristics and interdependencies, we introduce a non-linear hierarchical forecast reconciliation method that produces cross-temporal reconciled forecasts in a direct and automated way through the use of popular machine learning methods. The method is sufficiently fast to allow forecast-based high-frequency decision making that platforms require. We empirically test our framework on a unique, large-scale streaming dataset from a leading on-demand delivery plat
    
[^7]: 一种加速梯度方法求解具有凸下层问题的简单双层优化问题

    An Accelerated Gradient Method for Simple Bilevel Optimization with Convex Lower-level Problem

    [https://arxiv.org/abs/2402.08097](https://arxiv.org/abs/2402.08097)

    本文提出了一种加速梯度方法来解决具有凸下层问题的简单双层优化问题，通过局部逼近下层问题的解集和加速梯度更新方法，在有限次迭代内找到一个具有一定精度的最优解。

    

    本文主要研究简单的双层优化问题，即在另一个凸光滑约束优化问题的最优解集上最小化一个凸光滑目标函数。我们提出了一种新颖的双层优化方法，通过切平面方法局部逼近下层问题的解集，并采用加速梯度更新方法降低近似解集上的上层目标函数。我们通过子最优解和不可行误差度量我们方法的性能，并提供了对两个误差标准的非渐进收敛性保证。特别地，当可行集是紧致的时候，我们证明了我们的方法最多需要$\mathcal{O}(\max\{1/\sqrt{\epsilon_{f}}, 1/\epsilon_g\})$次迭代才能找到一个$\epsilon_f$-子最优且$\epsilon_g$-不可行的解。此外，在额外假设下，下层目标满足$r$阶H\"olderian误差时，我们给出了解的收敛速度估计。

    In this paper, we focus on simple bilevel optimization problems, where we minimize a convex smooth objective function over the optimal solution set of another convex smooth constrained optimization problem. We present a novel bilevel optimization method that locally approximates the solution set of the lower-level problem using a cutting plane approach and employs an accelerated gradient-based update to reduce the upper-level objective function over the approximated solution set. We measure the performance of our method in terms of suboptimality and infeasibility errors and provide non-asymptotic convergence guarantees for both error criteria. Specifically, when the feasible set is compact, we show that our method requires at most $\mathcal{O}(\max\{1/\sqrt{\epsilon_{f}}, 1/\epsilon_g\})$ iterations to find a solution that is $\epsilon_f$-suboptimal and $\epsilon_g$-infeasible. Moreover, under the additional assumption that the lower-level objective satisfies the $r$-th H\"olderian err
    
[^8]: 针对私有统计推断的重采样方法

    Resampling methods for Private Statistical Inference

    [https://arxiv.org/abs/2402.07131](https://arxiv.org/abs/2402.07131)

    这项研究提出了两种私有变体的非参数bootstrap方法，用于在差分隐私的情况下构建置信区间。方法在计算效率和置信区间长度上相比现有方法有显著改进。

    

    我们考虑使用差分隐私构建置信区间的任务。我们提出了两种私有变体的非参数bootstrap方法，该方法在数据的分区上私下计算多个“小”bootstrap的结果的中位数，并给出了得到的置信区间的渐进覆盖误差上界。对于固定的差分隐私参数ε，我们的方法在样本大小n上的误差率与非私有bootstrap相当，只是在对数因子内。我们使用真实数据和合成数据在均值估计、中位数估计和逻辑回归方面对我们的方法进行了经验验证。我们的方法在提供类似的覆盖精度的同时，比以前的方法提供了显著缩短（大约10倍）的置信区间。

    We consider the task of constructing confidence intervals with differential privacy. We propose two private variants of the non-parametric bootstrap, which privately compute the median of the results of multiple ``little'' bootstraps run on partitions of the data and give asymptotic bounds on the coverage error of the resulting confidence intervals. For a fixed differential privacy parameter $\epsilon$, our methods enjoy the same error rates as that of the non-private bootstrap to within logarithmic factors in the sample size $n$. We empirically validate the performance of our methods for mean estimation, median estimation, and logistic regression with both real and synthetic data. Our methods achieve similar coverage accuracy to existing methods (and non-private baselines) while providing notably shorter ($\gtrsim 10$ times) confidence intervals than previous approaches.
    
[^9]: 通过工具变量法识别和估计条件平均偏因果效应

    Identification and Estimation of Conditional Average Partial Causal Effects via Instrumental Variable. (arXiv:2401.11130v1 [cs.LG])

    [http://arxiv.org/abs/2401.11130](http://arxiv.org/abs/2401.11130)

    本文提出了一种通过工具变量法识别和估计连续处理的因果效应异质性的方法，并开发了三类相应的估计器，并对其进行了统计性质分析。

    

    近年来，对于估计异质因果效应的兴趣日益增加。本文引入了条件平均偏因果效应（CAPCE），以揭示连续处理的因果效应的异质性。我们给出了在工具变量设置下识别CAPCE的条件。我们开发了三类CAPCE估计器：筛选、参数化和再生核希尔伯特空间（RKHS）-基础，分析了它们的统计性质。我们通过合成和真实数据对提出的CAPCE估计器进行了说明。

    There has been considerable recent interest in estimating heterogeneous causal effects. In this paper, we introduce conditional average partial causal effects (CAPCE) to reveal the heterogeneity of causal effects with continuous treatment. We provide conditions for identifying CAPCE in an instrumental variable setting. We develop three families of CAPCE estimators: sieve, parametric, and reproducing kernel Hilbert space (RKHS)-based, and analyze their statistical properties. We illustrate the proposed CAPCE estimators on synthetic and real-world data.
    
[^10]: 高维度下重尾数据下的鲁棒回归: 渐近性和普适性

    High-dimensional robust regression under heavy-tailed data: Asymptotics and Universality. (arXiv:2309.16476v1 [math.ST])

    [http://arxiv.org/abs/2309.16476](http://arxiv.org/abs/2309.16476)

    本文研究了在高维度和重尾干扰条件下的鲁棒回归估计器的性质，通过研究一类椭圆协变量和噪声数据分布的M-估计器，我们发现在存在重尾噪声的情况下，Huber损失需要进一步正则化才能达到最优性能。同时，我们还推导出了岭回归的超额风险的衰减速率。

    

    我们研究了在协变量和响应函数都受重尾干扰的情况下，鲁棒回归估计量的高维性质。特别地，我们针对一类包含椭圆协变量和噪声数据分布的M-估计器提供了锐利的渐近特征化，包括二阶及以上矩不存在的情况。我们发现，在存在重尾噪声的高维情况下，尽管Huber损失通过最优调整的位置参数$\delta$是一致的，但其在性能上是次优的，突显了进一步正则化以达到最优性能的必要性。这个结果还揭示了$\delta$作为样本复杂度和污染的函数存在的一个有趣的转变。此外，我们还推导出岭回归中超额风险的衰减速率。我们发现，对于具有有限二阶矩的噪声分布，岭回归不仅是最优的，而且是普适的，但其衰减速率可以是...

    We investigate the high-dimensional properties of robust regression estimators in the presence of heavy-tailed contamination of both the covariates and response functions. In particular, we provide a sharp asymptotic characterisation of M-estimators trained on a family of elliptical covariate and noise data distributions including cases where second and higher moments do not exist. We show that, despite being consistent, the Huber loss with optimally tuned location parameter $\delta$ is suboptimal in the high-dimensional regime in the presence of heavy-tailed noise, highlighting the necessity of further regularisation to achieve optimal performance. This result also uncovers the existence of a curious transition in $\delta$ as a function of the sample complexity and contamination. Moreover, we derive the decay rates for the excess risk of ridge regression. We show that, while it is both optimal and universal for noise distributions with finite second moment, its decay rate can be consi
    
[^11]: 拜占庭鲁棒的联邦PCA和低秩矩阵恢复

    Byzantine-Resilient Federated PCA and Low Rank Matrix Recovery. (arXiv:2309.14512v1 [cs.IT])

    [http://arxiv.org/abs/2309.14512](http://arxiv.org/abs/2309.14512)

    这项工作提出了一个拜占庭鲁棒、通信高效和私密的算法(Subspace-Median)来解决在联邦环境中估计对称矩阵主子空间的问题，同时还研究了联邦主成分分析（PCA）和水平联邦低秩列感知（LRCCS）的特殊情况，并展示了Subspace-Median算法的优势。

    

    在这项工作中，我们考虑了在联邦环境中估计对称矩阵的主子空间（前r个奇异向量的张成）的问题，当每个节点都可以访问对这个矩阵的估计时。我们研究如何使这个问题具有拜占庭鲁棒性。我们引入了一种新颖的可证明的拜占庭鲁棒、通信高效和私密的算法，称为子空间中值算法（Subspace-Median），用于解决这个问题。我们还研究了这个问题的最自然的解法，基于几何中值的修改的联邦幂方法，并解释为什么它是无用的。在这项工作中，我们考虑了鲁棒子空间估计元问题的两个特殊情况 - 联邦主成分分析（PCA）和水平联邦低秩列感知（LRCCS）的谱初始化步骤。对于这两个问题，我们展示了子空间中值算法提供了既具有鲁棒性又具有高通信效率的解决方案。均值的中位数扩展也被开发出来了。

    In this work we consider the problem of estimating the principal subspace (span of the top r singular vectors) of a symmetric matrix in a federated setting, when each node has access to estimates of this matrix. We study how to make this problem Byzantine resilient. We introduce a novel provably Byzantine-resilient, communication-efficient, and private algorithm, called Subspace-Median, to solve it. We also study the most natural solution for this problem, a geometric median based modification of the federated power method, and explain why it is not useful. We consider two special cases of the resilient subspace estimation meta-problem - federated principal components analysis (PCA) and the spectral initialization step of horizontally federated low rank column-wise sensing (LRCCS) in this work. For both these problems we show how Subspace Median provides a resilient solution that is also communication-efficient. Median of Means extensions are developed for both problems. Extensive simu
    
[^12]: BayOTIDE: 基于贝叶斯方法的在线多元时间序列插补与函数分解

    BayOTIDE: Bayesian Online Multivariate Time series Imputation with functional decomposition. (arXiv:2308.14906v1 [cs.LG])

    [http://arxiv.org/abs/2308.14906](http://arxiv.org/abs/2308.14906)

    BayOTIDE是一种基于贝叶斯方法的在线多元时间序列插补与函数分解模型，通过将多元时间序列视为低秩时序因子组的加权组合来进行插补，同时解决了全局趋势和周期性模式的忽略以及不规则采样时间序列的处理问题。

    

    在真实世界的场景中，如交通和能源，经常观察到具有缺失值和噪声的大规模时间序列数据，甚至是不规则采样。尽管已经提出了许多插补方法，但它们大多数只适用于局部视角，即将长序列拆分为适当大小的批次进行训练。这种局部视角可能使模型忽略全局趋势或周期性模式。更重要的是，几乎所有方法都假设观测值在规则的时间间隔进行采样，并且无法处理来自不同应用的复杂不规则采样时间序列。此外，大多数现有方法都是在离线状态下进行学习的。因此，对于那些有快速到达的流数据的应用来说，它们并不合适。为了克服这些局限性，我们提出了BayOTIDE：基于贝叶斯方法的在线多元时间序列插补与函数分解。

    In real-world scenarios like traffic and energy, massive time-series data with missing values and noises are widely observed, even sampled irregularly. While many imputation methods have been proposed, most of them work with a local horizon, which means models are trained by splitting the long sequence into batches of fit-sized patches. This local horizon can make models ignore global trends or periodic patterns. More importantly, almost all methods assume the observations are sampled at regular time stamps, and fail to handle complex irregular sampled time series arising from different applications. Thirdly, most existing methods are learned in an offline manner. Thus, it is not suitable for many applications with fast-arriving streaming data. To overcome these limitations, we propose \ours: Bayesian Online Multivariate Time series Imputation with functional decomposition. We treat the multivariate time series as the weighted combination of groups of low-rank temporal factors with dif
    
[^13]: 从CNN到基于复小波的平移不变双模型

    From CNNs to Shift-Invariant Twin Models Based on Complex Wavelets. (arXiv:2212.00394v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2212.00394](http://arxiv.org/abs/2212.00394)

    论文提出了一种消除OCV（Aliasing）的方法，该方法基于复数卷积，同时采用Gabor样式的卷积核，可以提高卷积神经网络的分类准确性。

    

    我们提出了一种新颖的抗混叠方法来增加卷积神经网络中的平移不变性和预测准确性。具体来说，我们用“复值卷积+模运算”（$\mathbb{C}$Mod）代替第一层的“实值卷积+最大池化”（$\mathbb{R}$Max），因为它稳定于平移。为了证明我们的方法，我们声称当卷积核是带通和定向的（类似于Gabor滤波器）时，$\mathbb{C}$Mod和$\mathbb{R}$Max产生可比较的输出。在这种情况下，$\mathbb{C}$Mod可以被认为是$\mathbb{R}$Max的稳定替代品。因此，在抗混叠之前，我们强制卷积核采用这种Gabor样式的结构。相应的架构称为数学双模型，因为它使用一个明确定义的数学运算符来模拟原始的自由训练模型的行为。我们的抗混叠方法在Imagenet和CIFAR-10分类任务上实现了比之前更高的准确性。

    We propose a novel antialiasing method to increase shift invariance and prediction accuracy in convolutional neural networks. Specifically, we replace the first-layer combination "real-valued convolutions + max pooling" ($\mathbb{R}$Max) by "complex-valued convolutions + modulus" ($\mathbb{C}$Mod), which is stable to translations. To justify our approach, we claim that $\mathbb{C}$Mod and $\mathbb{R}$Max produce comparable outputs when the convolution kernel is band-pass and oriented (Gabor-like filter). In this context, $\mathbb{C}$Mod can be considered as a stable alternative to $\mathbb{R}$Max. Thus, prior to antialiasing, we force the convolution kernels to adopt such a Gabor-like structure. The corresponding architecture is called mathematical twin, because it employs a well-defined mathematical operator to mimic the behavior of the original, freely-trained model. Our antialiasing approach achieves superior accuracy on ImageNet and CIFAR-10 classification tasks, compared to prior 
    


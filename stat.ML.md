# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Subspace Identification for Multi-Source Domain Adaptation.](http://arxiv.org/abs/2310.04723) | 该论文提出了一个基于子空间识别理论的多源域自适应方法，通过最小化域之间的偏移对不变变量的影响，实现了源域的知识转移到目标域。该方法相对于现有方法更加灵活，不需要满足严格的假设条件。 |
| [^2] | [Diffusion Model in Causal Inference with Unmeasured Confounders.](http://arxiv.org/abs/2308.03669) | 本研究扩展了扩散模型的使用，提出了一种新的模型BDCM，可以在存在无法测量的混淆因素的情况下更准确地回答因果问题。 |
| [^3] | [Doubly Robust Estimator for Off-Policy Evaluation with Large Action Spaces.](http://arxiv.org/abs/2308.03443) | 本文提出了一种用于具有大动作空间的离策略评估的双重稳健估计器（MDR）。与现有的基准估计器相比，MDR能够在减小方差的同时保持无偏性，从而提高了估计的准确性。实验结果证实了MDR相对于现有估计器的优越性。 |
| [^4] | [Big Data - Supply Chain Management Framework for Forecasting: Data Preprocessing and Machine Learning Techniques.](http://arxiv.org/abs/2307.12971) | 本文介绍了一种新的大数据-供应链管理框架，通过数据预处理和机器学习技术实现供应链预测，优化操作管理、透明度，并讨论了幻影库存对预测的不利影响。 |
| [^5] | [Lagrangian Flow Networks for Conservation Laws.](http://arxiv.org/abs/2305.16846) | 该论文提出了LFlows模型，它使用可微和可逆的变换，在时间上规定参数化的微分同胚变换来对基础密度进行转换，以连续地建模流体密度和速度。与传统方法相比，其优势在于速度的解析表达式总是与密度保持一致，无需昂贵的数值求解器，也无需使用惩罚方法。 |
| [^6] | [Optimal inference of a generalised Potts model by single-layer transformers with factored attention.](http://arxiv.org/abs/2304.07235) | 我们将分析和数值推导结合，在基于广义 Potts 模型的数据上，对经过改进适应这种模型的self-attention机制进行训练，发现经过修改的self-attention机制可以在极限采样下准确学习Potts模型。这个“分解”注意力机制通过从数据中学习相关属性，可以提高Transformer的性能和可解释性。 |
| [^7] | [Distributed Stochastic Optimization under a General Variance Condition.](http://arxiv.org/abs/2301.12677) | 这项研究通过重新审视联邦平均算法，在最小假设下对分布式非凸目标进行了随机优化，建立了仅满足随机梯度温和条件的收敛结果。 |

# 详细

[^1]: 多源域自适应的子空间识别

    Subspace Identification for Multi-Source Domain Adaptation. (arXiv:2310.04723v1 [cs.LG])

    [http://arxiv.org/abs/2310.04723](http://arxiv.org/abs/2310.04723)

    该论文提出了一个基于子空间识别理论的多源域自适应方法，通过最小化域之间的偏移对不变变量的影响，实现了源域的知识转移到目标域。该方法相对于现有方法更加灵活，不需要满足严格的假设条件。

    

    多源域自适应（MSDA）方法旨在将多个有标签的源域的知识转移到一个无标签的目标域中。尽管当前的方法通过在域之间施加最小的变化来实现目标联合分布的可辨识性，但它们通常需要严格的条件，如足够数量的域、潜在变量的单调变换和不变的标签分布。这些要求在实际应用中很难满足。为了减轻对这些严格假设的需求，我们提出了一个子空间识别理论，它在关于域数量和变换特性方面具有较宽松的约束条件，从而通过最小化域之间的偏移对不变变量的影响来促进域自适应。基于这个理论，我们开发了一个利用变分推断的子空间识别保证（SIG）模型。

    Multi-source domain adaptation (MSDA) methods aim to transfer knowledge from multiple labeled source domains to an unlabeled target domain. Although current methods achieve target joint distribution identifiability by enforcing minimal changes across domains, they often necessitate stringent conditions, such as an adequate number of domains, monotonic transformation of latent variables, and invariant label distributions. These requirements are challenging to satisfy in real-world applications. To mitigate the need for these strict assumptions, we propose a subspace identification theory that guarantees the disentanglement of domain-invariant and domain-specific variables under less restrictive constraints regarding domain numbers and transformation properties, thereby facilitating domain adaptation by minimizing the impact of domain shifts on invariant variables. Based on this theory, we develop a Subspace Identification Guarantee (SIG) model that leverages variational inference. Furth
    
[^2]: 无法测量混淆因素下因果推断中的扩散模型

    Diffusion Model in Causal Inference with Unmeasured Confounders. (arXiv:2308.03669v1 [cs.LG])

    [http://arxiv.org/abs/2308.03669](http://arxiv.org/abs/2308.03669)

    本研究扩展了扩散模型的使用，提出了一种新的模型BDCM，可以在存在无法测量的混淆因素的情况下更准确地回答因果问题。

    

    我们研究了如何在无法测量的混淆因素存在的情况下，扩展扩散模型的使用，以从观测数据中回答因果问题。在Pearl的使用有向无环图（DAG）捕捉因果干预的框架中，提出了一种基于扩散模型的因果模型（DCM），可以更准确地回答因果问题，假设所有混淆因素都是可以观察到的。然而，实际中存在无法测量的混淆因素，这使得DCM无法应用。为了缓解DCM的这一局限性，我们提出了一个扩展模型，称为基于反门准则的DCM（BDCM），其思想根植于在DAG中找到要包括在扩散模型解码过程中的变量的反门准则，这样我们可以将DCM扩展到存在无法测量的混淆因素的情况。合成数据实验表明，我们提出的模型在无法测量混淆因素的情况下更精确地捕捉到了反事实分布。

    We study how to extend the use of the diffusion model to answer the causal question from the observational data under the existence of unmeasured confounders. In Pearl's framework of using a Directed Acyclic Graph (DAG) to capture the causal intervention, a Diffusion-based Causal Model (DCM) was proposed incorporating the diffusion model to answer the causal questions more accurately, assuming that all of the confounders are observed. However, unmeasured confounders in practice exist, which hinders DCM from being applicable. To alleviate this limitation of DCM, we propose an extended model called Backdoor Criterion based DCM (BDCM), whose idea is rooted in the Backdoor criterion to find the variables in DAG to be included in the decoding process of the diffusion model so that we can extend DCM to the case with unmeasured confounders. Synthetic data experiment demonstrates that our proposed model captures the counterfactual distribution more precisely than DCM under the unmeasured confo
    
[^3]: 用于具有大动作空间的离策略评估的双重稳健估计器

    Doubly Robust Estimator for Off-Policy Evaluation with Large Action Spaces. (arXiv:2308.03443v1 [stat.ML])

    [http://arxiv.org/abs/2308.03443](http://arxiv.org/abs/2308.03443)

    本文提出了一种用于具有大动作空间的离策略评估的双重稳健估计器（MDR）。与现有的基准估计器相比，MDR能够在减小方差的同时保持无偏性，从而提高了估计的准确性。实验结果证实了MDR相对于现有估计器的优越性。

    

    本文研究了在具有大动作空间的背景下的离策略评估（OPE）。现有的基准估计器存在严重的偏差和方差折衷问题。参数化方法由于很难确定正确的模型而导致偏差，而重要性加权方法由于方差而产生问题。为了克服这些限制，本文提出了基于判别式的不良行为抑制器（MIPS）来通过对动作的嵌入来减小估计器的方差。为了使估计器更准确，我们提出了MIPS的双重稳健估计器——边际化双重稳健（MDR）估计器。理论分析表明，所提出的估计器在比MIPS更弱的假设下是无偏的，同时保持了对IPS的方差减小，这是MIPS的主要优势。经验实验证实了MDR相对于现有估计器的优越性。

    We study Off-Policy Evaluation (OPE) in contextual bandit settings with large action spaces. The benchmark estimators suffer from severe bias and variance tradeoffs. Parametric approaches suffer from bias due to difficulty specifying the correct model, whereas ones with importance weight suffer from variance. To overcome these limitations, Marginalized Inverse Propensity Scoring (MIPS) was proposed to mitigate the estimator's variance via embeddings of an action. To make the estimator more accurate, we propose the doubly robust estimator of MIPS called the Marginalized Doubly Robust (MDR) estimator. Theoretical analysis shows that the proposed estimator is unbiased under weaker assumptions than MIPS while maintaining variance reduction against IPS, which was the main advantage of MIPS. The empirical experiment verifies the supremacy of MDR against existing estimators.
    
[^4]: 大数据-供应链管理框架的预测：数据预处理和机器学习技术

    Big Data - Supply Chain Management Framework for Forecasting: Data Preprocessing and Machine Learning Techniques. (arXiv:2307.12971v1 [cs.LG])

    [http://arxiv.org/abs/2307.12971](http://arxiv.org/abs/2307.12971)

    本文介绍了一种新的大数据-供应链管理框架，通过数据预处理和机器学习技术实现供应链预测，优化操作管理、透明度，并讨论了幻影库存对预测的不利影响。

    

    本文旨在系统地识别和比较分析最先进的供应链预测策略和技术。提出了一个新的框架，将大数据分析应用于供应链管理中，包括问题识别、数据来源、探索性数据分析、机器学习模型训练、超参数调优、性能评估和优化，以及预测对人力、库存和整个供应链的影响。首先讨论了根据供应链策略收集数据的需求以及如何收集数据。文章讨论了根据周期或供应链目标需要不同类型的预测。推荐使用供应链绩效指标和误差测量系统来优化表现最佳的模型。还讨论了幻影库存对预测的不利影响以及管理决策依赖供应链绩效指标来确定模型性能参数和改进运营管理、透明度的问题。

    This article intends to systematically identify and comparatively analyze state-of-the-art supply chain (SC) forecasting strategies and technologies. A novel framework has been proposed incorporating Big Data Analytics in SC Management (problem identification, data sources, exploratory data analysis, machine-learning model training, hyperparameter tuning, performance evaluation, and optimization), forecasting effects on human-workforce, inventory, and overall SC. Initially, the need to collect data according to SC strategy and how to collect them has been discussed. The article discusses the need for different types of forecasting according to the period or SC objective. The SC KPIs and the error-measurement systems have been recommended to optimize the top-performing model. The adverse effects of phantom inventory on forecasting and the dependence of managerial decisions on the SC KPIs for determining model performance parameters and improving operations management, transparency, and 
    
[^5]: 拉格朗日流网络用于守恒定律

    Lagrangian Flow Networks for Conservation Laws. (arXiv:2305.16846v1 [cs.LG])

    [http://arxiv.org/abs/2305.16846](http://arxiv.org/abs/2305.16846)

    该论文提出了LFlows模型，它使用可微和可逆的变换，在时间上规定参数化的微分同胚变换来对基础密度进行转换，以连续地建模流体密度和速度。与传统方法相比，其优势在于速度的解析表达式总是与密度保持一致，无需昂贵的数值求解器，也无需使用惩罚方法。

    

    我们提出了拉格朗日流网络（LFlows），用于连续地建模流体密度和速度。所提出的LFlows基于连续方程的解，其中连续方程是描述不同形式的质量守恒性质的偏微分方程。我们的模型基于这样的思路：连续方程的解可以通过可微和可逆的变换表示为时间依赖的密度变换。因此，我们通过在时间上规定参数化的微分同胚变换来对基础密度进行转换以建模流体密度。与依赖于Neural-ODE或PINNs的方法相比，关键的优势在于速度的解析表达式始终与密度保持一致。此外，无需昂贵的数值求解器，也无需使用惩罚方法来实施偏微分方程。拉格朗日流网络在合成密度数据上显示出了更高的预测精度。

    We introduce Lagrangian Flow Networks (LFlows) for modeling fluid densities and velocities continuously in space and time. The proposed LFlows satisfy by construction the continuity equation, a PDE describing mass conservation in its differentiable form. Our model is based on the insight that solutions to the continuity equation can be expressed as time-dependent density transformations via differentiable and invertible maps. This follows from classical theory of existence and uniqueness of Lagrangian flows for smooth vector fields. Hence, we model fluid densities by transforming a base density with parameterized diffeomorphisms conditioned on time. The key benefit compared to methods relying on Neural-ODE or PINNs is that the analytic expression of the velocity is always consistent with the density. Furthermore, there is no need for expensive numerical solvers, nor for enforcing the PDE with penalty methods. Lagrangian Flow Networks show improved predictive accuracy on synthetic densi
    
[^6]: 利用分解注意力机制的单层Transformer对广义Potts模型进行最优推断

    Optimal inference of a generalised Potts model by single-layer transformers with factored attention. (arXiv:2304.07235v1 [cond-mat.dis-nn])

    [http://arxiv.org/abs/2304.07235](http://arxiv.org/abs/2304.07235)

    我们将分析和数值推导结合，在基于广义 Potts 模型的数据上，对经过改进适应这种模型的self-attention机制进行训练，发现经过修改的self-attention机制可以在极限采样下准确学习Potts模型。这个“分解”注意力机制通过从数据中学习相关属性，可以提高Transformer的性能和可解释性。

    

    Transformer 是一种革命性的神经网络，在自然语言处理和蛋白质科学方面取得了实践上的成功。它们的关键构建块是一个叫做自注意力机制的机制，它被训练用于预测句子中缺失的词。尽管Transformer在应用中取得了实践上的成功，但是自注意力机制究竟从数据中学到了什么以及它是怎么做到的还不是很清楚。本文针对从具有相互作用的位置和 Potts 颜色中提取的数据在训练的Transformer上给出了精确的分析和数值刻画。我们证明，虽然一般的transformer需要多层学习才能准确学习这个分布，但是经过小改进的自注意力机制在无限采样的极限下可以完美地学习Potts模型。我们还计算了这个修改后的自注意力机制所谓“分解”的泛化误差，并在合成数据上数值演示了我们的发现。我们的结果为解释Transformer的内在工作原理以及提高其性能和可解释性提供了新的思路。

    Transformers are the type of neural networks that has revolutionised natural language processing and protein science. Their key building block is a mechanism called self-attention which is trained to predict missing words in sentences. Despite the practical success of transformers in applications it remains unclear what self-attention learns from data, and how. Here, we give a precise analytical and numerical characterisation of transformers trained on data drawn from a generalised Potts model with interactions between sites and Potts colours. While an off-the-shelf transformer requires several layers to learn this distribution, we show analytically that a single layer of self-attention with a small modification can learn the Potts model exactly in the limit of infinite sampling. We show that this modified self-attention, that we call ``factored'', has the same functional form as the conditional probability of a Potts spin given the other spins, compute its generalisation error using t
    
[^7]: 通用方差条件下的分布式随机优化

    Distributed Stochastic Optimization under a General Variance Condition. (arXiv:2301.12677v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2301.12677](http://arxiv.org/abs/2301.12677)

    这项研究通过重新审视联邦平均算法，在最小假设下对分布式非凸目标进行了随机优化，建立了仅满足随机梯度温和条件的收敛结果。

    

    分布式随机优化在解决大规模机器学习问题时表现出了很高的效率。尽管已经提出并成功应用于一般实际问题的算法很多，但它们的理论保证主要依赖于随机梯度的某些有界条件，从均匀有界性到放松增长条件。此外，在代理之间表征数据异质性及其对算法性能的影响依然具有挑战性。出于这样的动机，我们重新考虑了经典的联邦平均（FedAvg）算法，以解决分布式随机优化问题，并在平滑非凸目标函数的随机梯度仅满足温和方差条件的情况下建立了收敛结果。在此条件下，还建立了接近确定的收敛到一个稳态点。此外，我们讨论了一个更具信息性的度量标准。

    Distributed stochastic optimization has drawn great attention recently due to its effectiveness in solving large-scale machine learning problems. Though numerous algorithms have been proposed and successfully applied to general practical problems, their theoretical guarantees mainly rely on certain boundedness conditions on the stochastic gradients, varying from uniform boundedness to the relaxed growth condition. In addition, how to characterize the data heterogeneity among the agents and its impacts on the algorithmic performance remains challenging. In light of such motivations, we revisit the classical Federated Averaging (FedAvg) algorithm for solving the distributed stochastic optimization problem and establish the convergence results under only a mild variance condition on the stochastic gradients for smooth nonconvex objective functions. Almost sure convergence to a stationary point is also established under the condition. Moreover, we discuss a more informative measurement for
    


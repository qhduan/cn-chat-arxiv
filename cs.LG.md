# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Task-optimal data-driven surrogate models for eNMPC via differentiable simulation and optimization](https://arxiv.org/abs/2403.14425) | 提出了一种基于可微分模拟和优化的任务最优数据驱动替代模型方法，在eNMPC中表现出优越性能，为实现更具能力的控制器提供了有前途的途径。 |
| [^2] | [Reusing Historical Trajectories in Natural Policy Gradient via Importance Sampling: Convergence and Convergence Rate](https://arxiv.org/abs/2403.00675) | 通过重要性抽样在自然策略梯度中重用历史轨迹可提高收敛速率 |
| [^3] | [Improved Performances and Motivation in Intelligent Tutoring Systems: Combining Machine Learning and Learner Choice](https://arxiv.org/abs/2402.01669) | 本研究通过结合机器学习和学生选择，改进了智能辅导系统的性能和动机。使用ZPDES算法，该系统能够最大化学习进展，并在实地研究中提高了不同学生群体的学习成绩。研究还探讨了学生选择对学习效率和动机的影响。 |
| [^4] | [Unsupervised Video Domain Adaptation with Masked Pre-Training and Collaborative Self-Training](https://arxiv.org/abs/2312.02914) | 该方法提出了UNITE框架，利用图像教师模型和视频学生模型进行遮蔽预训练和协作自训练，在多个视频领域自适应基准上取得显著改进的结果。 |
| [^5] | [A Coreset-based, Tempered Variational Posterior for Accurate and Scalable Stochastic Gaussian Process Inference.](http://arxiv.org/abs/2311.01409) | 这篇论文提出了一种基于核心集的、温和变分后验的高斯过程推理方法，通过利用稀疏的、可解释的数据表示来降低参数大小，并且具有数值稳定性和较低的时间和空间复杂度。 |
| [^6] | [$\mu^2$-SGD: Stable Stochastic Optimization via a Double Momentum Mechanism.](http://arxiv.org/abs/2304.04172) | 提出了一种新的梯度估计方法，结合了最近的两种与动量概念相关的机制，实现了稳定的随机优化，对学习率选择具有鲁棒性，在无噪声和有噪声情况下的收敛速率均为最优。 |
| [^7] | [Verifiable and Provably Secure Machine Unlearning.](http://arxiv.org/abs/2210.09126) | 该论文提出了可证明安全的机器学习去除算法，可以让用户审计这个过程，以确保训练数据的隐私得到保护。 |

# 详细

[^1]: 基于可微分模拟和优化的任务最优数据驱动替代模型用于eNMPC

    Task-optimal data-driven surrogate models for eNMPC via differentiable simulation and optimization

    [https://arxiv.org/abs/2403.14425](https://arxiv.org/abs/2403.14425)

    提出了一种基于可微分模拟和优化的任务最优数据驱动替代模型方法，在eNMPC中表现出优越性能，为实现更具能力的控制器提供了有前途的途径。

    

    我们提出了一种用于控制中优化性能的Koopman替代模型端到端学习方法。与之前采用标准强化学习（RL）算法的贡献相反，我们使用一种训练算法，利用基于机械模拟模型的环境的潜在可微性。通过将我们的方法与文献已知的eNMPC案例研究中其他控制器类型和训练算法组合的性能进行比较，我们评估了我们方法的性能。我们的方法在这个问题上表现出优越的性能，因此在使用动态替代模型的更有能力的控制器方面构成了一个有前途的途径。

    arXiv:2403.14425v1 Announce Type: new  Abstract: We present a method for end-to-end learning of Koopman surrogate models for optimal performance in control. In contrast to previous contributions that employ standard reinforcement learning (RL) algorithms, we use a training algorithm that exploits the potential differentiability of environments based on mechanistic simulation models. We evaluate the performance of our method by comparing it to that of other controller type and training algorithm combinations on a literature known eNMPC case study. Our method exhibits superior performance on this problem, thereby constituting a promising avenue towards more capable controllers that employ dynamic surrogate models.
    
[^2]: 通过重要性抽样在自然策略梯度中重用历史轨迹：收敛性和收敛速率

    Reusing Historical Trajectories in Natural Policy Gradient via Importance Sampling: Convergence and Convergence Rate

    [https://arxiv.org/abs/2403.00675](https://arxiv.org/abs/2403.00675)

    通过重要性抽样在自然策略梯度中重用历史轨迹可提高收敛速率

    

    强化学习提供了一个学习控制的数学框架，其成功在很大程度上取决于它可以利用的数据量。有效利用先前策略得到的历史轨迹对于加快策略优化至关重要。实证证据表明基于重要性抽样的策略梯度方法效果良好。然而，现有文献往往忽视了不同迭代之间轨迹的相互依赖性，且良好的实证表现缺乏严格的理论证明。本文研究了一种通过重要性抽样重新利用历史轨迹的自然策略梯度方法的变体。我们表明了所提梯度估计器的偏差渐近可忽略，得到的算法是收敛的，并且重用过去的轨迹有助于提高收敛速率。我们进一步将所提估计器应用于

    arXiv:2403.00675v1 Announce Type: new  Abstract: Reinforcement learning provides a mathematical framework for learning-based control, whose success largely depends on the amount of data it can utilize. The efficient utilization of historical trajectories obtained from previous policies is essential for expediting policy optimization. Empirical evidence has shown that policy gradient methods based on importance sampling work well. However, existing literature often neglect the interdependence between trajectories from different iterations, and the good empirical performance lacks a rigorous theoretical justification. In this paper, we study a variant of the natural policy gradient method with reusing historical trajectories via importance sampling. We show that the bias of the proposed estimator of the gradient is asymptotically negligible, the resultant algorithm is convergent, and reusing past trajectories helps improve the convergence rate. We further apply the proposed estimator to 
    
[^3]: 智能辅导系统中的性能和动机的改进：结合机器学习和学习者选择

    Improved Performances and Motivation in Intelligent Tutoring Systems: Combining Machine Learning and Learner Choice

    [https://arxiv.org/abs/2402.01669](https://arxiv.org/abs/2402.01669)

    本研究通过结合机器学习和学生选择，改进了智能辅导系统的性能和动机。使用ZPDES算法，该系统能够最大化学习进展，并在实地研究中提高了不同学生群体的学习成绩。研究还探讨了学生选择对学习效率和动机的影响。

    

    在学校中，大规模的课堂规模给个性化学习带来了挑战，教育技术，尤其是智能辅导系统（ITS）试图解决这个问题。在这个背景下，基于学习进展假设（LPH）和多臂赌博机器学习技术的ZPDES算法对最大化学习进展（LP）的练习进行排序。该算法在之前的实地研究中已经显示出将学习表现提升到更广泛的学生群体中，与手工设计的课程相比。然而，其动机影响尚未评估。此外，ZPDES不允许学生发表选择意见。这种缺乏机构的限制与关注建模好奇驱动学习的LPH理论不一致。我们在这里研究了这种选择可能性的引入如何影响学习效率和动机。给定的选择与练习难度正交的维度有关，作为一种有趣的特性。

    Large class sizes pose challenges to personalized learning in schools, which educational technologies, especially intelligent tutoring systems (ITS), aim to address. In this context, the ZPDES algorithm, based on the Learning Progress Hypothesis (LPH) and multi-armed bandit machine learning techniques, sequences exercises that maximize learning progress (LP). This algorithm was previously shown in field studies to boost learning performances for a wider diversity of students compared to a hand-designed curriculum. However, its motivational impact was not assessed. Also, ZPDES did not allow students to express choices. This limitation in agency is at odds with the LPH theory concerned with modeling curiosity-driven learning. We here study how the introduction of such choice possibilities impact both learning efficiency and motivation. The given choice concerns dimensions that are orthogonal to exercise difficulty, acting as a playful feature.   In an extensive field study (265 7-8 years
    
[^4]: 无监督视频域自适应：采用遮蔽预训练和协作自训练

    Unsupervised Video Domain Adaptation with Masked Pre-Training and Collaborative Self-Training

    [https://arxiv.org/abs/2312.02914](https://arxiv.org/abs/2312.02914)

    该方法提出了UNITE框架，利用图像教师模型和视频学生模型进行遮蔽预训练和协作自训练，在多个视频领域自适应基准上取得显著改进的结果。

    

    在这项工作中，我们解决了视频动作识别的无监督域自适应（UDA）问题。我们提出的方法称为UNITE，使用图像教师模型来调整视频学生模型到目标域。UNITE首先采用自监督预训练，通过教师引导的遮蔽蒸馏目标得到具有区分性的特征学习。然后我们对目标数据进行遮蔽自训练，利用视频学生模型和图像教师模型一起为未标记的目标视频生成改进的伪标签。我们的自训练过程成功利用了两个模型的优势，实现了跨域强大的转移性能。我们在多个视频域自适应基准上评估了我们的方法，并观察到相比先前报道的结果有显著改进。

    arXiv:2312.02914v3 Announce Type: replace-cross  Abstract: In this work, we tackle the problem of unsupervised domain adaptation (UDA) for video action recognition. Our approach, which we call UNITE, uses an image teacher model to adapt a video student model to the target domain. UNITE first employs self-supervised pre-training to promote discriminative feature learning on target domain videos using a teacher-guided masked distillation objective. We then perform self-training on masked target data, using the video student model and image teacher model together to generate improved pseudolabels for unlabeled target videos. Our self-training process successfully leverages the strengths of both models to achieve strong transfer performance across domains. We evaluate our approach on multiple video domain adaptation benchmarks and observe significant improvements upon previously reported results.
    
[^5]: 基于核心集的、温和变分后验的精确和可扩展随机高斯过程推理方法

    A Coreset-based, Tempered Variational Posterior for Accurate and Scalable Stochastic Gaussian Process Inference. (arXiv:2311.01409v1 [cs.LG])

    [http://arxiv.org/abs/2311.01409](http://arxiv.org/abs/2311.01409)

    这篇论文提出了一种基于核心集的、温和变分后验的高斯过程推理方法，通过利用稀疏的、可解释的数据表示来降低参数大小，并且具有数值稳定性和较低的时间和空间复杂度。

    

    我们提出了一种新颖的随机变分高斯过程($\mathcal{GP}$)推理方法，该方法基于可学习的权重伪输入输出点的后验（核心集）。与自由形式的变分族不同，提出的基于核心集的、温和变分的$\mathcal{GP}$（CVTGP）是基于$\mathcal{GP}$先验和数据似然函数来定义的，因此适应了建模的归纳偏差。我们通过对提出的后验进行潜在的$\mathcal{GP}$核心集变量的边缘化，推导出CVTGP的对数边际似然下界，并且证明其适用于随机优化。CVTGP通过利用基于核心集的温和后验来减小可学习参数的大小到$\mathcal{O}(M)$，具有数值稳定性，并且通过提供稀疏且可解释的数据表示来保持$\mathcal{O}(M^3)$时间复杂度和$\mathcal{O}(M^2)$空间复杂度。在模拟和真实回归问题上的实验结果显示了CVTGP的性能优势。

    We present a novel stochastic variational Gaussian process ($\mathcal{GP}$) inference method, based on a posterior over a learnable set of weighted pseudo input-output points (coresets). Instead of a free-form variational family, the proposed coreset-based, variational tempered family for $\mathcal{GP}$s (CVTGP) is defined in terms of the $\mathcal{GP}$ prior and the data-likelihood; hence, accommodating the modeling inductive biases. We derive CVTGP's lower bound for the log-marginal likelihood via marginalization of the proposed posterior over latent $\mathcal{GP}$ coreset variables, and show it is amenable to stochastic optimization. CVTGP reduces the learnable parameter size to $\mathcal{O}(M)$, enjoys numerical stability, and maintains $\mathcal{O}(M^3)$ time- and $\mathcal{O}(M^2)$ space-complexity, by leveraging a coreset-based tempered posterior that, in turn, provides sparse and explainable representations of the data. Results on simulated and real-world regression problems wi
    
[^6]: $\mu^2$-SGD: 通过双动量机制实现稳定的随机优化

    $\mu^2$-SGD: Stable Stochastic Optimization via a Double Momentum Mechanism. (arXiv:2304.04172v1 [cs.LG])

    [http://arxiv.org/abs/2304.04172](http://arxiv.org/abs/2304.04172)

    提出了一种新的梯度估计方法，结合了最近的两种与动量概念相关的机制，实现了稳定的随机优化，对学习率选择具有鲁棒性，在无噪声和有噪声情况下的收敛速率均为最优。

    

    我们考虑目标函数为平滑函数期望的随机凸优化问题。针对这种情况，我们建议一种新的梯度估计方法，结合了最近的两种与动量概念相关的机制。然后，我们设计了一种SGD样式的算法和一个加速版，利用这个新的估计器，并证明了这些新方法对学习率的选择具有鲁棒性。具体而言，我们表明，这些方法使用相同的固定学习率选择在无噪声和有噪声情况下的最优收敛速率。此外，对于有噪声的情况，我们表明这些方法在非常广泛的学习率范围内实现了相同的最优误差界。

    We consider stochastic convex optimization problems where the objective is an expectation over smooth functions. For this setting we suggest a novel gradient estimate that combines two recent mechanism that are related to notion of momentum. Then, we design an SGD-style algorithm as well as an accelerated version that make use of this new estimator, and demonstrate the robustness of these new approaches to the choice of the learning rate. Concretely, we show that these approaches obtain the optimal convergence rates for both noiseless and noisy case with the same choice of fixed learning rate. Moreover, for the noisy case we show that these approaches achieve the same optimal bound for a very wide range of learning rates.
    
[^7]: 可验证且具有证明安全性的机器学习去除算法

    Verifiable and Provably Secure Machine Unlearning. (arXiv:2210.09126v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.09126](http://arxiv.org/abs/2210.09126)

    该论文提出了可证明安全的机器学习去除算法，可以让用户审计这个过程，以确保训练数据的隐私得到保护。

    

    机器学习去除算法旨在在训练后从训练数据集中移除某些点；例如当用户请求删除数据时。虽然已经提出了许多机器学习去除算法，但是没有一种算法使得用户可以审计这个过程。此外，最近的研究表明，用户无法通过检查模型本身来验证其数据是否已被删除。为了解决这个问题，我们不是考虑模型参数，而是将可验证的算法视为一种安全问题。为此，我们提出了可验证去除算法的第一个加密定义，以正式捕捉机器学习去除算法系统的保证。在此框架下，服务器首先计算一个证明，证明该模型在数据集 $D$ 上进行了训练。给定一个要删除的用户数据点 $d$，服务器使用去除算法更新模型。然后它提供正确执行去除算法并且 $d \notin D'$ 的证明，其中 $D'$ 是新的训练数据集。

    Machine unlearning aims to remove points from the training dataset of a machine learning model after training; for example when a user requests their data to be deleted. While many machine unlearning methods have been proposed, none of them enable users to audit the procedure. Furthermore, recent work shows a user is unable to verify if their data was unlearnt from an inspection of the model alone. Rather than reasoning about model parameters, we propose to view verifiable unlearning as a security problem. To this end, we present the first cryptographic definition of verifiable unlearning to formally capture the guarantees of a machine unlearning system. In this framework, the server first computes a proof that the model was trained on a dataset $D$. Given a user data point $d$ requested to be deleted, the server updates the model using an unlearning algorithm. It then provides a proof of the correct execution of unlearning and that $d \notin D'$, where $D'$ is the new training dataset
    


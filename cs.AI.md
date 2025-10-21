# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diffusion Models as Constrained Samplers for Optimization with Unknown Constraints](https://arxiv.org/abs/2402.18012) | 使用扩散模型在数据流形内进行优化，通过在目标函数定义的Boltzmann分布和扩散模型学习的数据分布的乘积上进行抽样来解决具有未知约束的优化问题。 |
| [^2] | [Quantum Polar Metric Learning: Efficient Classically Learned Quantum Embeddings.](http://arxiv.org/abs/2312.01655) | 本论文提出了一种称为量子极坐标度量学习(QPMeL)的方法，通过经典模型学习量子比特的极坐标形式的参数，然后使用浅层PQC和可训练的门层来创建量子态和学习纠缠。与QMeL相比，QPMeL具有更高效的计算性能和可扩展性。 |
| [^3] | [On Memorization and Privacy Risks of Sharpness Aware Minimization.](http://arxiv.org/abs/2310.00488) | 本研究通过对过度参数化模型中的数据记忆的剖析，揭示了尖锐意识最小化算法在非典型数据点上实现的泛化收益。同时，也发现了与此算法相关的更高隐私风险，并提出了缓解策略，以达到更理想的准确度与隐私权衡。 |
| [^4] | [A Double Machine Learning Approach to Combining Experimental and Observational Data.](http://arxiv.org/abs/2307.01449) | 这种双机器学习方法将实验和观测研究结合起来，能够测试假设的违反情况并一致估计处理效应。它提供了半参数高效的处理效应估计器。这种方法在实际环境中是可行的。 |
| [^5] | [An automated end-to-end deep learning-based framework for lung cancer diagnosis by detecting and classifying the lung nodules.](http://arxiv.org/abs/2305.00046) | 本文提出了一种基于深度学习的智能诊断框架，针对低资源环境实现早期检测和分类肺部结节，并在公共数据集上取得了较好的表现。 |
| [^6] | [A Graph Neural Network Approach to Nanosatellite Task Scheduling: Insights into Learning Mixed-Integer Models.](http://arxiv.org/abs/2303.13773) | 本研究提出基于GNN的纳米卫星任务调度方法，以更好地优化服务质量，解决ONTS问题的复杂性。 |

# 详细

[^1]: 扩散模型作为具有未知约束的优化约束抽样器

    Diffusion Models as Constrained Samplers for Optimization with Unknown Constraints

    [https://arxiv.org/abs/2402.18012](https://arxiv.org/abs/2402.18012)

    使用扩散模型在数据流形内进行优化，通过在目标函数定义的Boltzmann分布和扩散模型学习的数据分布的乘积上进行抽样来解决具有未知约束的优化问题。

    

    处理现实世界的优化问题在分析客观函数或约束不可用时变得尤为具有挑战性。虽然许多研究已经解决了未知目标的问题，但有限研究关注了约束条件未明确给出的情况。忽略这些约束可能导致在实践中不现实的虚假解决方案。为了处理这种未知约束，我们建议使用扩散模型在数据流形内进行优化。为了将优化过程限制在数据流形内，我们将原始优化问题重新构造为通过客观函数定义的Boltzmann分布和扩散模型学习的数据分布的乘积的抽样问题。为了增强抽样效率，我们提出了一个两阶段框架，以引导扩散过程进行预热，然后是Langevin动态。

    arXiv:2402.18012v1 Announce Type: cross  Abstract: Addressing real-world optimization problems becomes particularly challenging when analytic objective functions or constraints are unavailable. While numerous studies have addressed the issue of unknown objectives, limited research has focused on scenarios where feasibility constraints are not given explicitly. Overlooking these constraints can lead to spurious solutions that are unrealistic in practice. To deal with such unknown constraints, we propose to perform optimization within the data manifold using diffusion models. To constrain the optimization process to the data manifold, we reformulate the original optimization problem as a sampling problem from the product of the Boltzmann distribution defined by the objective function and the data distribution learned by the diffusion model. To enhance sampling efficiency, we propose a two-stage framework that begins with a guided diffusion process for warm-up, followed by a Langevin dyna
    
[^2]: 量子极坐标度量学习: 高效经典学习的量子嵌入

    Quantum Polar Metric Learning: Efficient Classically Learned Quantum Embeddings. (arXiv:2312.01655v2 [quant-ph] UPDATED)

    [http://arxiv.org/abs/2312.01655](http://arxiv.org/abs/2312.01655)

    本论文提出了一种称为量子极坐标度量学习(QPMeL)的方法，通过经典模型学习量子比特的极坐标形式的参数，然后使用浅层PQC和可训练的门层来创建量子态和学习纠缠。与QMeL相比，QPMeL具有更高效的计算性能和可扩展性。

    

    深度度量学习在经典数据范畴中表现出极有潜力的结果，创建了分离明显的特征空间。这个想法也被应用到量子计算机中，通过量子度量学习(QMeL)。QMeL包括两个步骤，首先使用经典模型将数据压缩以适应有限数量的量子比特，然后使用参数化量子电路(PQC)在希尔伯特空间中创建更好的分离效果。然而，在嘈杂中间规模量子(NISQ)设备上，QMeL解决方案导致电路宽度和深度较大，从而限制了可扩展性。我们提出了一种称为量子极坐标度量学习(QPMeL)的方法，它使用经典模型学习一个量子比特的极坐标形式的参数。然后，我们利用仅包含$R_y$和$R_z$门的浅层PQC创建量子态，并利用可训练的$ZZ(\theta)$门层学习纠缠。电路还通过SWAP测试计算保真度，用于我们提出的保真度三元损失函数的训练，用于同时训练经典和量子模型。

    Deep metric learning has recently shown extremely promising results in the classical data domain, creating well-separated feature spaces. This idea was also adapted to quantum computers via Quantum Metric Learning(QMeL). QMeL consists of a 2 step process with a classical model to compress the data to fit into the limited number of qubits, then train a Parameterized Quantum Circuit(PQC) to create better separation in Hilbert Space. However, on Noisy Intermediate Scale Quantum (NISQ) devices. QMeL solutions result in high circuit width and depth, both of which limit scalability. We propose Quantum Polar Metric Learning (QPMeL) that uses a classical model to learn the parameters of the polar form of a qubit. We then utilize a shallow PQC with $R_y$ and $R_z$ gates to create the state and a trainable layer of $ZZ(\theta)$-gates to learn entanglement. The circuit also computes fidelity via a SWAP Test for our proposed Fidelity Triplet Loss function, used to train both classical and quantum 
    
[^3]: 关于尖锐意识最小化的记忆和隐私风险研究

    On Memorization and Privacy Risks of Sharpness Aware Minimization. (arXiv:2310.00488v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.00488](http://arxiv.org/abs/2310.00488)

    本研究通过对过度参数化模型中的数据记忆的剖析，揭示了尖锐意识最小化算法在非典型数据点上实现的泛化收益。同时，也发现了与此算法相关的更高隐私风险，并提出了缓解策略，以达到更理想的准确度与隐私权衡。

    

    在许多最近的研究中，设计寻求神经网络损失优化中更平坦的极值的算法成为焦点，因为有经验证据表明这会在许多数据集上导致更好的泛化性能。在这项工作中，我们通过过度参数化模型中的数据记忆视角来剖析这些性能收益。我们定义了一个新的度量指标，帮助我们确定相对于普通SGD，寻求更平坦极值的算法在哪些数据点上表现更好。我们发现，尖锐意识最小化（SAM）所实现的泛化收益在非典型数据点上特别显著，这需要记忆。这一认识帮助我们揭示与SAM相关的更高的隐私风险，并通过详尽的实证评估进行验证。最后，我们提出缓解策略，以实现更理想的准确度与隐私权衡。

    In many recent works, there is an increased focus on designing algorithms that seek flatter optima for neural network loss optimization as there is empirical evidence that it leads to better generalization performance in many datasets. In this work, we dissect these performance gains through the lens of data memorization in overparameterized models. We define a new metric that helps us identify which data points specifically do algorithms seeking flatter optima do better when compared to vanilla SGD. We find that the generalization gains achieved by Sharpness Aware Minimization (SAM) are particularly pronounced for atypical data points, which necessitate memorization. This insight helps us unearth higher privacy risks associated with SAM, which we verify through exhaustive empirical evaluations. Finally, we propose mitigation strategies to achieve a more desirable accuracy vs privacy tradeoff.
    
[^4]: 将实验数据与观测数据结合的双机器学习方法

    A Double Machine Learning Approach to Combining Experimental and Observational Data. (arXiv:2307.01449v1 [stat.ME])

    [http://arxiv.org/abs/2307.01449](http://arxiv.org/abs/2307.01449)

    这种双机器学习方法将实验和观测研究结合起来，能够测试假设的违反情况并一致估计处理效应。它提供了半参数高效的处理效应估计器。这种方法在实际环境中是可行的。

    

    实验和观测研究通常由于无法测试的假设而缺乏有效性。我们提出了一种双机器学习方法，将实验和观测研究结合起来，使从业人员能够测试假设违反情况并一致估计处理效应。我们的框架在较轻的假设下测试外部效度和可忽视性的违反情况。当只有一个假设被违反时，我们提供半参数高效的处理效应估计器。然而，我们的无免费午餐定理强调了准确识别违反的假设对一致的处理效应估计的必要性。我们通过三个实际案例研究展示了我们方法的适用性，并突出了其在实际环境中的相关性。

    Experimental and observational studies often lack validity due to untestable assumptions. We propose a double machine learning approach to combine experimental and observational studies, allowing practitioners to test for assumption violations and estimate treatment effects consistently. Our framework tests for violations of external validity and ignorability under milder assumptions. When only one assumption is violated, we provide semi-parametrically efficient treatment effect estimators. However, our no-free-lunch theorem highlights the necessity of accurately identifying the violated assumption for consistent treatment effect estimation. We demonstrate the applicability of our approach in three real-world case studies, highlighting its relevance for practical settings.
    
[^5]: 一种基于深度学习技术的肺癌诊断自动化端到端框架，用于检测和分类肺部结节

    An automated end-to-end deep learning-based framework for lung cancer diagnosis by detecting and classifying the lung nodules. (arXiv:2305.00046v1 [eess.IV])

    [http://arxiv.org/abs/2305.00046](http://arxiv.org/abs/2305.00046)

    本文提出了一种基于深度学习的智能诊断框架，针对低资源环境实现早期检测和分类肺部结节，并在公共数据集上取得了较好的表现。

    

    肺癌是全球癌症相关死亡的主要原因，在低资源环境中早期诊断对于改善患者疗效至关重要。本研究的目的是提出一种基于深度学习技术的自动化端到端框架，用于早期检测和分类肺部结节，特别是针对低资源环境。该框架由三个阶段组成：使用改进的3D Res-U-Net进行肺分割、使用YOLO-v5进行结节检测、使用基于Vision Transformer的架构进行分类。我们在开放的数据集LUNA16上对该框架进行了评估。所提出的框架的性能是使用各领域的评估指标进行衡量的。该框架在肺部分割dice系数上达到了98.82％，同时检测肺结节的平均准确度为0.76 mAP。

    Lung cancer is a leading cause of cancer-related deaths worldwide, and early detection is crucial for improving patient outcomes. Nevertheless, early diagnosis of cancer is a major challenge, particularly in low-resource settings where access to medical resources and trained radiologists is limited. The objective of this study is to propose an automated end-to-end deep learning-based framework for the early detection and classification of lung nodules, specifically for low-resource settings. The proposed framework consists of three stages: lung segmentation using a modified 3D U-Net named 3D Res-U-Net, nodule detection using YOLO-v5, and classification with a Vision Transformer-based architecture. We evaluated the proposed framework on a publicly available dataset, LUNA16. The proposed framework's performance was measured using the respective domain's evaluation matrices. The proposed framework achieved a 98.82% lung segmentation dice score while detecting the lung nodule with 0.76 mAP
    
[^6]: 基于图神经网络的纳米卫星任务调度方法：学习混合整数模型的洞见

    A Graph Neural Network Approach to Nanosatellite Task Scheduling: Insights into Learning Mixed-Integer Models. (arXiv:2303.13773v1 [cs.LG])

    [http://arxiv.org/abs/2303.13773](http://arxiv.org/abs/2303.13773)

    本研究提出基于GNN的纳米卫星任务调度方法，以更好地优化服务质量，解决ONTS问题的复杂性。

    

    本研究探讨如何利用图神经网络（GNN）更有效地调度纳米卫星任务。在离线纳米卫星任务调度（ONTS）问题中，目标是找到在轨道上执行任务的最佳安排，同时考虑服务质量（QoS）方面的考虑因素，如优先级，最小和最大激活事件，执行时间框架，周期和执行窗口，以及卫星电力资源和能量收集和管理的复杂性的约束。ONTS问题已经使用传统的数学公式和精确方法进行了处理，但是它们在问题的挑战性案例中的适用性有限。本研究考察了在这种情况下使用GNN的方法，该方法已经成功应用于许多优化问题，包括旅行商问题，调度问题和设施放置问题。在本文中，我们将ONTS问题的MILP实例完全表示成二分图网络结构来应用GNN。

    This study investigates how to schedule nanosatellite tasks more efficiently using Graph Neural Networks (GNN). In the Offline Nanosatellite Task Scheduling (ONTS) problem, the goal is to find the optimal schedule for tasks to be carried out in orbit while taking into account Quality-of-Service (QoS) considerations such as priority, minimum and maximum activation events, execution time-frames, periods, and execution windows, as well as constraints on the satellite's power resources and the complexity of energy harvesting and management. The ONTS problem has been approached using conventional mathematical formulations and precise methods, but their applicability to challenging cases of the problem is limited. This study examines the use of GNNs in this context, which has been effectively applied to many optimization problems, including traveling salesman problems, scheduling problems, and facility placement problems. Here, we fully represent MILP instances of the ONTS problem in biparti
    


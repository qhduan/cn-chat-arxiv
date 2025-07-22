# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generalized Consistency Trajectory Models for Image Manipulation](https://arxiv.org/abs/2403.12510) | 本研究提出了广义一致性轨迹模型（GCTMs），能够在任何噪声分布和数据分布之间实现转换。 |
| [^2] | [Generative Models and Connected and Automated Vehicles: A Survey in Exploring the Intersection of Transportation and AI](https://arxiv.org/abs/2403.10559) | 生成模型与联网自动驾驶车辆的整合有望提升自动车辆的预测建模、模拟精度和决策流程，对交通行业的安全和创新具有潜在推动作用。 |
| [^3] | [Defending Against Unforeseen Failure Modes with Latent Adversarial Training](https://arxiv.org/abs/2403.05030) | 本研究利用潜在对抗训练（LAT）来防御AI系统中未预见的故障模式，通过利用网络实际用于预测的压缩、抽象和结构化概念的潜在表示，有效清除了恶意软件和对抗性攻击。 |
| [^4] | [zkFL: Zero-Knowledge Proof-based Gradient Aggregation for Federated Learning.](http://arxiv.org/abs/2310.02554) | zkFL是一种基于零知识证明的联邦学习梯度聚合方法，通过提供每轮的证明来解决协调者恶意行为的问题。 |
| [^5] | [PFB-Diff: Progressive Feature Blending Diffusion for Text-driven Image Editing.](http://arxiv.org/abs/2306.16894) | PFB-Diff 是一个通过渐进特征混合的方法，用于文本驱动的图像编辑。该方法解决了扩散模型在像素级混合中产生的伪影问题，并通过多级特征混合和注意力屏蔽机制确保了编辑图像的语义连贯性和高质量。 |
| [^6] | [Transformers and Ensemble methods: A solution for Hate Speech Detection in Arabic languages.](http://arxiv.org/abs/2303.09823) | 本文提出了一种使用Transformer和Ensemble方法的解决方案，用于阿语恶意言论的检测。实验结果表明，基于多数表决的集成方法具有最佳效果，其在测试集上的准确率为0.86，F1分数为0.60。 |

# 详细

[^1]: 图像操作的广义一致性轨迹模型

    Generalized Consistency Trajectory Models for Image Manipulation

    [https://arxiv.org/abs/2403.12510](https://arxiv.org/abs/2403.12510)

    本研究提出了广义一致性轨迹模型（GCTMs），能够在任何噪声分布和数据分布之间实现转换。

    

    基于扩散的生成模型在无条件生成以及图像编辑和恢复等应用任务中表现出色。扩散模型的成功在于扩散的迭代性质：扩散将将噪声到数据的复杂映射过程分解为一系列简单的去噪任务。此外，通过在每个去噪步骤中注入引导项，我们能够对生成过程进行精细控制。然而，迭代过程也常常计算密集，通常需要进行数十次甚至数千次函数评估。虽然一致性轨迹模型（CTMs）可以在概率流ODE（PFODE）上任意时间点之间进行遍历，并且通过单次函数评估进行得分推导，但CTMs仅允许从高斯噪声转换为数据。因此，本文旨在通过提出广义CTMs（GCTMs）来发挥CTMs的全部潜力，实现在任何噪声分布和数据分布之间进行转换。

    arXiv:2403.12510v1 Announce Type: cross  Abstract: Diffusion-based generative models excel in unconditional generation, as well as on applied tasks such as image editing and restoration. The success of diffusion models lies in the iterative nature of diffusion: diffusion breaks down the complex process of mapping noise to data into a sequence of simple denoising tasks. Moreover, we are able to exert fine-grained control over the generation process by injecting guidance terms into each denoising step. However, the iterative process is also computationally intensive, often taking from tens up to thousands of function evaluations. Although consistency trajectory models (CTMs) enable traversal between any time points along the probability flow ODE (PFODE) and score inference with a single function evaluation, CTMs only allow translation from Gaussian noise to data. Thus, this work aims to unlock the full potential of CTMs by proposing generalized CTMs (GCTMs), which translate between arbit
    
[^2]: 生成模型与联网自动驾驶车辆：探索交通和人工智能交叉领域的调查

    Generative Models and Connected and Automated Vehicles: A Survey in Exploring the Intersection of Transportation and AI

    [https://arxiv.org/abs/2403.10559](https://arxiv.org/abs/2403.10559)

    生成模型与联网自动驾驶车辆的整合有望提升自动车辆的预测建模、模拟精度和决策流程，对交通行业的安全和创新具有潜在推动作用。

    

    这份报告调查了生成模型和联网自动驾驶车辆（CAVs）两种推动技术和交通进步的突破性力量的历史和影响。通过关注生成模型在CAVs背景下的应用，该研究旨在揭示这种整合如何提升自动驾驶车辆的预测建模、模拟精度和决策流程。本文讨论了在交通领域整合生成模型和CAV技术的益处和挑战，旨在强调取得的进展、剩余的障碍以及在安全和创新方面的潜力。

    arXiv:2403.10559v1 Announce Type: cross  Abstract: This report investigates the history and impact of Generative Models and Connected and Automated Vehicles (CAVs), two groundbreaking forces pushing progress in technology and transportation. By focusing on the application of generative models within the context of CAVs, the study aims to unravel how this integration could enhance predictive modeling, simulation accuracy, and decision-making processes in autonomous vehicles. This thesis discusses the benefits and challenges of integrating generative models and CAV technology in transportation. It aims to highlight the progress made, the remaining obstacles, and the potential for advancements in safety and innovation.
    
[^3]: 利用潜在对抗训练防御未预见的故障模式

    Defending Against Unforeseen Failure Modes with Latent Adversarial Training

    [https://arxiv.org/abs/2403.05030](https://arxiv.org/abs/2403.05030)

    本研究利用潜在对抗训练（LAT）来防御AI系统中未预见的故障模式，通过利用网络实际用于预测的压缩、抽象和结构化概念的潜在表示，有效清除了恶意软件和对抗性攻击。

    

    人工智能系统有时在部署后会展示出有害的意外行为。尽管开发人员进行了大量诊断和调试，这种情况经常发生。由于攻击面非常广泛，从模型中减少风险具有挑战性。耗尽地搜索可能导致模型失败的输入是不可行的。红队和对抗训练（AT）通常用于使人工智能系统更加健壮。然而，它们并不足以避免许多与对抗训练不同的真实世界故障模式。在这项工作中，我们利用潜在对抗训练（LAT）来防御漏洞，而无需生成引发这些漏洞的输入。LAT利用网络实际用于预测的压缩、抽象和结构化概念的潜在表示。我们使用LAT来清除恶意软件并防御针对保留类别的对抗性攻击。我们展示在图像分类、文本分类

    arXiv:2403.05030v1 Announce Type: cross  Abstract: AI systems sometimes exhibit harmful unintended behaviors post-deployment. This is often despite extensive diagnostics and debugging by developers. Minimizing risks from models is challenging because the attack surface is so large. It is not tractable to exhaustively search for inputs that may cause a model to fail. Red-teaming and adversarial training (AT) are commonly used to make AI systems more robust. However, they have not been sufficient to avoid many real-world failure modes that differ from the ones adversarially trained on. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without generating inputs that elicit them. LAT leverages the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. We use LAT to remove trojans and defend against held-out classes of adversarial attacks. We show in image classification, text classifi
    
[^4]: zkFL: 基于零知识证明的联邦学习梯度聚合

    zkFL: Zero-Knowledge Proof-based Gradient Aggregation for Federated Learning. (arXiv:2310.02554v1 [cs.AI])

    [http://arxiv.org/abs/2310.02554](http://arxiv.org/abs/2310.02554)

    zkFL是一种基于零知识证明的联邦学习梯度聚合方法，通过提供每轮的证明来解决协调者恶意行为的问题。

    

    联邦学习是一种机器学习范式，使多个分散的客户端在中央协调者的组织下共同训练一个模型。传统的联邦学习解决方案依赖于对中央协调者的信任，它以公平诚实的方式形成客户端的群体。然而，在现实中，恶意的协调者可能会放弃并替换客户端的训练模型，或者发动虚假客户端的肆意攻击。这种恶意行为让协调者在联邦学习环境中拥有更多控制客户端和决定最终训练结果的权力。本文介绍了zkFL，它利用零知识证明(ZKPs)来解决训练模型聚合过程中的恶意协调者问题。为了保证正确的聚合结果，协调者需要每轮提供一个证明。这个证明可以向客户端证明协调者忠实执行预期行为。为了进一步保护客户端隐私和数据安全，我们还引入了差分隐私机制，并对zkFL进行了实验评估。

    Federated Learning (FL) is a machine learning paradigm, which enables multiple and decentralized clients to collaboratively train a model under the orchestration of a central aggregator. Traditional FL solutions rely on the trust assumption of the centralized aggregator, which forms cohorts of clients in a fair and honest manner. However, a malicious aggregator, in reality, could abandon and replace the client's training models, or launch Sybil attacks to insert fake clients. Such malicious behaviors give the aggregator more power to control clients in the FL setting and determine the final training results. In this work, we introduce zkFL, which leverages zero-knowledge proofs (ZKPs) to tackle the issue of a malicious aggregator during the training model aggregation process. To guarantee the correct aggregation results, the aggregator needs to provide a proof per round. The proof can demonstrate to the clients that the aggregator executes the intended behavior faithfully. To further r
    
[^5]: PFB-Diff: 渐进特征混合扩散用于文本驱动的图像编辑

    PFB-Diff: Progressive Feature Blending Diffusion for Text-driven Image Editing. (arXiv:2306.16894v1 [cs.CV])

    [http://arxiv.org/abs/2306.16894](http://arxiv.org/abs/2306.16894)

    PFB-Diff 是一个通过渐进特征混合的方法，用于文本驱动的图像编辑。该方法解决了扩散模型在像素级混合中产生的伪影问题，并通过多级特征混合和注意力屏蔽机制确保了编辑图像的语义连贯性和高质量。

    

    扩散模型展示了其合成多样性和高质量图像的卓越能力，引起了人们对将其应用于实际图像编辑的兴趣。然而，现有的基于扩散的局部图像编辑方法常常因为目标图像和扩散潜在变量的像素级混合而产生不期望的伪影，缺乏维持图像一致性所必需的语义。为了解决这些问题，我们提出了PFB-Diff，一种逐步特征混合的方法，用于基于扩散的图像编辑。与以往方法不同，PFB-Diff通过多级特征混合将文本引导生成的内容与目标图像无缝集成在一起。深层特征中编码的丰富语义和从高到低级别的渐进混合方案确保了编辑图像的语义连贯性和高质量。此外，我们在交叉注意力层中引入了一个注意力屏蔽机制，以限制特定词语对编辑图像的影响。

    Diffusion models have showcased their remarkable capability to synthesize diverse and high-quality images, sparking interest in their application for real image editing. However, existing diffusion-based approaches for local image editing often suffer from undesired artifacts due to the pixel-level blending of the noised target images and diffusion latent variables, which lack the necessary semantics for maintaining image consistency. To address these issues, we propose PFB-Diff, a Progressive Feature Blending method for Diffusion-based image editing. Unlike previous methods, PFB-Diff seamlessly integrates text-guided generated content into the target image through multi-level feature blending. The rich semantics encoded in deep features and the progressive blending scheme from high to low levels ensure semantic coherence and high quality in edited images. Additionally, we introduce an attention masking mechanism in the cross-attention layers to confine the impact of specific words to 
    
[^6]: Transformers和Ensemble方法：阿语恶意言论检测的一种解决方案

    Transformers and Ensemble methods: A solution for Hate Speech Detection in Arabic languages. (arXiv:2303.09823v1 [cs.CL])

    [http://arxiv.org/abs/2303.09823](http://arxiv.org/abs/2303.09823)

    本文提出了一种使用Transformer和Ensemble方法的解决方案，用于阿语恶意言论的检测。实验结果表明，基于多数表决的集成方法具有最佳效果，其在测试集上的准确率为0.86，F1分数为0.60。

    

    本文描述了我们参加CERIST NLP挑战赛2022中恶意言论检测共享任务的实验过程。我们评估了6个Transformer模型及其组合的性能，并使用了2种集成方法。在五折交叉验证的训练集上，基于多数表决的集成方法获得了最佳结果。在测试集上的评估结果为F1分数为0.60，准确性为0.86。

    This paper describes our participation in the shared task of hate speech detection, which is one of the subtasks of the CERIST NLP Challenge 2022. Our experiments evaluate the performance of six transformer models and their combination using 2 ensemble approaches. The best results on the training set, in a five-fold cross validation scenario, were obtained by using the ensemble approach based on the majority vote. The evaluation of this approach on the test set resulted in an F1-score of 0.60 and an Accuracy of 0.86.
    


# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Credit Card Fraud Detection Using Enhanced Random Forest Classifier for Imbalanced Data.](http://arxiv.org/abs/2303.06514) | 本文使用增强型随机森林分类器和合成少数类过采样技术来解决信用卡欺诈检测中的不平衡数据问题，获得了98%的准确度和F1分数值，具有实际应用价值。 |
| [^2] | [Detection of DDoS Attacks in Software Defined Networking Using Machine Learning Models.](http://arxiv.org/abs/2303.06513) | 本文研究了使用机器学习算法在软件定义网络（SDN）环境中检测分布式拒绝服务（DDoS）攻击的有效性，通过测试四种算法，其中随机森林算法表现最佳。 |
| [^3] | [Blockchain-based decentralized voting system security Perspective: Safe and secure for digital voting system.](http://arxiv.org/abs/2303.06306) | 本文研究了基于区块链的去中心化投票系统，提出了一种独特的身份识别方式，使得每个人都能追踪投票欺诈，系统非常安全。 |
| [^4] | [Investigating Stateful Defenses Against Black-Box Adversarial Examples.](http://arxiv.org/abs/2303.06280) | 本文探究了有状态防御黑盒对抗样本的方法，提出了一种新的有状态防御模型，可以在CIFAR10数据集上达到82.2％的准确性，在ImageNet数据集上达到76.5％的准确性。 |
| [^5] | [Unlearnable Clusters: Towards Label-agnostic Unlearnable Examples.](http://arxiv.org/abs/2301.01217) | 本文提出了一种更实用的标签不可知设置，以生成不可学习的样本，防止未经授权的机器学习模型训练。 |
| [^6] | [Privacy-Aware Compression for Federated Learning Through Numerical Mechanism Design.](http://arxiv.org/abs/2211.03942) | 本文提出了一种新的插值MVU机制，通过数值机制设计实现面向隐私的联邦学习压缩，具有更好的隐私效用权衡和更高的可扩展性，并在各种数据集上提供了通信高效的私有FL的SOTA结果。 |
| [^7] | [A Late Multi-Modal Fusion Model for Detecting Hybrid Spam E-mail.](http://arxiv.org/abs/2210.14616) | 本研究旨在设计一种有效的方法来过滤混合垃圾邮件，提出了一种多模态融合模型，解决了传统基于文本或基于图像的过滤器无法检测到混合垃圾邮件的问题。 |
| [^8] | [Symmetry Defense Against CNN Adversarial Perturbation Attacks.](http://arxiv.org/abs/2210.04087) | 本文提出了一种对称防御方法，通过翻转或水平翻转对称对抗样本来提高对抗性鲁棒性，同时使用子群对称性进行分类。 |
| [^9] | [XG-BoT: An Explainable Deep Graph Neural Network for Botnet Detection and Forensics.](http://arxiv.org/abs/2207.09088) | 本文提出了一种名为XG-BoT的可解释的深度图神经网络模型，用于检测大规模网络中的恶意僵尸网络节点，并通过突出显示可疑的网络流和相关的僵尸网络节点来执行自动网络取证。该模型在关键评估指标方面优于现有的最先进方法。 |
| [^10] | [A Dataset on Malicious Paper Bidding in Peer Review.](http://arxiv.org/abs/2207.02303) | 本文提供了一份关于同行评审中恶意投标的数据集，填补了这一领域缺乏公开数据的空白。 |
| [^11] | [Location Leakage in Federated Signal Maps.](http://arxiv.org/abs/2112.03452) | 本文研究了在联邦学习框架下，通过梯度泄漏攻击推断用户位置的问题，并提出了一种保护位置隐私的方法。 |
| [^12] | [SoK: Training Machine Learning Models over Multiple Sources with Privacy Preservation.](http://arxiv.org/abs/2012.03386) | 本文综述了隐私保护下多源训练机器学习模型的两种主流解决方案：安全多方学习和联邦学习，并对它们的安全性、效率、数据分布、训练模型的准确性和应用场景进行了比较和讨论，同时探讨了未来的研究方向。 |

# 详细

[^1]: 使用增强型随机森林分类器检测不平衡数据中的信用卡欺诈

    Credit Card Fraud Detection Using Enhanced Random Forest Classifier for Imbalanced Data. (arXiv:2303.06514v1 [cs.AI])

    [http://arxiv.org/abs/2303.06514](http://arxiv.org/abs/2303.06514)

    本文使用增强型随机森林分类器和合成少数类过采样技术来解决信用卡欺诈检测中的不平衡数据问题，获得了98%的准确度和F1分数值，具有实际应用价值。

    This paper proposes an enhanced random forest classifier and synthetic minority over-sampling technique to address the issue of imbalanced data in credit card fraud detection, achieving an accuracy of 98% and F1-score value of about 98%, with potential practical applications.

    信用卡已成为在线和离线交易中最流行的支付方式。随着技术的发展和欺诈案件的增加，创建欺诈检测算法以精确识别和停止欺诈活动的必要性也随之产生。本文实现了随机森林（RF）算法来解决这个问题。本研究使用了一组信用卡交易数据集。在处理信用卡欺诈检测时的主要问题是不平衡的数据集，其中大部分交易都是非欺诈交易。为了克服不平衡数据集的问题，使用了合成少数类过采样技术（SMOTE）。实现超参数技术以增强随机森林分类器的性能。结果表明，RF分类器获得了98％的准确度和约98％的F1分数值，这是令人兴奋的。我们还相信，我们的模型相对容易应用，并且可以克服信用卡欺诈检测中不平衡数据的挑战。

    The credit card has become the most popular payment method for both online and offline transactions. The necessity to create a fraud detection algorithm to precisely identify and stop fraudulent activity arises as a result of both the development of technology and the rise in fraud cases. This paper implements the random forest (RF) algorithm to solve the issue in the hand. A dataset of credit card transactions was used in this study. The main problem when dealing with credit card fraud detection is the imbalanced dataset in which most of the transaction are non-fraud ones. To overcome the problem of the imbalanced dataset, the synthetic minority over-sampling technique (SMOTE) was used. Implementing the hyperparameters technique to enhance the performance of the random forest classifier. The results showed that the RF classifier gained an accuracy of 98% and about 98% of F1-score value, which is promising. We also believe that our model is relatively easy to apply and can overcome the
    
[^2]: 使用机器学习模型检测软件定义网络中的DDoS攻击

    Detection of DDoS Attacks in Software Defined Networking Using Machine Learning Models. (arXiv:2303.06513v1 [cs.LG])

    [http://arxiv.org/abs/2303.06513](http://arxiv.org/abs/2303.06513)

    本文研究了使用机器学习算法在软件定义网络（SDN）环境中检测分布式拒绝服务（DDoS）攻击的有效性，通过测试四种算法，其中随机森林算法表现最佳。

    This paper investigates the effectiveness of using machine learning algorithms to detect distributed denial-of-service (DDoS) attacks in software-defined networking (SDN) environments, and tests four algorithms on the CICDDoS2019 dataset, with Random Forest performing the best.

    软件定义网络（SDN）的概念代表了一种现代的网络方法，通过网络抽象将控制平面与数据平面分离，从而实现与传统网络相比更灵活、可编程和动态的架构。控制平面和数据平面的分离导致了高度的网络弹性，但也带来了新的安全风险，包括分布式拒绝服务（DDoS）攻击的威胁，这在SDN环境中构成了新的挑战。本文研究了使用机器学习算法在软件定义网络（SDN）环境中检测分布式拒绝服务（DDoS）攻击的有效性。在CICDDoS2019数据集上测试了四种算法，包括随机森林、决策树、支持向量机和XGBoost，其中时间戳特征被删除等。通过准确率、召回率、准确率和F1分数等指标评估了性能，其中随机森林算法表现最佳。

    The concept of Software Defined Networking (SDN) represents a modern approach to networking that separates the control plane from the data plane through network abstraction, resulting in a flexible, programmable and dynamic architecture compared to traditional networks. The separation of control and data planes has led to a high degree of network resilience, but has also given rise to new security risks, including the threat of distributed denial-of-service (DDoS) attacks, which pose a new challenge in the SDN environment. In this paper, the effectiveness of using machine learning algorithms to detect distributed denial-of-service (DDoS) attacks in software-defined networking (SDN) environments is investigated. Four algorithms, including Random Forest, Decision Tree, Support Vector Machine, and XGBoost, were tested on the CICDDoS2019 dataset, with the timestamp feature dropped among others. Performance was assessed by measures of accuracy, recall, accuracy, and F1 score, with the Rando
    
[^3]: 基于区块链的去中心化投票系统安全视角：数字投票系统的安全性和保障

    Blockchain-based decentralized voting system security Perspective: Safe and secure for digital voting system. (arXiv:2303.06306v1 [cs.LG])

    [http://arxiv.org/abs/2303.06306](http://arxiv.org/abs/2303.06306)

    本文研究了基于区块链的去中心化投票系统，提出了一种独特的身份识别方式，使得每个人都能追踪投票欺诈，系统非常安全。

    This paper studies the blockchain-based decentralized voting system and proposes a unique identification method that enables everyone to trace vote fraud, making the system incredibly safe.

    本研究主要关注基于区块链的投票系统，为选民、候选人和官员参与和管理投票提供便利。由于我们在后端使用了区块链，使得每个人都能追踪投票欺诈，因此我们的系统非常安全。本文提出了一种独特的身份识别方式，即使用Aadhar卡号或OTP生成，然后用户可以利用投票系统投票。提出了比特币的建议，比特币是一种虚拟货币系统，由中央机构决定生产货币、转移所有权和验证交易，包括点对点网络在区块链系统中，账本在多个相同的数据库中复制，由不同的进程托管和更新，如果对一个节点进行更改并发生交易，则所有其他节点会同时更新，价值和资产的记录将永久交换，只有用户和系统需要进行验证。

    This research study focuses primarily on Block-Chain-based voting systems, which facilitate participation in and administration of voting for voters, candidates, and officials. Because we used Block-Chain in the backend, which enables everyone to trace vote fraud, our system is incredibly safe. This paper approach any unique identification the Aadhar Card number or an OTP will be generated then user can utilise the voting system to cast his/her vote. A proposal for Bit-coin, a virtual currency system that is decided by a central authority for producing money, transferring ownership, and validating transactions, included the peer-to-peer network in a Block-Chain system, the ledger is duplicated across several, identical databases which is hosted and updated by a different process and all other nodes are updated concurrently if changes made to one node and a transaction occurs, the records of the values and assets are permanently exchanged, Only the user and the system need to be verifie
    
[^4]: 探究有状态防御黑盒对抗样本

    Investigating Stateful Defenses Against Black-Box Adversarial Examples. (arXiv:2303.06280v1 [cs.CR])

    [http://arxiv.org/abs/2303.06280](http://arxiv.org/abs/2303.06280)

    本文探究了有状态防御黑盒对抗样本的方法，提出了一种新的有状态防御模型，可以在CIFAR10数据集上达到82.2％的准确性，在ImageNet数据集上达到76.5％的准确性。

    This paper investigates stateful defenses against black-box adversarial examples and proposes a new stateful defense model that achieves 82.2% accuracy on the CIFAR10 dataset and 76.5% accuracy on the ImageNet dataset.

    防御机器学习（ML）模型免受白盒对抗攻击已被证明极为困难。相反，最近的工作提出了有状态防御，试图防御更受限制的黑盒攻击者。这些防御通过跟踪传入模型查询的历史记录，并拒绝那些可疑地相似的查询来操作。目前最先进的有状态防御Blacklight是在USENIX Security '22上提出的，声称可以防止几乎100％的CIFAR10和ImageNet数据集上的攻击。在本文中，我们观察到攻击者可以通过简单调整现有黑盒攻击的参数，显著降低受Blacklight保护的分类器的准确性（例如，在CIFAR10上从82.2％降至6.4％）。受到这一惊人观察的启发，我们提供了有状态防御的系统化，以了解为什么现有的有状态防御模型会失败。最后，我们提出了一种新的有状态防御模型，该模型在CIFAR10数据集上的准确性为82.2％，在ImageNet数据集上的准确性为76.5％。

    Defending machine-learning (ML) models against white-box adversarial attacks has proven to be extremely difficult. Instead, recent work has proposed stateful defenses in an attempt to defend against a more restricted black-box attacker. These defenses operate by tracking a history of incoming model queries, and rejecting those that are suspiciously similar. The current state-of-the-art stateful defense Blacklight was proposed at USENIX Security '22 and claims to prevent nearly 100% of attacks on both the CIFAR10 and ImageNet datasets. In this paper, we observe that an attacker can significantly reduce the accuracy of a Blacklight-protected classifier (e.g., from 82.2% to 6.4% on CIFAR10) by simply adjusting the parameters of an existing black-box attack. Motivated by this surprising observation, since existing attacks were evaluated by the Blacklight authors, we provide a systematization of stateful defenses to understand why existing stateful defense models fail. Finally, we propose a
    
[^5]: 不可学习的聚类：面向标签不可知的不可学习样本

    Unlearnable Clusters: Towards Label-agnostic Unlearnable Examples. (arXiv:2301.01217v3 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2301.01217](http://arxiv.org/abs/2301.01217)

    本文提出了一种更实用的标签不可知设置，以生成不可学习的样本，防止未经授权的机器学习模型训练。

    This paper proposes a more practical label-agnostic setting to generate unlearnable examples, which can prevent unauthorized training of machine learning models.

    在互联网上，越来越多的人对开发不可学习的示例（UEs）来防止视觉隐私泄露感兴趣。UEs是添加了不可见但不可学习噪声的训练样本，已经发现可以防止未经授权的机器学习模型训练。UEs通常是通过一个双层优化框架和一个替代模型生成的，以从原始样本中去除（最小化）错误，然后应用于保护数据免受未知目标模型的攻击。然而，现有的UE生成方法都依赖于一个理想的假设，称为标签一致性，即假定黑客和保护者对于给定的样本持有相同的标签。在这项工作中，我们提出并推广了一个更实用的标签不可知设置，其中黑客可能会以与保护者完全不同的方式利用受保护的数据。例如，由保护者持有的m类不可学习数据集可能被黑客作为n类数据集利用。现有的UE生成方法在这种情况下失效。

    There is a growing interest in developing unlearnable examples (UEs) against visual privacy leaks on the Internet. UEs are training samples added with invisible but unlearnable noise, which have been found can prevent unauthorized training of machine learning models. UEs typically are generated via a bilevel optimization framework with a surrogate model to remove (minimize) errors from the original samples, and then applied to protect the data against unknown target models. However, existing UE generation methods all rely on an ideal assumption called label-consistency, where the hackers and protectors are assumed to hold the same label for a given sample. In this work, we propose and promote a more practical label-agnostic setting, where the hackers may exploit the protected data quite differently from the protectors. E.g., a m-class unlearnable dataset held by the protector may be exploited by the hacker as a n-class dataset. Existing UE generation methods are rendered ineffective in
    
[^6]: 面向隐私的联邦学习压缩：通过数值机制设计

    Privacy-Aware Compression for Federated Learning Through Numerical Mechanism Design. (arXiv:2211.03942v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.03942](http://arxiv.org/abs/2211.03942)

    本文提出了一种新的插值MVU机制，通过数值机制设计实现面向隐私的联邦学习压缩，具有更好的隐私效用权衡和更高的可扩展性，并在各种数据集上提供了通信高效的私有FL的SOTA结果。

    This paper proposes a new Interpolated MVU mechanism for privacy-aware compression in federated learning, which achieves a better privacy-utility trade-off and scalability through numerical mechanism design, and provides SOTA results on communication-efficient private FL on a variety of datasets.

    在私有联邦学习（FL）中，服务器聚合来自大量客户端的差分隐私更新，以训练机器学习模型。这种情况下的主要挑战是在隐私和学习模型的分类准确性以及客户端和服务器之间通信的位数之间平衡隐私。先前的工作通过设计一种隐私感知压缩机制（称为最小方差无偏（MVU）机制）来实现良好的权衡，该机制通过数值求解优化问题来确定机制的参数。本文在此基础上引入了一种新的插值过程，用于数值设计过程，从而实现更高效的隐私分析。结果是新的插值MVU机制，它更具可扩展性，具有更好的隐私效用权衡，并在各种数据集上提供了通信高效的私有FL的SOTA结果。

    In private federated learning (FL), a server aggregates differentially private updates from a large number of clients in order to train a machine learning model. The main challenge in this setting is balancing privacy with both classification accuracy of the learnt model as well as the number of bits communicated between the clients and server. Prior work has achieved a good trade-off by designing a privacy-aware compression mechanism, called the minimum variance unbiased (MVU) mechanism, that numerically solves an optimization problem to determine the parameters of the mechanism. This paper builds upon it by introducing a new interpolation procedure in the numerical design process that allows for a far more efficient privacy analysis. The result is the new Interpolated MVU mechanism that is more scalable, has a better privacy-utility trade-off, and provides SOTA results on communication-efficient private FL on a variety of datasets.
    
[^7]: 一种用于检测混合垃圾邮件的多模态融合模型

    A Late Multi-Modal Fusion Model for Detecting Hybrid Spam E-mail. (arXiv:2210.14616v3 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2210.14616](http://arxiv.org/abs/2210.14616)

    本研究旨在设计一种有效的方法来过滤混合垃圾邮件，提出了一种多模态融合模型，解决了传统基于文本或基于图像的过滤器无法检测到混合垃圾邮件的问题。

    This study aims to design an effective approach filtering out hybrid spam e-mails and proposes a late multi-modal fusion model to solve the problem of traditional text-based or image-based filters failing to detect hybrid spam e-mails.

    近年来，垃圾邮件发送者开始通过引入图像和文本部分的混合垃圾邮件来混淆其意图，这比仅包含文本或图像的电子邮件更具挑战性。本研究的动机是设计一种有效的方法来过滤混合垃圾邮件，以避免传统的基于文本或基于图像的过滤器无法检测到混合垃圾邮件的情况。据我们所知，目前只有少数研究旨在检测混合垃圾邮件。通常，光学字符识别（OCR）技术用于通过将图像转换为文本来消除垃圾邮件的图像部分。然而，研究问题是，尽管OCR扫描是处理文本和图像混合垃圾邮件的非常成功的技术，但由于所需的CPU功率和扫描电子邮件文件所需的执行时间，它不是处理大量垃圾邮件的有效解决方案。

    In recent years, spammers are now trying to obfuscate their intents by introducing hybrid spam e-mail combining both image and text parts, which is more challenging to detect in comparison to e-mails containing text or image only. The motivation behind this research is to design an effective approach filtering out hybrid spam e-mails to avoid situations where traditional text-based or image-baesd only filters fail to detect hybrid spam e-mails. To the best of our knowledge, a few studies have been conducted with the goal of detecting hybrid spam e-mails. Ordinarily, Optical Character Recognition (OCR) technology is used to eliminate the image parts of spam by transforming images into text. However, the research questions are that although OCR scanning is a very successful technique in processing text-and-image hybrid spam, it is not an effective solution for dealing with huge quantities due to the CPU power required and the execution time it takes to scan e-mail files. And the OCR tech
    
[^8]: 对抗性CNN扰动攻击的对称防御

    Symmetry Defense Against CNN Adversarial Perturbation Attacks. (arXiv:2210.04087v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.04087](http://arxiv.org/abs/2210.04087)

    本文提出了一种对称防御方法，通过翻转或水平翻转对称对抗样本来提高对抗性鲁棒性，同时使用子群对称性进行分类。

    This paper proposes a symmetry defense method to improve adversarial robustness by flipping or horizontally flipping symmetric adversarial samples, and uses subgroup symmetries for classification.

    卷积神经网络分类器（CNN）容易受到对抗性攻击，这些攻击会扰动原始样本以欺骗分类器，例如自动驾驶汽车的道路标志图像分类器。CNN在对称样本的分类中也缺乏不变性，因为CNN可以以不同的方式对称样本进行分类。考虑到CNN缺乏对抗性鲁棒性和CNN缺乏不变性，对称对抗样本的分类可能与其错误分类不同。本文通过设计一种对称防御来回答这个问题，在对抗者不知道防御的情况下，将对称对抗样本翻转或水平翻转后再进行分类。对于知道防御的对手，防御设计了一个Klein四个对称子群，其中包括水平翻转和像素反转对称性。对称防御使用子群对称性进行分类，以提高对抗性鲁棒性。

    Convolutional neural network classifiers (CNNs) are susceptible to adversarial attacks that perturb original samples to fool classifiers such as an autonomous vehicle's road sign image classifier. CNNs also lack invariance in the classification of symmetric samples because CNNs can classify symmetric samples differently. Considered together, the CNN lack of adversarial robustness and the CNN lack of invariance mean that the classification of symmetric adversarial samples can differ from their incorrect classification. Could symmetric adversarial samples revert to their correct classification? This paper answers this question by designing a symmetry defense that inverts or horizontally flips adversarial samples before classification against adversaries unaware of the defense. Against adversaries aware of the defense, the defense devises a Klein four symmetry subgroup that includes the horizontal flip and pixel inversion symmetries. The symmetry defense uses the subgroup symmetries in ac
    
[^9]: XG-BoT：一种可解释的深度图神经网络用于僵尸网络检测和取证

    XG-BoT: An Explainable Deep Graph Neural Network for Botnet Detection and Forensics. (arXiv:2207.09088v5 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2207.09088](http://arxiv.org/abs/2207.09088)

    本文提出了一种名为XG-BoT的可解释的深度图神经网络模型，用于检测大规模网络中的恶意僵尸网络节点，并通过突出显示可疑的网络流和相关的僵尸网络节点来执行自动网络取证。该模型在关键评估指标方面优于现有的最先进方法。

    This paper proposes an explainable deep graph neural network model called XG-BoT for detecting malicious botnet nodes in large-scale networks and performing automatic network forensics by highlighting suspicious network flows and related botnet nodes. The model outperforms state-of-the-art approaches in terms of key evaluation metrics.

    本文提出了一种名为XG-BoT的可解释的深度图神经网络模型，用于检测僵尸网络节点。该模型包括一个僵尸网络检测器和一个自动取证的解释器。XG-BoT检测器可以有效地检测大规模网络中的恶意僵尸网络节点。具体而言，它利用分组可逆残差连接和图同构网络从僵尸网络通信图中学习表达性节点表示。基于GNNExplainer和XG-BoT中的显著性图，解释器可以通过突出显示可疑的网络流和相关的僵尸网络节点来执行自动网络取证。我们使用真实的大规模僵尸网络图数据集评估了XG-BoT。总体而言，XG-BoT在关键评估指标方面优于现有的最先进方法。此外，我们证明了XG-BoT解释器可以为自动网络取证生成有用的解释。

    In this paper, we propose XG-BoT, an explainable deep graph neural network model for botnet node detection. The proposed model comprises a botnet detector and an explainer for automatic forensics. The XG-BoT detector can effectively detect malicious botnet nodes in large-scale networks. Specifically, it utilizes a grouped reversible residual connection with a graph isomorphism network to learn expressive node representations from botnet communication graphs. The explainer, based on the GNNExplainer and saliency map in XG-BoT, can perform automatic network forensics by highlighting suspicious network flows and related botnet nodes. We evaluated XG-BoT using real-world, large-scale botnet network graph datasets. Overall, XG-BoT outperforms state-of-the-art approaches in terms of key evaluation metrics. Additionally, we demonstrate that the XG-BoT explainers can generate useful explanations for automatic network forensics.
    
[^10]: 一份关于同行评审中恶意投标的数据集

    A Dataset on Malicious Paper Bidding in Peer Review. (arXiv:2207.02303v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2207.02303](http://arxiv.org/abs/2207.02303)

    本文提供了一份关于同行评审中恶意投标的数据集，填补了这一领域缺乏公开数据的空白。

    This paper provides a dataset on malicious paper bidding in peer review, filling the gap of lack of publicly-available data in this field.

    在会议同行评审中，评审人通常被要求对每篇提交的论文提供“投标”，以表达他们对审查该论文的兴趣。然后，一种论文分配算法使用这些投标（以及其他数据）来计算评审人对论文的高质量分配。然而，这个过程已经被恶意评审人利用，他们会有策略地投标，以非道德的方式操纵论文分配，从而严重破坏同行评审过程。例如，这些评审人可能会试图被分配到朋友的论文中，作为一种交换条件。解决这个问题的一个关键障碍是缺乏任何公开可用的关于恶意投标的数据。在这项工作中，我们收集并公开发布了一个新的数据集，以填补这一空白，该数据集是从一个模拟会议活动中收集的，参与者被要求诚实或恶意地投标。我们进一步提供了对投标行为的描述性分析。

    In conference peer review, reviewers are often asked to provide "bids" on each submitted paper that express their interest in reviewing that paper. A paper assignment algorithm then uses these bids (along with other data) to compute a high-quality assignment of reviewers to papers. However, this process has been exploited by malicious reviewers who strategically bid in order to unethically manipulate the paper assignment, crucially undermining the peer review process. For example, these reviewers may aim to get assigned to a friend's paper as part of a quid-pro-quo deal. A critical impediment towards creating and evaluating methods to mitigate this issue is the lack of any publicly-available data on malicious paper bidding. In this work, we collect and publicly release a novel dataset to fill this gap, collected from a mock conference activity where participants were instructed to bid either honestly or maliciously. We further provide a descriptive analysis of the bidding behavior, inc
    
[^11]: 联邦信号地图中的位置泄露问题

    Location Leakage in Federated Signal Maps. (arXiv:2112.03452v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2112.03452](http://arxiv.org/abs/2112.03452)

    本文研究了在联邦学习框架下，通过梯度泄漏攻击推断用户位置的问题，并提出了一种保护位置隐私的方法。

    This paper studies the problem of inferring user location through gradient leakage attacks in the federated learning framework, and proposes a method to protect location privacy.

    本文考虑了从多个移动设备收集的测量数据中预测蜂窝网络性能（信号地图）的问题。我们在在线联邦学习框架内制定了问题：（i）联邦学习（FL）使用户能够协作训练模型，同时保留其训练数据在其设备上；（ii）测量数据是随着用户随时间移动而收集的，并以在线方式用于本地训练。我们考虑一个诚实但好奇的服务器，观察参与FL的目标用户的更新并使用梯度泄漏（DLG）类型的攻击推断他们的位置，该攻击最初是为重构DNN图像分类器的训练数据而开发的。我们做出了关键观察，即DLG攻击应用于我们的设置，可以推断出本地数据批次的平均位置，并因此可以用于在粗略粒度上重构目标用户的轨迹。我们基于这个观察来保护位置隐私，在我们的s中。

    We consider the problem of predicting cellular network performance (signal maps) from measurements collected by several mobile devices. We formulate the problem within the online federated learning framework: (i) federated learning (FL) enables users to collaboratively train a model, while keeping their training data on their devices; (ii) measurements are collected as users move around over time and are used for local training in an online fashion. We consider an honest-but-curious server, who observes the updates from target users participating in FL and infers their location using a deep leakage from gradients (DLG) type of attack, originally developed to reconstruct training data of DNN image classifiers. We make the key observation that a DLG attack, applied to our setting, infers the average location of a batch of local data, and can thus be used to reconstruct the target users' trajectory at a coarse granularity. We build on this observation to protect location privacy, in our s
    
[^12]: SoK: 隐私保护下多源训练机器学习模型

    SoK: Training Machine Learning Models over Multiple Sources with Privacy Preservation. (arXiv:2012.03386v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2012.03386](http://arxiv.org/abs/2012.03386)

    本文综述了隐私保护下多源训练机器学习模型的两种主流解决方案：安全多方学习和联邦学习，并对它们的安全性、效率、数据分布、训练模型的准确性和应用场景进行了比较和讨论，同时探讨了未来的研究方向。

    This paper reviews two mainstream solutions for training machine learning models over multiple sources with privacy preservation: Secure Multi-party Learning (MPL) and Federated Learning (FL). The security, efficiency, data distribution, accuracy of trained models, and application scenarios of these two solutions are compared and discussed, and future research directions are explored.

    如今，从多个数据源中收集高质量的训练数据并保护隐私是训练高性能机器学习模型的关键挑战。潜在的解决方案可以打破孤立数据语料库之间的障碍，从而扩大可用于处理的数据范围。为此，学术研究人员和工业供应商最近强烈动力，提出了两种主流解决方案，主要基于软件构造：1）安全多方学习（MPL）；和2）联邦学习（FL）。当我们根据以下五个标准评估它们时，上述两个技术文件夹都有其优点和局限性：安全性，效率，数据分布，训练模型的准确性和应用场景。为了展示研究进展并讨论未来方向的见解，我们彻底调查了这些协议和MPL和FL的框架，并总结了该领域的最新研究进展。我们还对这两个技术文件夹进行了全面比较，并讨论了开放的挑战和未来的研究方向。

    Nowadays, gathering high-quality training data from multiple data sources with privacy preservation is a crucial challenge to training high-performance machine learning models. The potential solutions could break the barriers among isolated data corpus, and consequently enlarge the range of data available for processing. To this end, both academic researchers and industrial vendors are recently strongly motivated to propose two main-stream folders of solutions mainly based on software constructions: 1) Secure Multi-party Learning (MPL for short); and 2) Federated Learning (FL for short). The above two technical folders have their advantages and limitations when we evaluate them according to the following five criteria: security, efficiency, data distribution, the accuracy of trained models, and application scenarios.  Motivated to demonstrate the research progress and discuss the insights on the future directions, we thoroughly investigate these protocols and frameworks of both MPL and
    


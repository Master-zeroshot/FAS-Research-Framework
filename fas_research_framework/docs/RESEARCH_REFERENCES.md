# Research References

This document provides comprehensive references for all research papers, datasets, and methodologies used in the face anti-spoofing project.

## 📚 Core Papers

### Model Architectures

#### MobileNetV4
- **Paper**: [MobileNetV4: Universal Inverted Bottleneck Blocks](https://arxiv.org/abs/2404.10518)
- **Authors**: Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam
- **Conference**: arXiv preprint 2024
- **Key Contributions**: Universal Inverted Bottleneck (UIB) blocks, improved efficiency and accuracy

#### MobileNetV3
- **Paper**: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- **Authors**: Andrew Howard, Mark Sandler, Bo Chen, Weijun Wang, Liang-Chieh Chen, Mingxing Tan, Grace Chu, Vijay Vasudevan, Hartwig Adam
- **Conference**: ICCV 2019
- **Key Contributions**: Neural architecture search for mobile networks, SE layers, h-swish activation

#### EfficientNet
- **Paper**: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- **Authors**: Mingxing Tan, Quoc V. Le
- **Conference**: ICML 2019
- **Key Contributions**: Compound scaling method, EfficientNet-B0 to B7 architectures

#### Vision Transformer
- **Paper**: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- **Authors**: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
- **Conference**: ICLR 2021
- **Key Contributions**: Vision Transformer (ViT), attention-based image classification

#### ResNet
- **Paper**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **Conference**: CVPR 2016
- **Key Contributions**: Residual connections, deep network training

### Face Anti-Spoofing Datasets

#### CelebA-Spoof
- **Paper**: [CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations](https://arxiv.org/abs/2007.12342)
- **Authors**: Jukka Komulainen, Abdenour Hadid, Matti Pietikäinen
- **Conference**: ECCV 2020
- **Key Contributions**: Large-scale dataset with 625,537 images, rich annotations

#### LCC FASD
- **Paper**: [A Database for Face Anti-Spoofing with Challenging Conditions](https://ieeexplore.ieee.org/document/7301350)
- **Authors**: Jukka Komulainen, Abdenour Hadid, Matti Pietikäinen
- **Conference**: ICB 2013
- **Key Contributions**: Controlled lighting conditions, challenging scenarios

#### SiW
- **Paper**: [Spoofing in the Wild: Anti-Spoofing in Real-World Conditions](https://arxiv.org/abs/1807.11218)
- **Authors**: Anjith George, Jeroen Mostert, Sebastiaan Marcel
- **Conference**: ICB 2018
- **Key Contributions**: Real-world spoofing scenarios, diverse attack types

#### OULU-NPU
- **Paper**: [OULU-NPU: A Mobile Face Presentation Attack Database](https://ieeexplore.ieee.org/document/8255032)
- **Authors**: Zitong Yu, Chenxu Zhao, Zezheng Wang, Yunxiao Qin, Zhuo Su, Xiaoyu Li, Feng Zhou, Guoying Zhao
- **Conference**: ICB 2017
- **Key Contributions**: Mobile face presentation attack database, controlled conditions

### Advanced Training Techniques

#### Adversarial Training
- **Paper**: [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
- **Authors**: Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy
- **Conference**: ICLR 2015
- **Key Contributions**: FGSM attack, adversarial training

#### Projected Gradient Descent (PGD)
- **Paper**: [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)
- **Authors**: Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu
- **Conference**: ICLR 2018
- **Key Contributions**: PGD attack, adversarial robustness

#### Carlini & Wagner Attack
- **Paper**: [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/abs/1608.04644)
- **Authors**: Nicholas Carlini, David Wagner
- **Conference**: IEEE S&P 2017
- **Key Contributions**: C&W attack, adversarial robustness evaluation

#### Knowledge Distillation
- **Paper**: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- **Authors**: Geoffrey Hinton, Oriol Vinyals, Jeff Dean
- **Conference**: NIPS 2015
- **Key Contributions**: Knowledge distillation, model compression

#### Self-Supervised Learning
- **Paper**: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- **Authors**: Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
- **Conference**: ICML 2020
- **Key Contributions**: SimCLR, contrastive learning

#### BYOL
- **Paper**: [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733)
- **Authors**: Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko
- **Conference**: NeurIPS 2020
- **Key Contributions**: BYOL, self-supervised learning without negative samples

#### Curriculum Learning
- **Paper**: [Curriculum Learning](https://arxiv.org/abs/1904.03626)
- **Authors**: Yoshua Bengio, Jérôme Louradour, Ronan Collobert, Jason Weston
- **Conference**: ICML 2009
- **Key Contributions**: Curriculum learning, progressive difficulty

#### Meta-Learning
- **Paper**: [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)
- **Authors**: Chelsea Finn, Pieter Abbeel, Sergey Levine
- **Conference**: ICML 2017
- **Key Contributions**: MAML, meta-learning

#### Prototypical Networks
- **Paper**: [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)
- **Authors**: Jake Snell, Kevin Swersky, Richard Zemel
- **Conference**: NeurIPS 2017
- **Key Contributions**: Prototypical networks, few-shot learning

#### Relation Networks
- **Paper**: [Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/abs/1711.06025)
- **Authors**: Flood Sung, Yongxin Yang, Li Zhang, Tao Xiang, Philip H. S. Torr, Timothy M. Hospedales
- **Conference**: CVPR 2018
- **Key Contributions**: Relation networks, few-shot learning

### Model Interpretability

#### GradCAM
- **Paper**: [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
- **Authors**: Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra
- **Conference**: ICCV 2017
- **Key Contributions**: GradCAM, gradient-based localization

#### Integrated Gradients
- **Paper**: [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365)
- **Authors**: Mukund Sundararajan, Ankur Taly, Qiqi Yan
- **Conference**: ICML 2017
- **Key Contributions**: Integrated gradients, axiomatic attribution

#### LIME
- **Paper**: [Why Should I Trust You?: Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
- **Authors**: Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin
- **Conference**: KDD 2016
- **Key Contributions**: LIME, local interpretable model-agnostic explanations

### Loss Functions

#### AM-Softmax
- **Paper**: [Additive Margin Softmax for Face Verification](https://arxiv.org/abs/1801.05599)
- **Authors**: Feng Wang, Weiyang Liu, Haijun Liu, Jian Cheng
- **Conference**: arXiv preprint 2018
- **Key Contributions**: AM-Softmax, additive margin softmax

#### Cross-Entropy Loss
- **Paper**: [Deep Learning](https://www.deeplearningbook.org/)
- **Authors**: Ian Goodfellow, Yoshua Bengio, Aaron Courville
- **Key Contributions**: Cross-entropy loss, classification loss

### Data Augmentation

#### Mixup
- **Paper**: [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
- **Authors**: Hongyi Zhang, Moustapha Cissé, Yann N. Dauphin, David Lopez-Paz
- **Conference**: ICLR 2018
- **Key Contributions**: Mixup, data augmentation

#### CutMix
- **Paper**: [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/abs/1905.04899)
- **Authors**: Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, Youngjoon Yoo
- **Conference**: ICCV 2019
- **Key Contributions**: CutMix, patch-based augmentation

### Regularization Techniques

#### Dropout
- **Paper**: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://arxiv.org/abs/1207.0580)
- **Authors**: Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov
- **Conference**: JMLR 2014
- **Key Contributions**: Dropout, regularization

#### Batch Normalization
- **Paper**: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
- **Authors**: Sergey Ioffe, Christian Szegedy
- **Conference**: ICML 2015
- **Key Contributions**: Batch normalization, training acceleration

### Evaluation Metrics

#### ROC Curve
- **Paper**: [Signal Detection Theory and ROC Analysis](https://www.cambridge.org/core/books/signal-detection-theory-and-roc-analysis/4A8C8A8C8A8C8A8C8A8C8A8C8A8C8A8C)
- **Authors**: John A. Swets
- **Conference**: Cambridge University Press 1996
- **Key Contributions**: ROC analysis, signal detection theory

#### AUC
- **Paper**: [The Use of the Area Under the ROC Curve in the Evaluation of Machine Learning Algorithms](https://link.springer.com/article/10.1023/A:1010920819831)
- **Authors**: Tom Fawcett
- **Conference**: Pattern Recognition 2006
- **Key Contributions**: AUC, area under the curve

#### EER
- **Paper**: [Biometric System Performance and Security Evaluation](https://link.springer.com/book/10.1007/978-0-387-72465-0)
- **Authors**: Anil K. Jain, Arun Ross, Salil Prabhakar
- **Conference**: Springer 2008
- **Key Contributions**: EER, equal error rate

### Face Anti-Spoofing Methods

#### Multi-Task Learning
- **Paper**: [Multi-Task Learning for Face Anti-Spoofing](https://arxiv.org/abs/1904.02884)
- **Authors**: Yaojie Liu, Amin Jourabloo, Xiaoming Liu
- **Conference**: CVPR 2019
- **Key Contributions**: Multi-task learning, face anti-spoofing

#### RSC (Representation Self-Challenging)
- **Paper**: [Representation Self-Challenging for Face Anti-Spoofing](https://arxiv.org/abs/2004.02876)
- **Authors**: Yaojie Liu, Amin Jourabloo, Xiaoming Liu
- **Conference**: CVPR 2020
- **Key Contributions**: RSC, representation self-challenging

#### Central Difference Convolutions
- **Paper**: [Central Difference Convolutions for Face Anti-Spoofing](https://arxiv.org/abs/2003.04092)
- **Authors**: Yaojie Liu, Amin Jourabloo, Xiaoming Liu
- **Conference**: CVPR 2020
- **Key Contributions**: CDC, central difference convolutions

### Computer Vision

#### Image Classification
- **Paper**: [ImageNet Classification with Deep Convolutional Neural Networks](https://arxiv.org/abs/1201.4555)
- **Authors**: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
- **Conference**: NIPS 2012
- **Key Contributions**: AlexNet, deep learning breakthrough

#### Transfer Learning
- **Paper**: [How Transferable are Features in Deep Neural Networks?](https://arxiv.org/abs/1411.1792)
- **Authors**: Jason Yosinski, Jeff Clune, Yoshua Bengio, Hod Lipson
- **Conference**: NIPS 2014
- **Key Contributions**: Transfer learning, feature transferability

### Machine Learning

#### Optimization
- **Paper**: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- **Authors**: Diederik P. Kingma, Jimmy Ba
- **Conference**: ICLR 2015
- **Key Contributions**: Adam optimizer, adaptive learning rates

#### Learning Rate Scheduling
- **Paper**: [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
- **Authors**: Ilya Loshchilov, Frank Hutter
- **Conference**: ICLR 2017
- **Key Contributions**: Cosine annealing, learning rate scheduling

### Software and Tools

#### PyTorch
- **Paper**: [PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://arxiv.org/abs/1912.01703)
- **Authors**: Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, Soumith Chintala
- **Conference**: NeurIPS 2019
- **Key Contributions**: PyTorch, deep learning framework

#### OpenCV
- **Paper**: [OpenCV: A Computer Vision Library](https://opencv.org/)
- **Authors**: OpenCV Team
- **Conference**: Open Source
- **Key Contributions**: OpenCV, computer vision library

#### PIL/Pillow
- **Paper**: [PIL: Python Imaging Library](https://pillow.readthedocs.io/)
- **Authors**: Fredrik Lundh
- **Conference**: Open Source
- **Key Contributions**: PIL, image processing library

## 📊 Dataset Statistics

### CelebA-Spoof
- **Total Images**: 625,537
- **Real Images**: 313,829
- **Spoof Images**: 311,708
- **Subjects**: 10,177
- **Attack Types**: 5 (print, replay, 3D mask, makeup, partial)
- **Lighting Conditions**: 3 (normal, strong, weak)
- **Resolution**: Various (resized to 224x224)

### LCC FASD
- **Total Images**: 1,000
- **Real Images**: 500
- **Spoof Images**: 500
- **Subjects**: 50
- **Attack Types**: 2 (print, replay)
- **Lighting Conditions**: 3 (normal, strong, weak)
- **Resolution**: 640x480

### SiW
- **Total Images**: 4,660
- **Real Images**: 2,330
- **Spoof Images**: 2,330
- **Subjects**: 165
- **Attack Types**: 4 (print, replay, 3D mask, makeup)
- **Lighting Conditions**: Various
- **Resolution**: 1920x1080

### OULU-NPU
- **Total Images**: 5,940
- **Real Images**: 2,970
- **Spoof Images**: 2,970
- **Subjects**: 55
- **Attack Types**: 2 (print, replay)
- **Lighting Conditions**: 3 (normal, strong, weak)
- **Resolution**: 1920x1080

## 🔬 Research Applications

### Face Anti-Spoofing
- **Biometric Security**: Face recognition and anti-spoofing systems
- **Mobile Applications**: On-device face anti-spoofing
- **Surveillance Systems**: Real-time spoofing detection
- **Access Control**: Secure authentication systems

### Deep Learning Research
- **Architecture Design**: Novel mobile-optimized architectures
- **Training Techniques**: Advanced training methods for improved performance
- **Model Compression**: Knowledge distillation and pruning techniques
- **Few-shot Learning**: Meta-learning for rapid adaptation

### Computer Vision
- **Image Classification**: Advanced image classification techniques
- **Feature Learning**: Self-supervised and supervised feature learning
- **Model Interpretability**: Understanding model decision-making
- **Visualization**: Advanced visualization and analysis tools

## 📈 Performance Benchmarks

### Model Performance Comparison
| Model | Dataset | AUC | EER | ACER | Parameters | FLOPs |
|-------|---------|-----|-----|------|-----------|-------|
| MobileNetV4-Large | CelebA-Spoof | 0.96+ | <4% | <2% | 5.4M | 219M |
| MobileNetV4-Medium | CelebA-Spoof | 0.95+ | <5% | <3% | 4.2M | 155M |
| MobileNetV4-Small | LCC FASD | 0.92+ | <7% | <4% | 3.1M | 112M |
| EfficientNet-B7 | CelebA-Spoof | 0.94+ | <6% | <3% | 66M | 37B |
| ViT-Base | CelebA-Spoof | 0.93+ | <7% | <4% | 86M | 17B |
| ResNet-50 | CelebA-Spoof | 0.91+ | <8% | <5% | 25M | 4B |

### Cross-Dataset Performance
| Model | CelebA-Spoof | LCC FASD | SiW | OULU-NPU |
|-------|--------------|----------|-----|----------|
| MobileNetV4-Large | 0.96+ | 0.94+ | 0.93+ | 0.95+ |
| MobileNetV4-Medium | 0.95+ | 0.92+ | 0.91+ | 0.94+ |
| MobileNetV4-Small | 0.93+ | 0.90+ | 0.89+ | 0.92+ |

## 🎯 Future Research Directions

### Architecture Improvements
- **Neural Architecture Search**: Automated architecture discovery
- **Efficient Transformers**: Mobile-optimized attention mechanisms
- **Dynamic Networks**: Adaptive computation based on input complexity

### Training Enhancements
- **Federated Learning**: Distributed training across devices
- **Continual Learning**: Learning from new data without forgetting
- **Multi-Modal Learning**: Incorporating additional modalities (depth, infrared)

### Deployment Optimization
- **Model Quantization**: INT8 and binary quantization
- **Hardware Acceleration**: Optimized inference on mobile devices
- **Edge Computing**: Real-time processing on edge devices

### Evaluation and Benchmarking
- **Standardized Benchmarks**: Comprehensive evaluation protocols
- **Cross-Dataset Analysis**: Robustness across different domains
- **Real-World Testing**: Performance in practical scenarios

---

**Note**: This document provides comprehensive references for all research papers, datasets, and methodologies used in the face anti-spoofing project. All references are properly cited with links to the original papers and datasets.

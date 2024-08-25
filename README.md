# CL_Fair
This repository conatins the implementation of the paper "Mitigating Label Bias in Machine Learning: Fairness through Confident Learning," accepted at AAAI 2024.

## Abstract
Discrimination can occur when the underlying unbiased la- bels are overwritten by an agent with potential bias, resulting in biased datasets that unfairly harm specific groups and cause classifiers to inherit these biases. In this paper, we demon- strate that despite only having access to the biased labels, it is possible to eliminate bias by filtering the fairest instances within the framework of confident learning. In the context of confident learning, low self-confidence usually indicates potential label errors; however, this is not always the case. Instances, particularly those from underrepresented groups, might exhibit low confidence scores for reasons other than labeling errors. To address this limitation, our approach em- ploys truncation of the confidence score and extends the con- fidence interval of the probabilistic threshold. Additionally, we incorporate with co-teaching paradigm for providing a more robust and reliable selection of fair instances and effec- tively mitigating the adverse effects of biased labels. Through extensive experimentation and evaluation of various datasets, we demonstrate the efficacy of our approach in promoting fairness and reducing the impact of label bias in machine learning models.

## Reference
```
@article{Zhang_Li_Ling_Zhou_2024, title={Mitigating Label Bias in Machine Learning: Fairness through Confident Learning}, 
        volume={38}, 
        url={https://ojs.aaai.org/index.php/AAAI/article/view/29634}, DOI={10.1609/aaai.v38i15.29634}, 
        number={15}, 
        journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
        author={Zhang, Yixuan and Li, Boyu and Ling, Zenan and Zhou, Feng}, 
        year={2024}, 
        month={Mar.}, 
        pages={16917-16925} }
```

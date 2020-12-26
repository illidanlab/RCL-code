# Robust Collaborative Learning for Noisy Labels.
## Abstract

Learning with curriculum has shown great effectiveness in tasks where the data contains noisy (corrupted) labels, since the curriculum can be used to re-weight or filter out noisy samples via proper design. However, obtaining curriculum from a learner itself without additional supervision or feedback deteriorates the effectiveness due to sample selection bias. Therefore, methods that involve two or more networks have been recently proposed to mitigate such bias. Nevertheless, these studies utilize the collaboration between networks in a way that either emphasizes the disagreement or focuses on the agreement while ignores the other. In this paper, we study the underlying mechanism of how disagreement and agreement between networks can help reduce the noise in gradients and develop a novel framework called Robust Collaborative Learning (RCL) that leverages both disagreement and agreement among networks. We demonstrate the effectiveness of RCL on both synthetic benchmark image data and real-world large-scale bioinformatics data.

The current version of the paper can be found here.

## Compatibility

python 2.7

pythorch 1.4.0






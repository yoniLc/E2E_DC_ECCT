# Learning Linear Block Error Correction Codes

Implementation of the End-to-end deep codes Error Correction Code Transformer described in ["Learning Linear Block Error Correction Codes (ICML 2024)"](https://arxiv.org/abs/2405.04050).

The decoder implementation is related to the Foundation Error Correction Codes model published in ["A Foundation Model for Error Correction Codes (ICLR 2024)"](https://openreview.net/forum?id=7KDuQPrAF3) 


## Abstract
Error correction codes are a crucial part of the physical communication layer, ensuring the reliable transfer of data over noisy channels. The design of optimal linear block codes capable of being efficiently decoded is of major concern, especially for short block lengths. While neural decoders have recently demonstrated their advantage over classical decoding techniques, the neural design of the codes remains a challenge. In this work, we propose for the first time a unified encoder-decoder training of binary linear block codes. To this end, we adapt the coding setting to support efficient and differentiable training of the code for end-to-end optimization over the order two Galois field. We also propose a novel Transformer model in which the self-attention masking is performed in a differentiable fashion for the efficient backpropagation of the code gradient. Our results show that (i) the proposed decoder outperforms existing neural decoding on conventional codes, (ii) the suggested framework generates codes that outperform the {analogous} conventional codes, and (iii) the codes we developed not only excel with our decoder but also show enhanced performance with traditional decoding techniques.

## Install
- Pytorch

## Script
Use the following command to train a toy example. Every modification can be performed via the main function.

`python E2E_DC_ECCT.py`


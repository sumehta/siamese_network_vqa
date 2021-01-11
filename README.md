**Siamese Network for Binary Visual Question Answering**

*Introduction
Binary Visual Question Answering is the task of visual verification of concepts in the image. Concepts are represented by <P, R, S> tuples, if the concept is present then the answer is yes if the concept is absent the answer is no. Here P is the primary object, R is a relation and S is a secondary concept.
Example:  <cat, in, room>

*Model
Base Model - One branch extracts image features corresponding to the objects in the tuple and another branch encodes the question using LSTM. Features are fused using an MLP followed by cross entropy.

Proposed Model - A siamese network based architecture for binary(Yes/No) visual question answering (VQA) with a max-margin loss. The basic idea is to have two VQA networks that share parameters. The loss function is designed in such a way that the ‘yes’ probability output by the ‘yes’ network is atleast a margin away from the ‘yes’ probability output by the ‘no’ network.
For background and details on the binary VQA task please refer to this [paper](https://arxiv.org/abs/1511.05099).

*Project Structure
- VQA model used: https://github.com/VT-vision-lab/VQA_LSTM_CNN
- train.lua:  training script for the siamese VQA network
- eval.lua: evaluation script for the trained model
- Youtube video link: https://www.youtube.com/watch?v=nR2s0T426PA



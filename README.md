**Siamese Network for Binary Visual Question Answering**

A siamese network based architecture for binary(Yes/No) visual question answering (VQA). The basic idea is to have two networks that share parameters. One network learns to answer 'yes'-type questions (by minimizng the corresponding loss) while the other network learns to answer the 'no'-type questions by minimizing its corresponding loss. We use a contrasitive loss to backpropogate errors through the whole network.
For background and details on the binary VQA task please refer to this [paper](https://arxiv.org/abs/1511.05099).

- VQA model used: https://github.com/VT-vision-lab/VQA_LSTM_CNN
- train.lua:  training script for the siamese VQA network
- eval.lua: evaluation script for the trained model
- Youtube video link: https://www.youtube.com/watch?v=nR2s0T426PA



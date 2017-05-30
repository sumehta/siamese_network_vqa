**Siamese network for binary VQA**

A siamese network based architecture for binary(Yes/No) question answering. The basic idea behind the network is to have two networks sharing parameters. One network learns to answer 'yes'-type questions (by minimizng the corresponding loss) while the other network learns to answer the 'no'-type questions (again by minimizing the corresponding loss). Both losses are linearly combined and the error is back-propogated through the entire network. For more background and details on the binary VQA task please refer this [paper](https://arxiv.org/abs/1511.05099). I would suggest come back to this code after going through the above paper to get a better understanding of the task/code.

- VQA model used: https://github.com/VT-vision-lab/VQA_LSTM_CNN
- train.lua:  training script for the siamese VQA network
- eval.lua: evaluation script for the trained model
- Youtube video link: https://www.youtube.com/watch?v=nR2s0T426PA



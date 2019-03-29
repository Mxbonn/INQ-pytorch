# Incremental Network Quantization
A PyTorch implementation of "Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights"

	@inproceedings{zhou2017,
	title={Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights},
	author={Aojun Zhou, Anbang Yao, Yiwen Guo, Lin Xu, Yurong Chen},
	booktitle={International Conference on Learning Representations,ICLR2017},
	year={2017},
	}
[[Paper]](https://arxiv.org/abs/1702.03044)

Official Caffe implementation is available [[here]](https://github.com/AojunZhou/Incremental-Network-Quantization)
(Code in this repo is based on the paper and not on the official Caffe implementation)

----
### Installation
The code is implemented in Python 3.7 and PyTorch 1.0
```
pip install -e .
```

### Usage
`inq.SGD(...)` implements SGD that only updates weights according to the weight partitioning scheme of INQ.

`inq.INQScheduler(...)` handles the the weight partitioning and group-wise quantization stages of the incremental network quantization procedure.


`reset_lr_scheduler(...)` resets the learning rate scheduler.

### Examples

The INQ training procedure looks like the following:
```python
optimizer = inq.SGD(...)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, ...)
inq_scheduler = inq.INQScheduler(optimizer, [0.5, 0.75, 0.82, 1.0], strategy="pruning")
for inq_step in range(3): # Iteration for accumulated quantized weights of 50% 75% and 82% 
    inq.reset_lr_scheduler(scheduler)
    inq_scheduler.step()
    for epoch in range(5):
        scheduler.step()
        train(...)
inq_scheduler.step() # quantize all weights, further training is useless
validate(...)
```

`examples/imagenet_quantized.py` is a modified version of the official PyTorch imagenet example.
Using this example you can quantize a pretrained model on imagenet.
Compare the file to the [original example](https://github.com/pytorch/examples/tree/master/imagenet) to see the differences.

#### Results
Results using this code differ slightly from the results reported in the paper. 

|Network|Strategy|Bit-width|Top-1 Accuracy|Top-5 Accuracy|
|-------|--------|---------|--------------|--------------|
|resnet18|ref    |32       |69.758%       |89.078%       |
|resnet18|random |5        |69.064%       |88.766%       |
|resnet18|pruning|5        |              |              |
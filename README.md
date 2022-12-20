Fast Generation-Based Gradient Leakage Attacks against Highly Compressed Gradients



## Abstract

In federated learning, clients' private training data can be stolen from publicly shared gradients. The existing attacks either require modification of the FL model (analytics-based) or take a long time to converge (optimization-based) and fail against highly compressed gradients in practical FL systems. We contrive a new generation-based attack algorithm capable of reconstructing the original mini-batch of data from the compressed gradient in just a few milliseconds. 



## Overview

The whole attack process can be briefly described as:

- train a generator from the output feature of the convolutional layer to the raw images in advance using the auxiliary dataset.
- Obtain the output feature of the convolutional layer from the gradient of the fully connected layer.
- Sending the feature to the generator to get user images.

<img src="https://github.com/xuedongyun/FGLA/blob/master/images/generator.png?raw=true" alt="generator.png" style="zoom: 25%;" />

The core code of the algorithm is **extremely simple**:

```python
# training (key step)
criterion = nn.MSELoss()
origin_model.fc = nn.Sequential() 

dummy_img = generator(origin_model(img))
loss = criterion(dummy_img, img)
```

```python
# attack
g_w = grad[-2]
g_b = grad[-1]

offset_w = torch.stack([g for idx, g in enumerate(g_w) if idx not in y], dim=0).mean(dim=0) * (bz - 1) / bz
offset_b = torch.stack([g for idx, g in enumerate(g_b) if idx not in y], dim=0).mean() * (bz - 1) / bz
conv_out = (g_w[y] - offset_w) / (g_b[y] - offset_b).unsqueeze(1)

conv_out[torch.isnan(conv_out)] = 0.
conv_out[torch.isinf(conv_out)] = 0.

img = generator(conv_out)
```

![effect](https://github.com/xuedongyun/FGLA/blob/master/images/effect.png?raw=true)



## Sample Usage

Download generator weights file [gen_weights.pth](https://drive.google.com/file/d/1x6KIpGXJARc9F0SMZUyTDy3h5-CQRIaR/view?usp=sharing) and place it in folder ```./data/```



Reconstruct 10 batches of images using FGLA algorithm:

```shell
python reconstruct_exp.py --exp_name="my_exp" --dataset="imagenet" --reconstruct_num=10
```

The results all you need will be placed in ```data/reconstruct/{exp_name}```/

| argument        | help                                                       | optional value                 |
| --------------- | ---------------------------------------------------------- | ------------------------------ |
| algorithm       | gradient leakage attack algorithm                          | fgla, dlg, stg, ig             |
| model_weights   | weights path for generator                                 |                                |
| reconstruct_num | number of reconstructed batches                            |                                |
| dataset         | dataset to use                                             | imagenet, cifar100, caltech256 |
| max_iteration   | number of iterations when use optimization-based algorithm |                                |
| exp_name        | the name of the experiment, used to create a folder        |                                |
| batch_size      | batch size                                                 |                                |
| device          | which device to use                                        |                                |
| seed            | random seed                                                |                                |



Also, you can train your own generator:

```shell
python train_generator.py --exp_name="my_train_exp"
```

The results all you need will be placed in ```data/train_generator/{exp_name}/```

| argument   | help                                                | optional value |
| ---------- | --------------------------------------------------- | -------------- |
| exp_name   | the name of the experiment, used to create a folder |                |
| batch_size | batch size                                          |                |
| epochs     | epochs for training                                 |                |
| device     | which device to use                                 |                |
| seed       | random seed                                         |                |



## Colab

<p align="left">We provide a simplified version
    <a href="https://colab.research.google.com/drive/1c7On5cO0tlZGgLafgAqqxE1NV0vAZDcV?usp=sharing" target="_parent">
        <img src="https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a>
     for experience.
</p>

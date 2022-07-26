# FGLA: Fast Generation-Based Gradient Leakage Attacks against Highly Compressed Gradients

## Abstract
In federated learning, clients' private training data can be stolen from publicly shared gradients. 
The existing attack algorithms are either based on analytics or optimization to reconstruct the private data. 
Nevertheless, the optimization-based algorithms are time-consuming and the analytics-based algorithms can only succeed in contrived settings (active attack). 
In addition, the gradients shared in real-world federated learning scenarios tend to be highly compressed, resulting in the most advanced attack algorithms being ineffective. 

We contrive a new generation-based attack algorithm capable of reconstructing the original mini-batch of data from the compressed gradient in just a few milliseconds. 

## Overview

# Search-For-Exotic-Particles-Using-Deep-Learning

# Introduction

The problem at hand is a classification problem, where we need to distinguish between two processes,
<br>
1. Signal process
<br>
2. Background process
<br>
<br>
The distinction of these two processes is of interest to us as the signal processes produce the Higgs bosons, also know as God Particles, which are of great interest to scientists, whereas the background processes do not produce any such exotic particles and are just considered noise. It’s interesting to know that detection of the signal processes is vital for progress in particle physics and hence classification of the same is crucial.
<br>
<br>

# Objective

The objective of this project is to build a deep neural network to accurately classify the processes. We will evaluate the whole model based on test accuracy, prediction time for a sample and training time.
<br>
<br>

# Data

The data has been produced using Monte Carlo simulations. The first 21 features (low-level features) are kinematic properties measured by the particle detectors in the accelerator. The last 7 features are functions of the first 21 features; these are the high level features derived by physicists to help discriminate between the two classes.
<br>
<br>
The first column is the class label (1 for signal, 0 for background), followed by 21 low-level features then 7 high-level features.  All the attribute values are real values.
<br>
<br>
The data can be obtained <a href="">here</a>. For more information on the data and the attributes, read the <a href="">original paper</a>.

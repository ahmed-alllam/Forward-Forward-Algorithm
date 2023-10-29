<a name="readme-top"></a>


<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">

<h3 align="center">Forward-Forward Algorithm: An Alternative Learning Algorithm to Backpropagation</h3>

  <p align="center">
<!--     <a href="https://github.com/ahmed-alllam/Forward-Forward-Algorithm">View Demo</a> -->
<!--     · -->
    <a href="https://github.com/ahmed-alllam/Forward-Forward-Algorithm/issues">Report Bug</a>
    ·
    <a href="https://github.com/ahmed-alllam/Forward-Forward-Algorithm/issues">Request Feature</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->


## About The Project

This project is an implementation of Geoffrey Hinton's Forward-Forward Algorithm, an alternative learning algorithm to Backpropagation. The algorithm is described in the research paper [The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/abs/2212.13345). This new algorithm replaces the traditional forward and backward passes of backpropagation with two forward passes: one using positive (i.e. real) data and another using negative data. Each layer in the neural network is associated with its own objective function to maximize the "goodness" for positive data and minimize it for negative data.

In traditional neural network training using backpropagation, the gradients are computed in the backward pass. However, the Forward-Forward Algorithm provides an alternative approach where the precise details of the forward computation are not necessary, enabling training even in the presence of unknown non-linearities.

The main advantages of the Forward-Forward Algorithm include:

- Not requiring precise knowledge of the forward pass.
- Allowing neural networks to pipeline sequential data without needing to store neural activities or propagate error derivatives.
- A potential model for learning in the cortex and optimal utilization of low-power analog hardware without the need for reinforcement learning.

However, it's worth noting that in preliminary tests, the Forward-Forward Algorithm has been found to be somewhat slower than backpropagation and may not generalize as effectively on some problems. But as the understanding and refinement of the algorithm progress, its potential applications in areas like analog hardware and neuromorphic computing could make it a game-changer in the future of machine learning and neural network design.


<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/ahmed-alllam/Forward-Forward-Algorithm.svg?style=for-the-badge
[contributors-url]: https://github.com/ahmed-alllam/Forward-Forward-Algorithm/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ahmed-alllam/Forward-Forward-Algorithm.svg?style=for-the-badge
[forks-url]: https://github.com/ahmed-alllam/Forward-Forward-Algorithm/network/members
[stars-shield]: https://img.shields.io/github/stars/ahmed-alllam/Forward-Forward-Algorithm.svg?style=for-the-badge
[stars-url]: https://github.com/ahmed-alllam/Forward-Forward-Algorithm/stargazers
[issues-shield]: https://img.shields.io/github/issues/ahmed-alllam/Forward-Forward-Algorithm.svg?style=for-the-badge
[issues-url]: https://github.com/ahmed-alllam/Forward-Forward-Algorithm/issues
[license-shield]: https://img.shields.io/github/license/ahmed-alllam/Forward-Forward-Algorithm.svg?style=for-the-badge
[license-url]: https://github.com/ahmed-alllam/Forward-Forward-Algorithm/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/ahmed-e-allam
[product-screenshot]: images/screenshot.png

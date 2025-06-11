## In this CUDA VERSION

The cuda version of the MLP doesn't function with SFML. Therefore, there is no interface to play with. Nevertheless, it was quite interesting to notice the gain in speed (x2 to x4) using the GPU compared to CPU.

Basically, in the Cuda Version, only some function were changed: the computation of Cross Entropy Loss, the Adam optimizer, and every matrix transformation (for the forward pass).
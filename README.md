# snake_DQN
Implementation of DQN algorithm to simple snake game. 
Neural networks implementation is provided by PyTorch. Stats are visualised in TensorBoard.

There are 2 regimes of learning:

1) Without any screen init. 
It's faster and can be transfered to Kaggle to train on GPU
which is much more faster.

2) Screen init (pygame) with game process window.
It's visual rendering of snake playing. 
Also in that regime you can take over control of a snake and just play by yourself,
use it for generating new experience that snake_ai will learn. 
(keep pressed 'ctrl' for taking control and use arrow buttons for playing).

There are 2 branches of developing code: 
One is for wide options of IDE, and second is for KAggle GPU version (only without screen regime). 

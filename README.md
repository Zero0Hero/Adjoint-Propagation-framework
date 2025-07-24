# Adjoint Propagation (AP) Framework
Paper at https://www.researchsquare.com/article/rs-6759684/v1
## Environments
```
python = 3.9.20
pytorch = 2.4.1
numpy = 1.26.4
torchvision = 0.19.1
matplotlib = 3.9.2
tqdm = 4.66.5
```
## Example1

You can directly run the demo. Then you will get the trend of accuracy on FMNIST (two layer, 1024 neurons per layer, 256 per block).


## Example2

'M_range' and 'ParaRange' is a set of two tunable parameters. You can draw different plots by adjusting them. 

Take Fig. 3c STE as an example: 
'M_range' for number of iterations, 'ParaRange' for SR. Repeat the test 5 times.
```
config.flag_ycz = False # STE
M_range = [2,4,8,16]
ParaRange = [0.01,0.1,0.25,0.5,0.75,1,1.25,1.5,2] 
multitest = 5
```
And assign values to specific parameters in the loop.
```
config.RNN_t2sta = M_range[iM]
config.RNN_t2sta2 = M_range[iM]
config.RNN_SR = ParaRange[iPara]
```
At the final figure, the legends represent the 'M_range' and the axis x represent the 'ParaRange'. 

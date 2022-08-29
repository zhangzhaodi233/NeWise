# Installation
All the scripts and code were tested on a workstation running Ubuntu 18.04.

1. Download the code:
   ```
   git clone https://github.com/zhangzhaodi233/NeWise.git
   cd NeWise
   ```
2. Install all the necessary dependencies:
    ```
    . install.sh
    ```

When all the necessary dependencies are installed, the message "The enviroment has been deployed!" pops up.

In addition, we also provide the docker image for NeWise:

1. Download the docker image from xxxx. (The image contains the NeWise's code).
2. Load the docker image:
   ```
   docker load -i newise.tar
   ```
3. Start a container with the image:
   ```
   docker run -it newise:v2 /bin/bash
   ```

## Run NeWise and reproduce the results

1. Activate the virtual environment:
   ```
   source venv/bin/activate
   ```

2. Reproduce the results:
   ```. run.sh``` or ```nohup sh run.sh > nohup.out 2>&1 &``` (if you prefer to run it in the background).

3. The tables will be saved in **results/table_results.txt** while the figures in **figs/**.

Example format of Table 6 (partial):
```

==========================================================================================================================================================================================================================================================
Table 6: 

Model                                                             	 NW_aver 	 DC_aver 	 Impr.(%) 	 VN_aver 	 Impr.(%) 	 RV_aver 	 Impr.(%) 	 NW_std 	 DC_std 	 Impr.(%) 	 VN_std 	 Impr.(%) 	 RV_std 	 Impr.(%) 	    Time(s)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
mnist_ffnn_5x100_with_positive_weights_8858.h5                    	 0.0091 	 0.0071 	  28.15 	 0.0071 	  27.25 	 0.0064 	  40.68 	 0.0057 	 0.0042 	  37.11 	 0.0042 	  35.48 	 0.0034 	  68.34 	   4.47 0.01
mnist_ffnn_3x700_with_positive_weights_9537.h5                    	 0.0037 	 0.0030 	  24.92 	 0.0030 	  22.85 	 0.0029 	  26.62 	 0.0018 	 0.0013 	  41.86 	 0.0014 	  34.56 	 0.0013 	  40.77 	 121.40 0.13
mnist_cnn_6layer_5_3_with_positive_weights.h5                     	 0.0968 	 0.0788 	  22.82 	 0.0778 	  24.37 	 0.0699 	  38.48 	 0.0372 	 0.0280 	  32.92 	 0.0276 	  35.09 	 0.0212 	  75.70 	   5.69 0.41
mnist_ffnn_3x50_with_positive_weights_9118.h5                     	 0.0105 	 0.0088 	  19.23 	 0.0088 	  19.50 	 0.0080 	  31.42 	 0.0051 	 0.0038 	  32.72 	 0.0038 	  32.72 	 0.0029 	  71.86 	   0.14 0.00
mnist_ffnn_3x100_with_positive_weights_9162.h5                    	 0.0139 	 0.0120 	  15.46 	 0.0120 	  15.56 	 0.0111 	  25.47 	 0.0071 	 0.0057 	  24.82 	 0.0057 	  23.30 	 0.0046 	  53.13 	   2.22 0.01
mnist_cnn_5layer_5_3_with_positive_weights.h5                     	 0.0801 	 0.0708 	  13.14 	 0.0704 	  13.75 	 0.0683 	  17.32 	 0.0238 	 0.0200 	  18.87 	 0.0198 	  20.50 	 0.0180 	  32.35 	   2.88 0.30
mnist_ffnn_3x200_with_positive_weights_9420.h5                    	 0.0080 	 0.0071 	  12.54 	 0.0071 	  12.85 	 0.0068 	  17.16 	 0.0046 	 0.0037 	  26.43 	 0.0037 	  25.41 	 0.0034 	  37.28 	   8.38 2.87
mnist_ffnn_3x400_with_positive_weights_9630.h5                    	 0.0061 	 0.0056 	   9.66 	 0.0056 	   9.86 	 0.0054 	  12.89 	 0.0035 	 0.0030 	  16.78 	 0.0030 	  16.39 	 0.0027 	  26.55 	  40.80 0.12
mnist_cnn_3layer_2_3_with_positive_weights_9120.h5                	 0.0521 	 0.0483 	   7.82 	 0.0483 	   7.94 	 0.0478 	   8.88 	 0.0180 	 0.0161 	  12.13 	 0.0160 	  12.41 	 0.0156 	  15.44 	   0.17 0.04
mnist_cnn_4layer_5_3_with_positive_weights_8563.h5                	 0.0505 	 0.0473 	   6.68 	 0.0471 	   7.26 	 0.0464 	   8.81 	 0.0207 	 0.0186 	  11.26 	 0.0183 	  12.84 	 0.0175 	  17.80 	   1.17 0.19
mnist_cnn_3layer_4_3_with_positive_weights_9151.h5                	 0.0448 	 0.0422 	   6.09 	 0.0421 	   6.24 	 0.0418 	   6.98 	 0.0156 	 0.0142 	   9.71 	 0.0141 	  10.18 	 0.0138 	  12.56 	   0.30 0.08
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
fashion_mnist_ffnn_4x100_with_positive_weights_sigmoid.h5         	 0.0312 	 0.0188 	  65.48 	 0.0194 	  60.62 	 0.0159 	  96.22 	 0.0403 	 0.0210 	  92.28 	 0.0220 	  83.20 	 0.0176 	 129.33 	   3.31 0.02
fashion_mnist_ffnn_3x100_with_positive_weights_sigmoid.h5         	 0.0326 	 0.0263 	  24.02 	 0.0270 	  21.03 	 0.0238 	  36.87 	 0.0335 	 0.0262 	  27.67 	 0.0282 	  18.92 	 0.0234 	  43.22 	   2.21 0.01
fashion_mnist_ffnn_2x100_with_positive_weights_sigmoid.h5         	 0.0306 	 0.0250 	  22.49 	 0.0254 	  20.51 	 0.0230 	  33.03 	 0.0286 	 0.0211 	  36.04 	 0.0228 	  25.88 	 0.0194 	  47.76 	   1.13 0.01
fashion_mnist_ffnn_3x200_with_positive_weights_sigmoid.h5         	 0.0223 	 0.0184 	  21.80 	 0.0187 	  19.45 	 0.0170 	  31.63 	 0.0220 	 0.0159 	  38.66 	 0.0170 	  29.38 	 0.0143 	  54.42 	  11.21 0.05
fashion_mnist_ffnn_2x200_with_positive_weights_sigmoid.h5         	 0.0263 	 0.0220 	  19.52 	 0.0223 	  17.86 	 0.0204 	  28.77 	 0.0279 	 0.0200 	  39.55 	 0.0211 	  31.96 	 0.0176 	  58.76 	   5.63 0.04
```



**Note:** the results of Table 4 and Table 9 would be slightly different for each run as the images were taken randomly. However, the conclusions keep  consistent as made in the paper: the approximation computed by Algorithm 1 is the optimal approximation for a neural network containing only one hidden layer.
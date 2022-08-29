# NeWise Framework

NeWise is a general and efficient framework for certifying the robustness of neural networks. Given a neural network and an input image, NeWise can calculate more precise certified lower robustness bound. Technical details can be found in the accepted ASE'22 [paper](https://github.com/zhangzhaodi233/NeWise/blob/main/ASE22_submission_159_technical_report.pdf).

## Project Structure
> tool's code:
> + alg1
>     + utils_pack
>     + certify_mlp.py
>     + frown_general_activation_neuronwise.py
>     + get_bound_for_general_activation_function.py
>     + main.py
>     + mlp.py
> + activations.py 
> + cnn_bounds_full_core_with_LP.py 
> + cnn_bounds_full_with_LP.py 
> + mlp_keras_2_pytorch.py 
> + train_myself_model.py 
> + utils.py 

> experimental data:
> + alg1
>     + datasets
> + data
> + models

> reproduce figures and tables in the paper:
> + figure_xxx.py 
> + output_middel_layer_data.py 
> + table_xxx.py 

> test results produced when the code was run on our local machine:
> + xxx_local

> modify one file in virtual environment:
> + modify_file.py

> scripts to install and run.
> + install.sh
> + requirements.txt
> + run.sh

## Install NeWise

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

1. Download the docker image from https://figshare.com/articles/software/NeWise/20709868. (The image contains the NeWise's code).
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

Example output of Table 6 (partial):
``` 

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

## Example
To use NeWise to compute the certified lower bound for models with non-negative weights, run the following command:
```
python example.py --model models/models_with_positive_weights/sigmoid/mnist_cnn_3layer_2_3_with_positive_weights_9120.h5 --images 10 --data_from_local 1 --method NeWise --activation sigmoid --dataset mnist --logname example --purecnn 1 
```
- ```--model```: path to the network file.
- ```--images```: the number of images.
- ```--data_from_local```: whether use local images (local: 1; download: 0).
- ```--method```: the method of approximation (can be either NeWise, DeepCert, VeriNet, or RobustVerifier).
- ```--activation```: the type of activation (can be either sigmoid, tanh, atan).
- ```--dataset```: the type of dataset (can be either mnist, cifar, or fashion_mnist).
- ```--logname```: the name of log.
- ```--purecnn```: whether the model is pure cnn.

After the command runs, the terminal will print out a series of related information. The result is the last row：
```
[L0] method = NeWise-sigmoid, total_images=9, avg=0.05415, std=0.02051, avg_runtime=0.16
```

To use NeWise to compute the certified lower bound for models containing only one hidden layer, run the following command:
```
python alg1/main.py --log_name alg1_example --batch_size 5 --model_dir models/one_layer_models/ --model_name mnist_fnn_1x50_sigmoid_local.pth --num_neurons 50 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize
```
- ```--log_name```: the name of log.
- ```--batch_size```: the number of images.
- ```--model_dir```: the directory where the pretrained model is stored.
- ```--model_name```: the name of the pretrained model.
- ```--num_neurons```: the number of neurons in the hidden layer of the pretrained model.
- ```--num_layers```: the number of layers of the pretrained model.
- ```--activation```: the type of activation.
- ```--dataset```: the type of dataset.
- ```--neuronwise_optimize```: whether optimize every neuron.

After the command runs, the terminal will print out a series of related information. The result is the last four row：
```
model models/one_layer_models/mnist_fnn_1x50_sigmoid_local.pth in 3.00 seconds
average 0.60 seconds
statistics of l_eps
mean=0.03527344 std=0.01387654
```

## Contributors
- Zhaodi Zhang (contact) - zdzhang@stu.ecnu.edu.cn
- Yiting Wu - 51205902026@stu.ecnu.edu.cn
- Si Liu - si.liu@inf.ethz.ch
- Jing Liu - jliu@sei.ecnu.edu.cn
- Min Zhang (contact) - zhangmin@sei.ecnu.edu.cn

## Cite
```
@article{2208.09872,
	Author = {Zhaodi Zhang and Yiting Wu and Si Liu and Jing Liu and Min Zhang},
   Title = {Provably Tightest Linear Approximation for Robustness Verification of Sigmoid-like Neural Networks},
   Year = {2022},
	journal   = {CoRR},
	volume    = {arXiv:2208.09872},
}
```


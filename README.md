# Deep-Learning  Assignment 1
This project is an implementation of a neural network from scratch using python also using wandb for logging the data like accuracy and losses. You can create customized neural network and train with either mnist or fashion_mnist dataset with different configurations of parameters, then visualize the losses and accuracies via wandb.

### Dependencies
 - python
 - numpy library
 - wandb library
 - copy library
 - keras library (for downloading mnist and fashion_mnist dataset)

Either download the above dependencies or run :  `pip install -r requirements.txt`

### Usage
First make sure the above dependencies have been installed in your system. Then to use this project, simply clone the repository and run the train.py file. To download the repository, type the below command in the command line interface of git:

`git clone https://github.com/TharikH/deep-learning`

You can also download the entire repository as a zip file from the Download ZIP option on the above button.



We can give different values for the parameters and this can be done by specifying it in the command line arguments. Different possible arguments and its corresponding values are shown in the table below:

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | cs22m058  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 10 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 16 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | momentum | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.001 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.9 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.9 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.9 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.99 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.1 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | 0.0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | xavier | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 5 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 128 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | tanh | choices:  ["identity", "sigmoid", "tanh", "relu"] |

</br>

Here it is an example of how to do run with epoch = 5 and activation as relu : `python train.py -e 5 -a relu`

It also contains dl_assignment.ipynb which can be open via colab or jupyter notebook. It contains all the codes for neural network, running the sweeps, logging the confusion matrices, logging the images for the dataset etc. But it doesn't have every configurations, need to change the second last and last cell to run for different configurations separately, for example to run and log confusion matrix, losses and accuracies, just run before second last cell and run last cell as second last cell is for sweep. To run sweep run till second last cell (can change config accordingly). So .pynb file can be used as reference for the flow.



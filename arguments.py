import argparse

def parsArg():
    parser = argparse.ArgumentParser()

    parser.add_argument('-wp','--wandb_project',default='myprojectname',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we','--wandb_entity',default='cs22m058',
                        help='wandb entity name')
    parser.add_argument('-d','--dataset',default='fashion_mnist',choices=['fashion_mnist','mnist'],
                        help='Dataset to be used - choices: ["mnist", "fashion_mnist"]')
    parser.add_argument('-e','--epochs',default=10,type=int,
                        help='Number of epochs to train neural network')
    parser.add_argument('-b','--batch_size',default=16, type=int,
                        help='Batch size used to train neural network')
    parser.add_argument('-l','--loss',default='cross_entropy',choices = ["mean_squared_error", "cross_entropy"],
                        help='choices: ["mean_squared_error", "cross_entropy"]')
    parser.add_argument('-o','--optimizer',default='momentum', choices = ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                        help='choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]')
    parser.add_argument('-lr','--learning_rate',default=0.001, type=float,
                        help='Learning rate used to optimize model parameters')
    parser.add_argument('-m','--momentum',default=0.9, type=float,
                        help='Momentum used by momentum and nag optimizers')
    parser.add_argument('-beta','--beta',default=0.9, type=float,
                        help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1','--beta1',default=0.9, type=float,
                        help='Beta1 used by adam and nadam optimizers.')
    parser.add_argument('-beta2','--beta2',default=0.99, type=float,
                        help='Beta2 used by adam and nadam optimizers')
    parser.add_argument('-eps','--epsilon',default=0.1, type=float,
                        help='Epsilon used by optimizers')
    parser.add_argument('-w_d','--weight_decay',default=0, type=float,
                        help='Weight decay used by optimizers')
    parser.add_argument('-w_i','--weight_init',default='xavier', choices = ["random", "Xavier"],
                        help='choices: ["random", "Xavier"]')
    parser.add_argument('-nhl','--num_layers',default=5, type=int,
                        help='Number of hidden layers used in feedforward neural network')
    parser.add_argument('-sz','--hidden_size',default=128, type=int,
                        help='Number of hidden neurons in a feedforward layer')
    parser.add_argument('-a','--activation',default='tanh', choices = ["identity", "sigmoid", "tanh", "ReLU"],
                        help='choices: ["identity", "sigmoid", "tanh", "ReLU"]')
    args = parser.parse_args()
    
    return args
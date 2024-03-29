import numpy as np
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import copy
import wandb
import arguments

from network import *
from loss_file import *

args = arguments.parsArg()
datasets = {'fashion_mnist':fashion_mnist,'mnist':mnist}

#constants
key = 'c425b887e2c725018a7f3a772582610fa54ef52c'
input_size = 784
output_size = 10
output_layer_activation="softmax"
data_name = args.dataset
num_hidden_layer = args.num_layers
hidden_size = args.hidden_size
hidden_layer_size = np.full(num_hidden_layer,hidden_size)
hidden_layer_activation = args.activation
weight_name = args.weight_init
epochs = args.epochs
weight_decay = args.weight_decay
optimizer_name = args.optimizer
lr = args.learning_rate
batch_size = args.batch_size
loss_name = args.loss
momentum_parameter = [args.momentum]
rmsprop_parameter = [args.beta,args.epsilon]
adam_parameter = [args.beta1,args.beta2,args.epsilon]

parameters = []
if(optimizer_name == 'rmsprop'):
  parameters = adam_parameter
if(optimizer_name == 'adam' or optimizer_name == 'nadam'):
  parameters == adam_parameter
if(optimizer_name == 'momentum' or 'nag'):
  parameters == momentum_parameter


#train and test split

(X_train, Y_train), (X_test, Y_test) = datasets[data_name].load_data()


# # Split train data to train and validation (10% of data)
num_train_samples = X_train.shape[0]
num_test_samples = X_test.shape[0]
num_validate_samples = num_train_samples//10
num_train_samples -= num_validate_samples

X_valid = X_train[:num_validate_samples,:].reshape(num_validate_samples,input_size).T /255.0
Y_valid = Y_train[:num_validate_samples]

X = X_train[num_validate_samples:,:].reshape(num_train_samples,input_size).T / 255.0
Y = Y_train[num_validate_samples:]

X_test = X_test.reshape(num_test_samples,input_size).T / 255.0
Y_test = Y_test

#wandb initialization
wandb.login(key = key)
wandb.init(project = args.wandb_project,entity=args.wandb_entity)

wandb.run.name = f'hln_{num_hidden_layer}_hls_{hidden_size}_hla_{hidden_layer_activation}_winit_{weight_name}_ep_{epochs}_op_{optimizer_name}_lr_{lr}_bs_{batch_size}_wd_{weight_decay}_ln_{loss_name}'

# Neural network initialization and training start
nn = NN(input_size = input_size,output_size = output_size,num_samples = num_train_samples, num_hidden_layer = num_hidden_layer, hidden_layer_size = hidden_layer_size,data_name= data_name, hidden_layer_activation = hidden_layer_activation, output_layer_activation=output_layer_activation,weight_name = weight_name,loss_name=loss_name)
val_loss_list,val_accuracy_list,train_loss_list,train_accuracy_list = nn.training(X, Y, X_valid, Y_valid, epochs = epochs, weight_decay = weight_decay, optimizer_name = optimizer_name,lr = lr, batch_size = batch_size,parameters=parameters)

# Calculate accuracy and losses and logging it onto wandb

test_predict = nn.feedforward(X_test)
test_loss = nn.loss(test_predict,Y_test,weight_decay,nn.W)
test_accuracy = nn.calculateAccuracy(X_test, Y_test)

list_length = len(val_loss_list)
for i in range(list_length):
  wandb.log({'validation_loss': val_loss_list[i],
            'training_loss': train_loss_list[i],
            'validation_accuracy': val_accuracy_list[i],
            'training_accuracy': train_accuracy_list[i]
            })

wandb.log({'testing_loss': test_loss,
            'testing_accuracy': test_accuracy
            })

wandb.finish()
print(f'training loss : {train_loss_list[list_length-1]} \ntraining accuracy : {train_accuracy_list[list_length-1]}\ntesing loss : {test_loss} \ntesting accuracy : {test_accuracy} \n ')



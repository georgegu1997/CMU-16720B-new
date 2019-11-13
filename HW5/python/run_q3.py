import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = None
learning_rate = None
hidden_size = 64
##########################
##### your code here #####
# print(train_x.shape) # (10800, 1024)
# print(train_y.shape) # (10800, 36)
input_size = train_x.shape[1]
output_size = train_y.shape[1]
batch_size = 50
learning_rate = 2e-3
##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
##########################
##### your code here #####
initialize_weights(input_size, hidden_size, params, "layer1")
initialize_weights(hidden_size, output_size, params, "output")
epoch_list = []
train_acc_list = []
valid_acc_list = []
train_loss_list = []
valid_loss_list = []
##########################


# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        # forward
        h1 = forward(xb,params,'layer1') # First layer
        probs = forward(h1,params,'output',softmax) # Second layer
        # loss
        loss, acc = compute_loss_and_acc(yb, probs)
        # be sure to add loss and accuracy to epoch totals
        total_loss += loss
        total_acc += acc / batch_num

        # backward
        delta1 = probs.copy()
        delta1[np.arange(probs.shape[0]),yb.argmax(axis=1)] -= 1
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)

        # apply gradient
        for k,v in sorted(list(params.items())):
            if 'grad' in k:
                name = k.split('_')[1]
                # print(np.linalg.norm(learning_rate * v))
                params[name] -= learning_rate * v
        ##########################

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
        ##########################
        ##### your code here #####
        epoch_list.append(itr+1)
        train_acc_list.append(total_acc)
        train_loss_list.append(total_loss)

        # validation forward
        h1 = forward(valid_x,params,'layer1') # First layer
        probs = forward(h1,params,'output',softmax) # Second layer
        valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs) # Loss
        valid_acc_list.append(valid_acc)
        valid_loss_list.append(valid_loss)
        ##########################

# run on validation set and report accuracy! should be above 75%
valid_acc = None
##########################
##### your code here #####
# Final training forward
h1 = forward(train_x,params,'layer1') # First layer
probs = forward(h1,params,'output',softmax) # Second layer
loss, acc = compute_loss_and_acc(train_y, probs) # loss
print("Training final: \t loss: {:.2f} \t acc : {:.2f}".format(loss,acc))

# Validation forward
h1 = forward(valid_x,params,'layer1') # First layer
probs = forward(h1,params,'output',softmax) # Second layer
valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs) # loss
print('Validation accuracy: ',valid_acc)

# Visualization of the training progress
def visualizaProgress():
    import matplotlib.pyplot as plt
    plt.subplot(211)
    plt.plot(epoch_list, train_acc_list, label="Training")
    plt.plot(epoch_list, valid_acc_list, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(212)
    plt.plot(epoch_list, train_loss_list, label="Training")
    plt.plot(epoch_list, valid_loss_list, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.show()

# visualizaProgress()
##########################

print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here
##########################
##### your code here #####

##########################

# Q3.1.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
##########################
##### your code here #####
##########################

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()

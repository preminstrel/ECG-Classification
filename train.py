import torch
from binarynet import ConvLayer_bin, FCLayer_bin
import warnings
warnings.filterwarnings("ignore")

TOPK = (1,5)

def accuracy(output, target, correct_sum, topk=(1,)):
    """Compute the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for (i,k) in enumerate(topk):
            correct_sum[i] += (correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)).item()
        return 


def get_accuracy(net, train_loader, loss_func):
    """Get the training loss and training accuracy."""
    net.eval()
    with torch.no_grad():
        train_loss = 0.
        num_batches = 0
        correct_sum = [0. for i in range(len(TOPK))]
        total = 0
        for (inputs, labels) in train_loader:               
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            accuracy(outputs, labels, correct_sum, topk=TOPK)
            total += labels.size(0)
            train_loss += loss.data.item()
            num_batches += 1
        print('training loss: ', train_loss/num_batches)
        print('training accuracy: ', [ci/total for ci in correct_sum])


def train_fullprecision(net, train_loader, loss_func, optimizer, epoch):
    """Train the original full precision network for one epoch."""
    net.train()
    train_loss = 0.
    num_batches = 0
    correct_sum = [0. for i in range(len(TOPK))]
    total = 0
    for (inputs, labels) in train_loader:               
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optimizer.zero_grad()    
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()   
        optimizer.step()
        accuracy(outputs, labels, correct_sum, topk=TOPK)
        total += labels.size(0)
        train_loss += loss.data.item()
        num_batches += 1
    print("epoch: ", epoch, ", training loss: ", train_loss/num_batches)            
    print('training accuracy: ', [ci/total for ci in correct_sum])


def train_coordinate(net, train_loader, loss_func, optimizer_w, optimizer_b, parameters_w_bin, epoch):
    """Train the coordinates for one epoch."""
    net.train()
    train_loss = 0.
    num_batches = 0
    correct_sum = [0. for i in range(len(TOPK))]
    total = 0
    for (inputs, labels) in train_loader:               
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optimizer_w.zero_grad()
        optimizer_b.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()   
        optimizer_b.step()
        optimizer_w.step(parameters_w_bin, 'coordinate')
        accuracy(outputs, labels, correct_sum, topk=TOPK)
        total += labels.size(0)
        train_loss += loss.data.item()
        num_batches += 1    
    print("epoch: ", epoch, ", training loss: ", train_loss/num_batches) 
    print('training accuracy: ', [ci/total for ci in correct_sum])
 
  
def train_basis(net, train_loader, loss_func, optimizer_w, optimizer_b, parameters_w_bin, epoch):
    """Train the binary bases (with speedup) for one epoch."""
    net.train()
    train_loss = 0.
    num_batches = 0
    correct_sum = [0. for i in range(len(TOPK))]
    total = 0
    for inputs, labels in train_loader:               
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optimizer_w.zero_grad()
        optimizer_b.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()   
        optimizer_b.step()
        optimizer_w.step(parameters_w_bin, 'basis')
        accuracy(outputs, labels, correct_sum, topk=TOPK)
        total += labels.size(0)
        train_loss += loss.data.item()
        num_batches += 1   
    print("epoch: ", epoch, ", training loss: ", train_loss/num_batches)                
    print('training accuracy: ', [ci/total for ci in correct_sum])


def train_basis_STE(net, train_loader, loss_func, optimizer_w, optimizer_b, parameters_w_bin, epoch):
    """Train the binary bases (with speedup) by STE for one epoch."""
    net.train()
    train_loss = 0.
    num_batches = 0
    correct_sum = [0. for i in range(len(TOPK))]
    total = 0
    for (inputs, labels) in train_loader:               
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optimizer_w.zero_grad()
        optimizer_b.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()   
        optimizer_b.step()
        optimizer_w.step(parameters_w_bin, 'ste')
        accuracy(outputs, labels, correct_sum, topk=TOPK)
        total += labels.size(0)
        train_loss += loss.data.item()
        num_batches += 1    
    print("epoch: ", epoch, ", training loss: ", train_loss/num_batches)                
    print('training accuracy: ', [ci/total for ci in correct_sum])


def prune(net, train_loader, loss_func, optimizer_w, optimizer_b, parameters_w_bin, pruning_rate, epoch):
    """Prune alpha for one epoch."""
    net.eval()
    train_loss = 0.
    num_batches = 0
    correct_sum = [0. for i in range(len(TOPK))]
    total = 0
    for (inputs, labels) in train_loader:               
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optimizer_w.zero_grad()
        optimizer_b.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()   
        optimizer_b.step()
        optimizer_w.step(parameters_w_bin, 'coordinate', pruning_rate)
        accuracy(outputs, labels, correct_sum, topk=TOPK)
        train_loss += loss.data.item()
        num_batches += 1
        total += labels.size(0)
    print("epoch: ", epoch, ", pruning loss: ", train_loss/num_batches)                
    print('pruning accuracy: ', [ci/total for ci in correct_sum])
    num_weight_layer = 0.
    num_bit_layer = 0.
    print('currrent number of binary filters per layer: ')
    for p_w_bin in parameters_w_bin:
        print(p_w_bin.num_bin_filter)
    print('currrent average bitwidth per layer: ')
    for p_w_bin in parameters_w_bin:
        num_weight_layer += p_w_bin.num_weight
        num_bit_layer += p_w_bin.avg_bit*p_w_bin.num_weight
        print(p_w_bin.avg_bit)
    print('currrent average bitwidth: ', num_bit_layer/num_weight_layer)

 
def initialize(net, train_loader, loss_func, structure, num_subchannel, max_bit):
    """Initialize the weight tensors of all layers to multi-bit form using structured sketching. 
    Return the iterator over all weight parameters, the iterator over all other parameters, and the iterator over the multi-bit forms of all weight parameters.  
    """
    parameters_w = []
    parameters_b = []
    parameters_w_bin = []
    i = 0
    for name, param in net.named_parameters():
        # Only initialize weight tensors to multi-bit form
        if 'weight' in name and param.dim()>1:
            parameters_w.append(param)
            #print(param.dim())
            # Initialize fully connected layers (param.dim()==2)
            if 'fc' in name or 'classifier' in name:
                print(structure[i])
                parameters_w_bin.append(FCLayer_bin(param.data, len(parameters_w)-1, structure[i], num_subchannel[i], max_bit[i]))  
                i += 1
                tmp_param = param.detach()
                tmp_param.zero_().add_(parameters_w_bin[-1].reconstruct_w())
            # Initialize convolutional layers (param.dim()==3)
            else:
                #print(len(parameters_w))
                print(structure[i])
                parameters_w_bin.append(ConvLayer_bin(param.data, len(parameters_w)-1, structure[i], max_bit[i]))    
                i += 1
                tmp_param = param.detach()
                tmp_param.zero_().add_(parameters_w_bin[-1].reconstruct_w())    
        # Maintain other parameters (e.g. bias, batch normalization) in full precision 
        else:
            parameters_b.append(param)
    net.eval()
    train_loss = 0.
    num_batches = 0
    correct_sum = [0. for i in range(len(TOPK))]
    total = 0
    for (inputs, labels) in train_loader:               
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        accuracy(outputs, labels, correct_sum, topk=TOPK)
        total += labels.size(0)
        train_loss += loss.data.item()
        num_batches += 1
    print('train loss: ', train_loss/num_batches)
    print('train accuracy: ', [ci/total for ci in correct_sum]) 
    num_weight_layer = 0.
    num_bit_layer = 0.
    print('currrent binary filter number per layer: ')
    for p_w_bin in parameters_w_bin:
        print(p_w_bin.num_bin_filter)
    print('currrent average bitwidth per layer: ')
    for p_w_bin in parameters_w_bin:
        num_weight_layer += p_w_bin.num_weight
        num_bit_layer += p_w_bin.avg_bit*p_w_bin.num_weight
        print(p_w_bin.avg_bit)
        print(p_w_bin.num_weight)
    print('currrent average bitwidth: ', num_bit_layer/num_weight_layer)
    return parameters_w, parameters_b, parameters_w_bin 
      
     
def validate(net, val_loader, loss_func):
    """Get the validation loss and validation accuracy."""
    net.eval()
    val_loss = 0.
    num_batches = 0
    correct_sum = [0. for i in range(len(TOPK))]
    total = 0
    with torch.no_grad():
        for (inputs, labels) in val_loader:
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            outputs = net(inputs)
            loss = loss_func(outputs, labels)  
            accuracy(outputs, labels, correct_sum, topk=TOPK)
            total += labels.size(0)
            val_loss += loss.data.item()
            num_batches += 1 
        print('validation loss: ', val_loss/num_batches)
        print("validation accuracy: ", [ci/total for ci in correct_sum])
        return [ci/total for ci in correct_sum]


def test(net, test_loader, loss_func):
    """Get the test loss and test accuracy."""
    net.eval()
    test_loss = 0.
    num_batches = 0
    correct_sum = [0. for i in range(len(TOPK))]
    total = 0
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            outputs = net(inputs)
            loss = loss_func(outputs, labels)  
            accuracy(outputs, labels, correct_sum, topk=TOPK)
            total += labels.size(0)
            test_loss += loss.data.item()
            num_batches += 1
        print("test loss: ", test_loss/num_batches)
        print("test accuracy: ", [ci/total for ci in correct_sum])
        

def save_model(file_name, net, optimizer_w, optimizer_b, parameters_w_bin):
    """Save the state dictionary of model and optimizers."""
    print('saving...')   
    torch.save({
        'net_state_dict': net.state_dict(),
        'optimizer_w_state_dict': optimizer_w.state_dict(),
        'optimizer_b_state_dict': optimizer_b.state_dict(),
        'parameters_w_bin': parameters_w_bin,
        }, file_name)


def save_model_ori(file_name, net, optimizer):
    """Save the state dictionary of model and optimizer for full precision training."""
    print('saving...')   
    torch.save({
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, file_name)
def save_model_simple(file_name, net):
    """Save the state dictionary of model and optimizer for final training."""
    print('saving...')   
    torch.save({
        'net_state_dict': net.state_dict(),
        #'optimizer_state_dict': optimizer.state_dict(),
        }, file_name)
  

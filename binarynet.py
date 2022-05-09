import torch
from myoptimizer import ALQ_optimizer


REL_NORM_THRES = 1e-6


def construct_bit_table(bit):
    """Construct a look-up-table to store bitwise values of all intergers given a bitwidth."""
    bit_table = -torch.ones((2**bit, bit), dtype=torch.int8)
    for i in range(1,2**bit):
        for j in range(bit):
            if (i & (1<<j)):
                bit_table[i,j] = 1
    return bit_table.to('cuda')


def binarize(input_t): 
    """Binarize input tensor."""
    dim = input_t.nelement()
    output_t = torch.ones(dim)
    output_t[input_t<0] = -1
    return output_t


def transform_bin_basis(w_vec, max_dim, rel_norm_thres=REL_NORM_THRES):
    """Transform a full precision weight vector into multi-bit form, i.e. binary bases and coordiantes."""
    # Reshape the coordinates vector in w domain
    crd_w = w_vec.detach().view(-1,1) # 展开成一个列向量
    #print(crd_w)    
    # Get the dimensionality in w domain
    dim_w = crd_w.nelement()  #列向量的长度（16）
    #print(dim_w)
    # Determine the max number of dimensionality in alpha domain
    if dim_w <= max_dim:
        max_dim_alpha = dim_w
    else:
        max_dim_alpha = max_dim
    # Initialize binary basis matrix in alpha domain
    bin_basis_alpha = torch.zeros((dim_w, max_dim_alpha))
    # Initialize coordinates vector in alpha domain
    crd_alpha = torch.zeros(max_dim_alpha) 
    res = crd_w.detach()
    res_L2Norm_square = torch.sum(torch.pow(res,2))
    ori_L2Norm_square = torch.sum(torch.pow(crd_w,2))  
    for i in range(max_dim_alpha):
        if res_L2Norm_square/ori_L2Norm_square < rel_norm_thres:
            break
        new_bin_basis = binarize(res.view(-1))
        bin_basis_alpha[:,i] = new_bin_basis 
        B_ = bin_basis_alpha[:,:i+1]
        # Find the optimal coordinates in the space spanned by B_ 
        alpha_ = torch.mm(torch.inverse(torch.mm(torch.t(B_),B_)),torch.mm(torch.t(B_),crd_w)) 
        # Compute the residual (orthogonal to the space spanned by B_)
        res = crd_w - torch.mm(B_, alpha_)
        crd_alpha[:i+1] = alpha_.view(-1)
        res_L2Norm_square = torch.sum(torch.pow(res,2))   
    ind_neg = crd_alpha < 0
    crd_alpha[ind_neg] = -crd_alpha[ind_neg]
    bin_basis_alpha[:,ind_neg] = -bin_basis_alpha[:,ind_neg]
    # Get the valid indexes 
    ind_valid = crd_alpha != 0
    # Get the valid dimensionality in alpha domain
    dim_alpha = torch.sum(ind_valid) 
    sorted_ind = torch.argsort(crd_alpha[ind_valid])
    #print(dim_alpha)  #alpha 的长度 6
    #print(bin_basis_alpha)  # 二值化网络权重 二维tensor
    #print(ind_valid)
    #print(sorted_ind)
    if dim_alpha == 0:
        return [], [], 0
    else:
        return bin_basis_alpha[:,ind_valid][:,sorted_ind].to(torch.int8), crd_alpha[ind_valid][sorted_ind], dim_alpha


class ConvLayer_bin(object):
    """This class defines the multi-bit form of the weight tensor of a convolutional layer used in ALQ. 

    Arguments:
        w_ori (float tensor): the 3-dim pretrained weight tensor of a convolutional layer.
        ind_layer (int): the index of this layer in the network.
        structure (string): the structure used for grouping the weights in this layer, optional values: 'kernelwise', 'pixelwise', 'channelwise'.
        max_bit (int): the maximum bitwidth used in initialization.
    """
    def __init__(self, w_ori, ind_layer, structure, max_bit):
        # The layer type
        self.layer_type = 'conv'
        # The shape of the weight tensor of this layer
        self.tensor_shape = w_ori.size()
        # The maintained full precision weight tensor of this layer used in STE
        self.w_FP = w_ori.clone().to('cuda')
        # The index of this layer in the network
        self.ind_layer = ind_layer
        # The structure used for grouping the weights in this layer
        self.structure = structure
        # The maximum bitwidth used in initialization
        self.max_bit = max_bit
        # The binary bases, the coordinates, and the mask (only for parallel computing purposes) of each group
        self.B, self.alpha, self.mask = self.structured_sketch()
        # The total number of binary filters in this layer, namely the total number of (valid) alpha's
        self.num_bin_filter = torch.sum(self.mask)
        # The average bitwidth of this layer
        self.avg_bit = self.num_bin_filter.float()/(self.mask.size(0)*self.mask.size(1))
        # The total number of weights of this layer
        self.num_weight = self.w_FP.nelement()
        # The used look-up-table for bitwise values
        self.bit_table = construct_bit_table(self.max_bit)
        
    def structured_sketch(self):
        """Initialize the weight tensor using structured sketching. 
        Namely, structure the weights in groupwise, and quantize each group's weights in multi-bit form w.r.t. the reconstruction error.
        Return the binary bases, the coordinates, and the mask (only for parallel computing purposes) of each group. 
        """
        w_cpu = self.w_FP.to('cpu')
        if self.structure == 'kernelwise':
            B = torch.zeros((self.tensor_shape[0],self.tensor_shape[1],self.max_bit,self.tensor_shape[2])).to(torch.int8)
            alpha = torch.zeros((self.tensor_shape[0],self.tensor_shape[1],self.max_bit,1)).to(torch.float32)
            mask =  torch.zeros((self.tensor_shape[0],self.tensor_shape[1],self.max_bit,1)).to(torch.bool)
        elif self.structure == 'pixelwise':
            B = torch.zeros((self.tensor_shape[0],self.tensor_shape[2],self.max_bit,self.tensor_shape[1])).to(torch.int8)
            alpha = torch.zeros((self.tensor_shape[0],self.tensor_shape[2],self.max_bit,1)).to(torch.float32)
            mask =  torch.zeros((self.tensor_shape[0],self.tensor_shape[2],self.max_bit,1)).to(torch.bool)
        elif self.structure == 'channelwise':
            B = torch.zeros((self.tensor_shape[0],1,self.max_bit,self.tensor_shape[1]*self.tensor_shape[2])).to(torch.int8)
            alpha = torch.zeros((self.tensor_shape[0],1,self.max_bit)).to(torch.float32)
            mask =  torch.zeros((self.tensor_shape[0],1,self.max_bit)).to(torch.bool)
        for k in range(self.tensor_shape[0]):
            if self.structure == 'kernelwise':
                for q in range(self.tensor_shape[1]):
                    bin_basis, crd, dim = transform_bin_basis(w_cpu[k,q,:].view(-1), self.max_bit)
                    #print(crd) # tensor([0.0030, 0.0093, 0.0231, 0.0371, 0.0655, 0.1454])
                    #print(dim) # tensor(6)
                    mask[k,q,:dim,0] = 1
                    #print(B.shape) # torch.Size([8, 1, 6, 16])
                    #print(t(bin_basis).shape)    ## 这里我进行了改动
                    #print(torch.t(bin_basis).shape) # torch.Size([6, 16])
                    B[k,q,:dim,:] = torch.t(bin_basis)
                    #print(alpha.shape)
                    alpha[k,q,:dim,0] = crd
            elif self.structure == 'pixelwise':
                for h in range(self.tensor_shape[2]):
                    for w in range(self.tensor_shape[3]):
                        bin_basis, crd, dim = transform_bin_basis(w_cpu[k,:,w].view(-1), self.max_bit)
                        mask[k,h*self.tensor_shape[3]+w,:dim,0] = 1
                        B[k,h*self.tensor_shape[3]+w,:dim,:] = torch.t(bin_basis)
                        alpha[k,h*self.tensor_shape[3]+w,:dim,0] = crd
            if self.structure == 'channelwise':
                bin_basis, crd, dim = transform_bin_basis(w_cpu[k,:,:,:].view(-1), self.max_bit)
                mask[k,0,:dim,0] = 1
                B[k,0,:dim,:] = torch.t(bin_basis)
                alpha[k,0,:dim,0] = crd
        return B.to('cuda'), alpha.to('cuda'), mask.to('cuda')

    def reconstruct_w(self):
        """Reconstruct the weight tensor from the current quantization.
        Return the reconstructed weight tensor of this layer, i.e. \hat{w}.
        """
        #print(self.B.shape)
        #print(self.alpha.shape)
        #print(self.mask.shape)
        w_bin = torch.sum(self.B.float()*(self.alpha*self.mask.float()),dim=2)
        #print(w_bin)
        #print(tensor_shape)
        if self.structure == 'kernelwise':
            return w_bin.reshape(self.tensor_shape)
        elif self.structure == 'pixelwise':
            return torch.transpose(w_bin,1,2).reshape(self.tensor_shape)
        elif self.structure == 'channelwise':
            return w_bin.reshape(self.tensor_shape)

    def update_w_FP(self, w_FP_new=None):
        """Update the full precision weight tensor.
        In STE with loss-aware optimization, w_FP is the maintained full precision weight tensor.
        In ALQ optimization, w_FP is used to store the reconstructed weight tensor from the current quantization. 
        """
        if w_FP_new is not None:
            self.w_FP.add_(w_FP_new)
        else:
            self.w_FP.zero_().add_(self.reconstruct_w())

    def construct_grad_alpha(self, grad_w):
        """Compute and return the gradient (or the first momentum) in alpha domain w.r.t the loss.
        """
        if self.structure == 'kernelwise':
            return torch.matmul(self.B.float(), grad_w.reshape((self.tensor_shape[0],self.tensor_shape[1],-1,1)))*self.mask.float()
        elif self.structure == 'pixelwise':
            return torch.matmul(self.B.float(), torch.transpose(grad_w.reshape((self.tensor_shape[0],self.tensor_shape[1],-1,1)), 1,2) )*self.mask.float()
        elif self.structure == 'channelwise':
            return torch.matmul(self.B.float(), grad_w.reshape((self.tensor_shape[0],1,-1,1)))*self.mask.float()

    def construct_hessian_alpha(self, diag_hessian_w):
        """Compute and return the diagonal Hessian (or the second momentum) in alpha domain w.r.t the loss.
        """
        if self.structure == 'kernelwise':
            diag_hessian = torch.matmul(self.B.float()*diag_hessian_w.reshape((self.tensor_shape[0],self.tensor_shape[1],1,-1)), torch.transpose(self.B,2,3).float())
            return torch.diagonal(diag_hessian,dim1=-2,dim2=-1).unsqueeze(-1)*self.mask.float()
        elif self.structure == 'pixelwise':
            diag_hessian = torch.matmul(self.B.float()*torch.transpose(diag_hessian_w.reshape((self.tensor_shape[0],self.tensor_shape[1],1,-1)), 1,3), torch.transpose(self.B,2,3).float())
            return torch.diagonal(diag_hessian,dim1=-2,dim2=-1).unsqueeze(-1)*self.mask.float()
        elif self.structure == 'channelwise':
            diag_hessian = torch.matmul(self.B.float()*diag_hessian_w.reshape((self.tensor_shape[0],1,1,-1)), torch.transpose(self.B,2,3).float())
            return torch.diagonal(diag_hessian,dim1=-2,dim2=-1).unsqueeze(-1)*self.mask.float()

    def sort_importance_bin_filter(self, grad_alpha, diag_hessian_alpha, num_top):
        """Compute and sort the importance of binary filters (alpha's) in this layer.
        The importance is defined by the modeled loss increment caused by pruning each individual alpha.
        Return the selected num_top alpha's with the least importance.
        """
        delta_loss_prune = -grad_alpha*self.alpha+0.5*torch.pow(self.alpha,2)*diag_hessian_alpha
        sorted_ind = torch.argsort(delta_loss_prune[self.mask].view(-1))
        top_importance_list = torch.tensor([[self.ind_layer, sorted_ind[i], delta_loss_prune.view(-1)[sorted_ind[i]]] for i in range(num_top)])  
        return top_importance_list
                
    def prune_alpha(self, ind_prune): 
        """Prune the cooresponding alpha's of this layer give the indexes.
        """
        num_bin_filter_ = torch.sum(self.mask)
        self.mask.view(-1)[self.mask.view(-1).nonzero().view(-1)[ind_prune]]=0   
        self.B *= self.mask.char()
        self.alpha *= self.mask.float()  
        self.num_bin_filter = torch.sum(self.mask)  
        self.avg_bit = self.num_bin_filter.float()/(self.mask.size(0)*self.mask.size(1))
        if num_bin_filter_-self.num_bin_filter != ind_prune.size(0):
            print('wrong pruning')
            return False
        return True
        
    def optimize_bin_basis(self, pseudo_grad, pseudo_hessian):
        """Take one optimization step on the binary bases of this layer while fixing coordinates.
        """
        # Compute the target weight tensor, i.e. the optimal point in w domain according to the quadratic model function 
        target_w = self.w_FP-pseudo_grad/pseudo_hessian
        if self.structure == 'kernelwise':
            all_disc_w = torch.matmul(self.bit_table.view((1,1,self.bit_table.size(0),self.bit_table.size(1))).float(),self.alpha)
            ind_opt = torch.argmin(torch.abs(target_w.view((self.tensor_shape[0],self.tensor_shape[1],1,-1)) - all_disc_w), dim=2)
            self.B = torch.transpose((self.bit_table[ind_opt.view(-1),:]).view(self.tensor_shape[0],self.tensor_shape[1],self.tensor_shape[2],self.max_bit), 2,3)
            self.B *= self.mask.char()
        elif self.structure == 'pixelwise':
            all_disc_w = torch.matmul(self.bit_table.view((1,1,self.bit_table.size(0),self.bit_table.size(1))).float(),self.alpha)
            ind_opt = torch.argmin(torch.abs(torch.transpose(target_w.view((self.tensor_shape[0],self.tensor_shape[1],1,-1)), 1,3) - all_disc_w), dim=2)
            self.B = torch.transpose((self.bit_table[ind_opt.view(-1),:]).view(self.tensor_shape[0],self.tensor_shape[2]*self.tensor_shape[3],self.tensor_shape[1],self.max_bit), 2,3)
            self.B *= self.mask.char()
        elif self.structure == 'channelwise':
            all_disc_w = torch.matmul(self.bit_table.view((1,1,self.bit_table.size(0),self.bit_table.size(1))).float(),self.alpha)
            ind_opt = torch.argmin(torch.abs(target_w.view((self.tensor_shape[0],1,1,-1)) - all_disc_w), dim=2)
            self.B = torch.transpose((self.bit_table[ind_opt.view(-1),:]).view(self.tensor_shape[0],1,self.tensor_shape[1]*self.tensor_shape[2]*self.tensor_shape[3],self.max_bit), 2,3)
            self.B *= self.mask.char()
        return True
            
    def speedup(self, pseudo_grad, pseudo_hessian):
        """Speed up the optimization on binary bases, i.e. take a following optimization step on coordinates while fixing binary bases. 
        """
        revised_grad_w = -pseudo_hessian*self.w_FP+pseudo_grad
        if self.structure == 'kernelwise':
            revised_hessian = torch.matmul(self.B.float()*pseudo_hessian.view((self.tensor_shape[0],self.tensor_shape[1],1,-1)),torch.transpose(self.B,2,3).float())
            revised_hessian += torch.diag_embed(1+1e-6-(self.mask.float().squeeze(-1))) 
            revised_grad = torch.matmul(self.B.float(),revised_grad_w.view((self.tensor_shape[0],self.tensor_shape[1],-1,1)))
            self.alpha = -torch.matmul(torch.inverse(revised_hessian),revised_grad)
        elif self.structure == 'pixelwise':
            revised_hessian = torch.matmul(self.B.float()*torch.transpose(pseudo_hessian.view((self.tensor_shape[0],self.tensor_shape[1],1,-1)),1,3),torch.transpose(self.B,2,3).float())
            revised_hessian += torch.diag_embed(1+1e-6-(self.mask.float().squeeze(-1)))
            revised_grad = torch.matmul(self.B.float(),torch.transpose(revised_grad_w.view((self.tensor_shape[0],self.tensor_shape[1],-1,1)),1,2))
            self.alpha = -torch.matmul(torch.inverse(revised_hessian),revised_grad)
        elif self.structure == 'channelwise':
            revised_hessian = torch.matmul(self.B.float()*pseudo_hessian.view((self.tensor_shape[0],1,1,-1)),torch.transpose(self.B,2,3).float())
            revised_hessian += torch.diag_embed(1+1e-6-(self.mask.float().squeeze(-1))) 
            revised_grad = torch.matmul(self.B.float(),revised_grad_w.view((self.tensor_shape[0],1,-1,1)))
            self.alpha = -torch.matmul(torch.inverse(revised_hessian),revised_grad)
        self.alpha *= self.mask.float()
        ind_neg = self.alpha<0
        self.alpha[ind_neg] *= -1
        self.B.contiguous().view(-1,self.B.size(-1))[ind_neg.view(-1),:] *= -1
        self.num_bin_filter = torch.sum(self.mask)
        self.avg_bit = self.num_bin_filter.float()/(self.mask.size(0)*self.mask.size(1))
        return True
 
    
class FCLayer_bin(object):
    """This class defines the multi-bit form of the weight tensor of a convolutional layer used in ALQ. 

    Arguments:
        w_ori (float tensor): the 4-dim pretrained weight tensor of a convolutional layer.
        ind_layer (int): the index of this layer in the network.
        structure (string): the structure used for grouping the weights in this layer, optional values: 'subchannelwise'.
        max_bit (int): the maximum bitwidth used in initialization.
    """
    def __init__(self, w_ori, ind_layer, structure, num_subchannel, max_bit):
        # The layer type
        self.layer_type = 'fc'
        # The shape of the weight tensor of this layer
        self.tensor_shape = w_ori.size()
        # The maintained full precision weight tensor of this layer used in STE
        self.w_FP = w_ori.clone().to('cuda')
        # The index of this layer in the network
        self.ind_layer = ind_layer
        # The structure used for grouping the weights in this layer
        self.structure = structure
        # The maximum bitwidth used in initialization
        self.max_bit = max_bit
        # The number of groups in each channel, i.e. the number of subchannels 
        self.num_subchannel = num_subchannel
        # The number of weights in each subchannel
        self.num_w_subc = int(self.tensor_shape[1]/self.num_subchannel)
        # The binary bases, the coordinates, and the mask (only for parallel computing purposes) of each group
        self.B, self.alpha, self.mask = self.structured_sketch()
        # The total number of binary filters in this layer, namely the total number of (valid) alpha's
        self.num_bin_filter = torch.sum(self.mask)
        # The average bitwidth of this layer
        self.avg_bit = self.num_bin_filter.float()/(self.mask.size(0)*self.mask.size(1))
        # The total number of weights of this layer
        self.num_weight = self.w_FP.nelement()
        # The used look-up-table for bitwise values
        self.bit_table = construct_bit_table(self.max_bit)
        
    def structured_sketch(self):
        """Initialize the weight tensor using structured sketching. 
        Namely, structure the weights in groupwise, and quantize each group's weights in multi-bit form w.r.t. the reconstruction error.
        Return the binary bases, the coordinates, and the mask (only for parallel computing purposes) of each group. 
        """
        w_cpu = self.w_FP.to('cpu')
        B = torch.zeros((self.tensor_shape[0],self.num_subchannel,self.max_bit,self.num_w_subc)).to(torch.int8)
        alpha = torch.zeros((self.tensor_shape[0],self.num_subchannel,self.max_bit,1)).to(torch.float32)
        mask =  torch.zeros((self.tensor_shape[0],self.num_subchannel,self.max_bit,1)).to(torch.bool)
        for k in range(self.tensor_shape[0]):
            for (q,i) in enumerate(range(0,self.tensor_shape[1],self.num_w_subc)):
                bin_basis, crd, dim = transform_bin_basis(w_cpu[k,i:i+self.num_w_subc].view(-1), self.max_bit)
                mask[k,q,:dim,0] = 1
                B[k,q,:dim,:] = torch.t(bin_basis)
                alpha[k,q,:dim,0] = crd
        return B.to('cuda'), alpha.to('cuda'), mask.to('cuda')

    def reconstruct_w(self):
        """Reconstruct the weight tensor from the current quantization.
        Return the reconstructed weight tensor of this layer, i.e. \hat{w}.
        """
        w_bin = torch.sum(self.B.float()*(self.alpha*self.mask.float()),dim=2)
        return w_bin.reshape(self.tensor_shape)
    
    def update_w_FP(self, w_FP_new=None):
        """Update the full precision weight tensor.
        In STE with loss-aware optimization, w_FP is the maintained full precision weight tensor.
        In ALQ optimization, w_FP is used to store the reconstructed weight tensor from the current quantization. 
        """
        if w_FP_new is not None:
            self.w_FP.add_(w_FP_new)
        else:
            self.w_FP.zero_().add_(self.reconstruct_w())        
    
    def construct_grad_alpha(self, grad_w):
        """Compute and return the gradient (or the first momentum) in alpha domain w.r.t the loss.
        """
        return torch.matmul(self.B.float(), grad_w.reshape((self.tensor_shape[0],self.num_subchannel,self.num_w_subc,1)))*self.mask.float()
        
    def construct_hessian_alpha(self, diag_hessian_w):
        """Compute and return the diagonal Hessian (or the second momentum) in alpha domain w.r.t the loss.
        """
        diag_hessian_alpha = torch.matmul(self.B.float()*diag_hessian_w.reshape((self.tensor_shape[0],self.num_subchannel,1,self.num_w_subc)), torch.transpose(self.B,2,3).float())
        return torch.diagonal(diag_hessian_alpha,dim1=-2,dim2=-1).unsqueeze(-1)*self.mask.float()
        
    def sort_importance_bin_filter(self, grad_alpha, diag_hessian_alpha, num_top):
        """Compute and sort the importance of binary filters (alpha's) in this layer.
        The importance is defined by the modeled loss increment caused by pruning each individual alpha.
        Return the selected num_top alpha's with the least importance.
        """
        delta_loss_prune = -grad_alpha*self.alpha+0.5*torch.pow(self.alpha,2)*diag_hessian_alpha
        sorted_ind = torch.argsort(delta_loss_prune[self.mask].view(-1))
        top_importance_list = torch.tensor([[self.ind_layer, sorted_ind[i], delta_loss_prune.view(-1)[sorted_ind[i]]] for i in range(num_top)])  
        return top_importance_list
                   
    def prune_alpha(self, ind_prune): 
        """Prune the cooresponding alpha's of this layer give the indexes.
        """
        num_bin_filter_ = torch.sum(self.mask)
        self.mask.view(-1)[self.mask.view(-1).nonzero().view(-1)[ind_prune]]=0   
        self.B *= self.mask.char()
        self.alpha *= self.mask.float()   
        self.num_bin_filter = torch.sum(self.mask) 
        self.avg_bit = self.num_bin_filter.float()/(self.mask.size(0)*self.mask.size(1))
        if num_bin_filter_-self.num_bin_filter != ind_prune.size(0):
            print('wrong pruning')
            return False
        return True    
                   
    def optimize_bin_basis(self, pseudo_grad, pseudo_hessian):
        """Take one optimization step on the binary bases of this layer while fixing coordinates.
        """
        # Compute the target weight tensor, i.e. the optimal point in w domain according to the quadratic model function 
        target_w = self.w_FP-pseudo_grad/pseudo_hessian
        all_disc_w = torch.matmul(self.bit_table.view((1,1,self.bit_table.size(0),self.bit_table.size(1))).float(),self.alpha)
        ind_opt = torch.argmin(torch.abs(target_w.view((self.tensor_shape[0],self.num_subchannel,1,-1)) - all_disc_w), dim=2)
        self.B = torch.transpose((self.bit_table[ind_opt[:],:]).view(self.tensor_shape[0],self.num_subchannel,self.num_w_subc,self.max_bit), 2,3)
        self.B *= self.mask.char()
        return True
                               
    def speedup(self, pseudo_grad, pseudo_hessian):
        """Speed up the optimization on binary bases, i.e. take a following optimization step on coordinates while fixing binary bases. 
        """
        revised_grad_w = -pseudo_hessian*self.w_FP+pseudo_grad
        revised_hessian = torch.matmul(self.B.float()*pseudo_hessian.view((self.tensor_shape[0],self.num_subchannel,1,-1)),torch.transpose(self.B,2,3).float())
        revised_hessian += torch.diag_embed(1+1e-6-(self.mask.float().squeeze(-1))) 
        revised_grad = torch.matmul(self.B.float(),revised_grad_w.view((self.tensor_shape[0],self.num_subchannel,-1,1)))
        self.alpha = -torch.matmul(torch.inverse(revised_hessian),revised_grad)
        self.alpha *= self.mask.float()
        ind_neg = self.alpha<0
        self.alpha[ind_neg] *= -1
        self.B.contiguous().view(-1,self.B.size(-1))[ind_neg.view(-1),:] *= -1
        self.num_bin_filter = torch.sum(self.mask)
        self.avg_bit = self.num_bin_filter.float()/(self.mask.size(0)*self.mask.size(1))
        return True
                               
    
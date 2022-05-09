import math
import random
import torch
from torch.optim.optimizer import Optimizer, required


class ALQ_optimizer(Optimizer):

    """Implement ALQ optimizer.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay (float, optional): weight decay (L2 regularization) (default: 0).
        
    Reference:
        Adam optimizer by Pytorch:
        https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam
        On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.):
        # Check the validity 
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(ALQ_optimizer, self).__init__(params, defaults)
        
    def __setstate__(self, state):
        super(ALQ_optimizer, self).__setstate__(state)
           
    def step(self, params_bin, mode, pruning_rate=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            # Check if this is a pruning step
            if pruning_rate is not None:
                importance_list = torch.tensor([])
            
            for i, (p_bin, p) in enumerate(zip(params_bin, group['params'])):
                if p.grad is None:
                    continue
                
                # Compute the gradient in both w domain and alpha domain
                grad = p.grad.data
                grad_alpha = p_bin.construct_grad_alpha(grad)
                state = self.state[p]
                
                # Initialize the state parameters in both w domain and alpha domain
                if len(state) == 0:
                    state['step_alpha'] = 0
                    state['exp_avg_alpha'] = torch.zeros_like(p_bin.alpha)
                    state['exp_avg_sq_alpha'] = torch.zeros_like(p_bin.alpha)
                    state['max_exp_avg_sq_alpha'] = torch.zeros_like(p_bin.alpha)
                    
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                if mode == 'coordinate':
                    # Update the state parameters in w domain
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    beta1, beta2 = group['betas']
                    state['step'] += 1
                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    # Maintain the maximum of all second moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)

                    # Update the state parameters in alpha domain
                    exp_avg_alpha, exp_avg_sq_alpha = state['exp_avg_alpha'], state['exp_avg_sq_alpha']
                    max_exp_avg_sq_alpha = state['max_exp_avg_sq_alpha']
                    state['step_alpha'] += 1
                    # L2 regularization on coordinates (in alpha domain)
                    if group['weight_decay'] != 0:
                        grad_alpha = grad_alpha.add(p_bin.alpha, alpha=group['weight_decay'])
                    # Decay the first and second moment running average coefficient
                    exp_avg_alpha.mul_(beta1).add_(1 - beta1, grad_alpha)
                    exp_avg_sq_alpha.mul_(beta2).addcmul_(1 - beta2, grad_alpha, grad_alpha)
                    # Maintain the maximum of all second moment running avg. till now
                    torch.max(max_exp_avg_sq_alpha, exp_avg_sq_alpha, out=max_exp_avg_sq_alpha)
                    # Use the max. for normalizing running avg. of gradient
                    denom_alpha = max_exp_avg_sq_alpha.sqrt().add_(group['eps'])
                    bias_correction1 = 1 - beta1 ** state['step_alpha']
                    bias_correction2 = 1 - beta2 ** state['step_alpha']

                    # Compute the pseudo gradient and the pseudo diagonal Hessian 
                    pseudo_grad_alpha = (group['lr'] / bias_correction1) * exp_avg_alpha 
                    pseudo_hessian_alpha = denom_alpha.div(math.sqrt(bias_correction2))
                    
                    # Check if this is a pruning step
                    if pruning_rate is not None:
                        # Compute the integer used to determine the number of selected alpha's in this layer
                        float_tmp = p_bin.num_bin_filter.item()*pruning_rate[0]
                        int_tmp = int(float_tmp)
                        if random.random()<float_tmp-int_tmp:
                            int_tmp += 1 
                        # Sort the importance of binary filters (alpha's) in this layer and select Top-k% (int_tmp) unimportant ones
                        p_bin_importance_list = p_bin.sort_importance_bin_filter(pseudo_grad_alpha, pseudo_hessian_alpha, int_tmp) 
                        importance_list = torch.cat((importance_list,p_bin_importance_list), 0) 
                    else:
                        # Take one optimization step on coordinates
                        p_bin.alpha.add_(-pseudo_grad_alpha/pseudo_hessian_alpha)
                        # Reconstruct the weight tensor from the current quantization
                        p_bin.update_w_FP()
                        tmp_p = p.detach()
                        tmp_p.zero_().add_(p_bin.w_FP.data)
                                     
                elif mode == 'basis':
                    # Update the state parameters in w domain
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    beta1, beta2 = group['betas']
                    state['step'] += 1
                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    # Maintain the maximum of all second moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    # Compute the pseudo gradient and the pseudo diagonal Hessian 
                    pseudo_grad = (group['lr'] / bias_correction1) * exp_avg 
                    pseudo_hessian = denom.div(math.sqrt(bias_correction2))
                    # Take one optimization step on binary bases
                    p_bin.optimize_bin_basis(pseudo_grad, pseudo_hessian)
                    # Speed up with an optimization step on coordinates
                    p_bin.speedup(pseudo_grad, pseudo_hessian)
                    # Reconstruct the weight tensor from the current quantization
                    p_bin.update_w_FP()
                    tmp_p = p.detach()
                    tmp_p.zero_().add_(p_bin.w_FP.data)

                    # Update the state parameters in alpha domain (approximately)
                    state['step_alpha'] += 1
                    state['exp_avg_alpha'] = p_bin.construct_grad_alpha(exp_avg)
                    state['exp_avg_sq_alpha'] = p_bin.construct_hessian_alpha(exp_avg_sq)
                    state['max_exp_avg_sq_alpha'] = p_bin.construct_hessian_alpha(max_exp_avg_sq)
                    
                elif mode == 'ste':
                    # Update the state parameters in w domain
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    beta1, beta2 = group['betas']
                    state['step'] += 1
                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    # Maintain the maximum of all second moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    # Compute the pseudo gradient and the pseudo diagonal Hessian 
                    pseudo_grad = (group['lr'] / bias_correction1) * exp_avg 
                    pseudo_hessian = denom.div(math.sqrt(bias_correction2))
                    
                    # Take one optimization step on binary bases
                    p_bin.optimize_bin_basis(pseudo_grad, pseudo_hessian)
                    # Speed up with an optimization step on coordinates
                    p_bin.speedup(pseudo_grad, pseudo_hessian)
                    # Update the maintained full precision weights
                    p_bin.update_w_FP(-pseudo_grad/pseudo_hessian)
                    # Reconstruct the weight tensor from the current quantization
                    tmp_p = p.detach()
                    tmp_p.zero_().add_(p_bin.reconstruct_w())

                    # Update the state parameters in alpha domain (approximately)
                    state['step_alpha'] += 1
                    state['exp_avg_alpha'] = p_bin.construct_grad_alpha(exp_avg)
                    state['exp_avg_sq_alpha'] = p_bin.construct_hessian_alpha(exp_avg_sq)
                    state['max_exp_avg_sq_alpha'] = p_bin.construct_hessian_alpha(max_exp_avg_sq)
            
            # Check if this is a pruning step        
            if pruning_rate is not None:
                # Resort the importance of selected binary filters (alpha's) over all layers 
                sorted_ind = torch.argsort(importance_list[:,-1])
                # Compute the number of pruned alpha's in this iteration
                # Note that unlike the paper, M_p varies over iterations here, but this does not influence the pruning schedule. 
                M_p = int(sorted_ind.nelement()*pruning_rate[1])
                # Determine indexes of alpha's to be pruned
                ind_prune = sorted_ind[:M_p]
                list_prune = importance_list[ind_prune,:]
                # Prune alpha's in each layer and reconstruct the weight tensor
                for i, (p_bin, p) in enumerate(zip(params_bin, group['params'])):
                    p_bin.prune_alpha((torch.sort(list_prune[list_prune[:,0]==i,1])[0]).to(torch.int64))
                    p_bin.update_w_FP()
                    tmp_p = p.detach()
                    tmp_p.zero_().add_(p_bin.w_FP.data)
        return loss



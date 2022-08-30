import torch
import torch.distributions as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import cat, zeros, stack
from torch.autograd import grad
from backpack import extend, backpack, extensions
from tqdm import tqdm, trange
import numpy as np


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


        ###############################
        #    Model training utils     #
        ###############################
        

class EarlyStopping:

    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):

        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = - val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):

        '''Saves model when validation loss decrease.'''

        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} ---> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
        
        
        
def train_model(model, X_train, X_val, y_train, y_val, patience, n_epochs, optimizer, criterion, verbose=False):
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=verbose)
    
    for epoch in tqdm(range(1, n_epochs + 1)):

        ###################
        # train the model #
        ###################
        model.train() # prep model for training
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(X_train)
        # calculate the loss
        loss = criterion(output, y_train)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # record training loss
        train_losses.append(loss.item())

        ######################    
        # validate the model #
        ######################
        #model.eval() # prep model for evaluation
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(X_val)
        # calculate the loss
        loss = criterion(output, y_val)
        # record validation loss
        valid_losses.append(loss.item())

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(n_epochs))
        
        if verbose:
          print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                      f'train_loss: {train_loss:.5f} ' +
                      f'valid_loss: {valid_loss:.5f}')
          
          print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    return  model, avg_train_losses, avg_valid_losses
    
    
    
        ###############################
        # Laplace approximation utils #
        ###############################


def exact_hessian(f, parameters, show_progress=False):
    r"""Compute all second derivatives of a scalar w.r.t. `parameters`.
​
  The order of parameters corresponds to a one-dimensional
  vectorization followed by a concatenation of all tensors in
  `parameters`.
​
  Parameters
  ----------
  f : scalar torch.Tensor
    Scalar PyTorch function/tensor.
  parameters : list or tuple or iterator of torch.Tensor
    Iterable object containing all tensors acting as variables of `f`.
  show_progress : bool
    Show a progressbar while performing the computation.
​
  Returns
  -------
  torch.Tensor
    Hessian of `f` with respect to the concatenated version
    of all flattened quantities in `parameters`
  Note
  ----
  The parameters in the list are all flattened and concatenated
  into one large vector `theta`. Return the matrix :math:`d^2 E /
  d \theta^2` with
  .. math::
​
    (d^2E / d \theta^2)[i, j] = (d^2E / d \theta[i] d \theta[j]).
​
  The code is a modified version of
  https://discuss.pytorch.org/t/compute-the-hessian-matrix-of-a-
  network/15270/3
  """
  
    params = list(parameters)
    if not all(p.requires_grad for p in params):
        raise ValueError("All parameters have to require_grad")
    df = grad(f, params, create_graph=True)
    # flatten all parameter gradients and concatenate into a vector
    dtheta = None
    for grad_f in df:
        dtheta = (
            grad_f.contiguous().view(-1)
            if dtheta is None
            else cat([dtheta, grad_f.contiguous().view(-1)])
        )
    # compute second derivatives
    hessian_dim = dtheta.size(0)
    hessian = zeros(hessian_dim, hessian_dim)
    progressbar = tqdm(
        iterable=range(hessian_dim),
        total=hessian_dim,
        desc="[exact] Full Hessian",
        disable=(not show_progress),
    )
    for idx in progressbar:
        df2 = grad(dtheta[idx], params, create_graph=True)
        d2theta = None
        for d2 in df2:
            d2theta = (
                d2.contiguous().view(-1)
                if d2theta is None
                else cat([d2theta, d2.contiguous().view(-1)])
            )
        hessian[idx] = d2theta
    return hessian


def exact_hessian_diagonal_blocks(f, parameters, show_progress=True):
    """Compute diagonal blocks of a scalar function's Hessian.
​
    Parameters
    ----------
    f : scalar of torch.Tensor
        Scalar PyTorch function
    parameters : list or tuple or iterator of torch.Tensor
        List of parameters whose second derivatives are to be computed
        in a blockwise manner
    show_progress : bool, optional
        Show a progressbar while performing the computation.
​
    Returns
    -------
    list of torch.Tensor
        Hessian blocks. The order is identical to the order specified
        by `parameters`
​
    Note
    ----
    For each parameter, `exact_hessian` is called.
    """
    
    return [exact_hessian(f, [p], show_progress=show_progress)
            for p in parameters]
            
            

@torch.no_grad()
def laplace_predict(x, model, weights, U, V):

  """
  Make predictions by approximating the posterior predictive distribution using MC integration, where weights are sampled using a Kronecker-factored Laplace approximation to the last       layer weights posterior. 
  For more details of the Kornecker-factored Laplace approximation: see Appendix B.1 of https://arxiv.org/abs/2002.10118
  """

  phi = model.feature_extr(x)
  
  # MAP prediction (mean vector in Laplace approx.)
  m = phi @ weights.T
  
  # v is the induced covariance. 
  v = torch.diag(phi @ V @ phi.T).reshape(-1, 1, 1) * U
      
  # The induced distribution over the output (pre-softmax) - this is the approximate posterior
  output_dist = MultivariateNormal(m, v)

  # Approximation of the predictive distribution integral using MC integration
  n_sample = 100_000 # number of MC samples from the posterior
  prob_y = 0

  for _ in range(n_sample):
      out_s = output_dist.rsample()
      prob_y += torch.softmax(out_s, 1)

  prob_y /= n_sample

  return prob_y.argmax(axis=1) , prob_y # predicted labels and probabilities
            
            
            
            
        ###############################
        #      SGLD sampling utils    #
        ###############################
        
        
        
def log_likelihood(network, X, y):
  
    """
    This function computes the log probability `log p(y | x, theta)`
    for a batch of inputs X.
    
    INPUT:
    network : instance of classifier network, extends `nn.Module`
    X       : batch of inputs; torch.FloatTensor, matrix of shape = (batch_size, 2)
    y       : batch of targets: torch.FloatTensor, vector of shape = (batch_size,)
    
    OUTPUT:
    lp      : log probability value of log p(y | x, theta); scalar
    
    """
    
    pred_probs = torch.softmax(network(X), axis=1) # outputs of the network
    lp = dist.Categorical(probs = pred_probs).log_prob(y).sum() # Calculate the log_likelihood sum over all datapoints
    
    return lp


def log_prior(network):
    
    """
    Multivariate standard Normal prior for the last layer weights.
    """
    
    last_layer_params = list(network.parameters())[-2]
    last_layer_params_vector = nn.utils.parameters_to_vector(last_layer_params).to(device)
    n_params = len(last_layer_params_vector)
    # Create the standard Normal multivariate prior
    prior = dist.MultivariateNormal(loc=torch.zeros(n_params).to(device), covariance_matrix=torch.eye(n_params).to(device))

    # Evaluate at the network parameters
    return prior.log_prob(last_layer_params_vector)


def log_joint(network, X_batch, y_batch):

    """ Return an estimate of the full log joint probability. 
    
    INPUT:
    network    : instance of classifier network, extends `nn.Module`
    X_batch    : batch of inputs; torch.FloatTensor, matrix of shape = (batch_size, 2)
    y_batch    : batch of targets: torch.FloatTensor, vector of shape = (batch_size,)
    N_training : total number of training data instances in the full training set

    OUTPUT:
    lp : return an estimate of log p(y, theta | X), as computed on the batch; scalar.

    """
    
    mini_batch_size = X_batch.shape[0]
    N_training = mini_batch_size
    mini_batch_log_lik = log_likelihood(network, X_batch, y_batch) * (N_training / mini_batch_size)
    lp = log_prior(network) + mini_batch_log_lik
    
    return lp
        
        


def SGLD_step(network, X, y, epsilon):

    """
    Run one step of SGLD given a mini-batch, and update the parameters of the network.
    
    INPUT:
    network    : instance of classifier network, extends `nn.Module`
    X_batch    : batch of inputs; torch.FloatTensor, matrix of shape = (batch_size, 2)
    y_batch    : batch of targets: torch.FloatTensor, vector of shape = (batch_size,)
    N_training : total number of training data instances in the full training set
    epsilon    : step size / learning rate parameters (scalar)
    """
    
    network.zero_grad() # zero the gradients
    theta = nn.utils.parameters_to_vector(list(network.parameters())[-2]).to(device)
        
    n_params = len(theta)
    log_joint_ = log_joint(network, X, y)
    log_joint_.backward()
    
    grad_theta = nn.utils.parameters_to_vector(list(network.parameters())[-2].grad).to(device) # get the gradients of the last layer weights
    
    z = dist.Normal(loc=0, scale=1).sample(sample_shape=(n_params,)).to(device) # sample the noise vector z
    
    # Update the networks params
    theta_new = theta + ( (epsilon**2 / 2) * grad_theta ) + (epsilon * z) 

    for layer_num , param in enumerate(network.parameters()):
      if layer_num == 12: # access last layer weights 
        param.data = nn.parameter.Parameter(theta_new.view(3, -1))
    
    return 



def learning_rate_schedule(N_steps, N_samples, epsilon):

  """
  Pre-compute a learning-rate schedule for SGLD.
  
  INPUT:
  N_steps   : number of SGD updates between each "reset"
  N_samples : number of times we reach the lowest target learning rate
  epsilon   : base learning rate
  
  OUTPUT:
  epsilon_t : vector of length N_steps*N_samples, containing epsilon_t at each iteration t
  """
  
  return epsilon * (np.cos(np.pi * (np.arange(N_samples*N_steps) % N_steps)/N_steps) + 1)


def draw_sgld_samples(network, X, y, N_samples, N_steps_per_sample, base_epsilon=0.02):

    """
    Draw samples using SGLD, following a prescribed learning rate schedule
    
    OUTPUT:
    samples : torch.FloatTensor, shape = (N_samples, "# of parameters in network")
    """

    lr_schedule = learning_rate_schedule(N_steps_per_sample, N_samples, base_epsilon)
    samples = []
    step = 0

    while True:

        SGLD_step(network, X, y, epsilon=lr_schedule[step])
        step += 1

        if step % N_steps_per_sample == 0:
            samples.append(nn.utils.parameters_to_vector(list(network.parameters())[-2]).detach())

        if step == len(lr_schedule):
            return torch.stack(samples).to(device)


def predict_from_posterior_samples(X, network, samples):

    """
    INPUT:
    X       : batch of input points at which to make predictions; shape = (batch_size, 2)
    network : instance of classifier network, extends `nn.Module`
    samples : torch.FloatTensor containing samples of theta; shape = (num_samples, "# of parameters in network")
    
    OUTPUT:
    y_hat_samples : torch.FloatTensor containing samples of y_hat; shape = (num_samples, batch_size)
    """
    
    batch_size = X.shape[0]
    thetas = [samples[i,:].to(device) for i in range(len(samples))]
    y_hat_samples = torch.zeros((batch_size, 3, len(samples))).to(device)
    
    for i , theta in enumerate(thetas):
        # Run the model forward manually for each MCMC posterior sample
        nn.utils.vector_to_parameters(theta, list(network.parameters())[-2]) # put the sampled parameters in the network's last layer
        y_hat_samples[: , : , i] =  torch.softmax(network.forward(X), axis=1).to(device) # generate the predictions
        
        
    return y_hat_samples.mean(axis=-1).to(device)
        
        

import torch
from torch.optim import Adam
from torch.special import expit, logit
import torch.nn.functional as F

# Define constants and parameters
m = 100  # number of samples for q
n = 100  # number of data points
eta = 0.1
gamma = 2.0
alpha = 0.5

# Get data
x_data = torch.randn(n)
y_data = torch.bernoulli(torch.full((n,), 0.5))
theta = torch.randn(m, requires_grad=True)


def get_pi(mu, sigma, x):
    """
    Get the probabilities allowing the gradients to flow.
    Assumes Sigma = sigma * I
    """
    p = x.size(1)  # Assuming x is of shape (n, p, 1)
    z = torch.randn(p)  # Generate random tensor for each element in x
    return torch.sigmoid(z / sigma + (mu @ x).squeeze(-1))  

# Define the focus loss function
def focus_loss(mu, sigma, x, y, gamma=1.0):
    """
    Return the focus loss for multiple observations 
    """
    p_i = get_pi(mu, sigma, x)  # Get probabilities for all x and (mu, sigma) tuples
    return - (y * (1 - p_i) ** gamma * torch.log(p_i) + (1 - y) * p_i ** gamma * torch.log(1 - p_i))


def logit_normal_pdf(x, mu, Sigma):
    """
    Returns the pdf of a LogisticNormal(mu,sigma) evaluated at z.
    """

    # Compute the logit of x
    logit_x = logit(x)

    # Compute the multivariate normal pdf in the logit space
    dim = x.shape[-1]
    det_Sigma = torch.det(Sigma)
    inv_Sigma = torch.inverse(Sigma)
    diff = logit_x - mu
    exponent = -0.5 * torch.sum(diff @ inv_Sigma * diff, dim=-1)
    
    normalization_constant = 1 / ((2 * torch.pi) ** (dim / 2) * torch.sqrt(det_Sigma))
    jacobian = torch.prod(x * (1 - x), dim=-1)

    pdf = normalization_constant * (1 / jacobian) * torch.exp(exponent)
    return pdf



# Define the alpha-divergence
def alpha_divergence(q, p):
    """
    Return the alpha divergence for one observation.
    Note: this needs to be fully computed per iteration, ie it doesn't participate in the mini batch
    Params:
    - q: tuple of (mu, Sigma) for the variational proposal
    - p: pdf (function) that outputs the prior probability for each x
    """
    q_pdf = lambda x: logit_normal_pdf(x, q[0], q[1])
    x = torch.random.rand(100)
    int_est = torch.mean(q_pdf(x) ** alpha * p(x) ** (1 - alpha))
    return (1 / (alpha * (alpha - 1))) * torch.log(int_est)

# Define the objective function
def objective(q, x, y, eta):
    """
    Compute the value of the objective function at each iteration


    Parameters:
    -  q: tuple with (mu, Sigma)
    """
    # Compute the loss L(q)
    n = len(y_data)
    # simulate from q
    if x.ndimension() <= 1 or x.shape[1] <= 1:
        raise ValueError("The tensor x does not have more than one column.")

    thetas = [get_pi(q[0], q[1], x[i,:]) for i in range(n)]

    # Expand dimensions to use broadcasting
    x_expanded = x.unsqueeze(2)  # Shape: (n, p, 1)
    y_expanded = y.unsqueeze(1)  # Shape: (n, 1)
    mu_expanded = mu.unsqueeze(0)  # Shape: (1, p, m)
    sigma_expanded = sigma.unsqueeze(0)  # Shape: (1, p, m)
    
    L_q = focus_loss(mu_expanded, sigma_expanded, x_expanded, y_expanded)  # Shape: (n, m)
    # Calculate the mean over the different thetas
    L_q = -eta * torch.mean(L_1, dim = 1) # Shape: (n,)


    # Compute the divergence D_alpha
    p_q = torch.mean(logistic(torch.matmul(theta, q)), dim=0)
    p_t = torch.mean(logistic(torch.matmul(theta, q)), dim=0)
    D_alpha = alpha_divergence(p_t, p_q)

    # Combine the loss and divergence
    return L_q + D_alpha

# Initial guess for q
q_init = torch.randn(m, requires_grad=True)

# Define the optimizer
optimizer = Adam([q_init], lr=0.01)

# Training loop
for step in range(1000):
    optimizer.zero_grad()
    loss = objective(q_init, x_data, y_data, theta, eta)
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f'Step {step}, Loss: {loss.item()}')

# Final optimized q
q_opt = q_init.detach()
print('Optimized q:', q_opt)
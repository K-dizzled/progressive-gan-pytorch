import torch # noqa

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
latent_dim_size = 128
total_steps = 6
epochs = [50000 for _ in range(total_steps)]
batch_size = [4 for _ in range(total_steps)]
initial_step = 1
n_critic = 1
data_directory = "hearts/1/"
use_tangent_hyperbolic_function = False
use_pixel_normalization = False
use_minibatch_stddev = False
sample_freq = 1000
save_freq = 1000

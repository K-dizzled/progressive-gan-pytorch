import torch  # noqa
from torch import optim  # noqa
from torch.autograd import grad  # noqa
from torch import nn  # noqa
from datetime import datetime
from alive_progress import alive_bar  # noqa
import numpy as np  # noqa

from discriminator_model import Discriminator
from generator_model import Generator
from image_pool import ImagePool
from image_processing import *
from config import *
import json


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


class ProGAN(object):
    def __init__(self,
                 init_step=initial_step,
                 epochs_per_step=epochs,
                 b_size=batch_size,
                 t_steps=6):

        self.max_available_steps = 6

        # Initialize models
        self.generator = Generator(
            in_channel=128,
            input_code_dim=latent_dim_size,
            pixel_norm=use_pixel_normalization,
            tanh=use_tangent_hyperbolic_function
        ).to(device)

        self.discriminator = Discriminator(
            feat_dim=128,
            allow_std_dev=use_minibatch_stddev
        ).to(device)

        self.g_running = Generator(
            in_channel=128,
            input_code_dim=latent_dim_size,
            pixel_norm=use_pixel_normalization,
            tanh=use_tangent_hyperbolic_function
        ).to(device)

        self.g_running.train(False)

        # Optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=0.001,
            betas=(0.0, 0.99)
        )

        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=0.001,
            betas=(0.0, 0.99)
        )

        accumulate(self.g_running, self.generator, 0)

        # initialize fields
        self.step = init_step
        self.epochs = epochs_per_step
        self.batch_size = b_size
        self.total_steps = min(self.max_available_steps, t_steps)
        self.disc_loss_val, self.gen_loss_val, self.grad_loss_val = 0, 0, 0
        self.fixed_point_for_gif = None

        # Variable that controls how much to weight the first
        # and second inputs in the Weighted sum
        self.alpha = None

        # Load the dataset
        self.image_pool = ImagePool(data_directory,
                                    resolution=4 * (2 ** self.step),
                                    batch_size=self.batch_size[self.step - 1])

        # TODO: Add custom loss function
        # TODO: Save net parameters to log folder for model recreation

        self.num_samples_to_generate = 7
        self.log_folder = 'run_'

    def loadModel(self, path_to_generator_model, path_to_discriminator_model):
        self.generator.load_state_dict(torch.load(path_to_generator_model))
        self.discriminator.load_state_dict(torch.load(path_to_discriminator_model))

        self.generator.eval()
        self.discriminator.eval()

        with open(self.log_folder + '/models/params.json', 'r') as json_file:
            data = json.load(json_file)
            self.step = data['step']
            self.alpha = data['alpha']

    def prepare_before_training(self):
        """
        Create log folders
        """
        date_time = datetime.now()
        self.log_folder += date_time.strftime("%Y-%m-%d_%H-%M")

        os.mkdir(self.log_folder)
        os.mkdir(self.log_folder + '/models')
        os.mkdir(self.log_folder + '/sample')
        os.mkdir(self.log_folder + '/gif')

    def generate_latent_points(self):
        return torch.randn(
            self.num_samples_to_generate * self.num_samples_to_generate,
            latent_dim_size
        ).to(device)

    def checkpoint(self, iteration, latent_points, subdirectory):
        with torch.no_grad():
            images = self.g_running(
                latent_points,
                step=self.step,
                alpha=self.alpha
            ).data.cpu()

            save_samples(self.num_samples_to_generate, images, iteration, self.log_folder + subdirectory)

    def save_models(self, iteration):
        torch.save(
            self.g_running.state_dict(),
            '%s/models/%s_g.model' % (self.log_folder, str(iteration + 1).zfill(6))
        )
        torch.save(
            self.discriminator.state_dict(),
            '%s/models/%s_d.model' % (self.log_folder, str(iteration + 1).zfill(6))
        )
        with open(self.log_folder + '/models/params.json', 'w') as out_file:
            params = {
                'alpha': self.alpha,
                'step': self.step
            }

            json_string = json.dumps(params)
            out_file.write(json_string)

    def train(self):
        self.disc_loss_val, self.gen_loss_val, self.grad_loss_val = 0, 0, 0
        self.prepare_before_training()

        # Fix random points in latent space which will be used to
        # measure progress of generating images on these particular points
        self.fixed_point_for_gif = torch.randn(
            self.num_samples_to_generate * self.num_samples_to_generate,
            latent_dim_size
        ).to(device)

        one = torch.tensor(1, dtype=torch.float).to(device)
        oneT = one * -1
        iteration = 0
        global_iteration = 0

        for resolution in range(self.total_steps):
            bar_title = 'Resolution: %dx%d' % (4 * (2 ** (resolution + 1)), 4 * (2 ** (resolution + 1)))
            # Show progress bar for current resolution

            with alive_bar(self.epochs[resolution], dual_line=True, title=bar_title) as bar:
                for _ in range(self.epochs[resolution]):
                    self.discriminator.zero_grad()

                    self.alpha = min(1, (2 / self.epochs[resolution]) * iteration)

                    real_images = self.image_pool.next()

                    iteration += 1

                    # From NCWH to NCHW
                    real_images = np.transpose(real_images, [0, 3, 1, 2])

                    # Convert to pytorch tensors
                    real_images = torch.from_numpy(real_images).to(device)

                    # Train Discriminator
                    b_size = real_images.shape[0]
                    real_predict = self.discriminator(
                        real_images, step=self.step, alpha=self.alpha)
                    real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
                    real_predict.backward(oneT)

                    # Sample point from latent space
                    gen_z = torch.randn(b_size, latent_dim_size).to(device)

                    fake_image = self.generator(
                        gen_z,
                        step=self.step,
                        alpha=self.alpha
                    )
                    fake_predict = self.discriminator(
                        fake_image.detach(),
                        step=self.step,
                        alpha=self.alpha
                    )
                    fake_predict = fake_predict.mean()
                    fake_predict.backward(one)

                    # Gradient penalty for Discriminator
                    eps = torch.rand(b_size, 1, 1, 1).to(device)
                    x_hat = eps * real_images.data + (1 - eps) * fake_image.detach().data
                    x_hat.requires_grad = True
                    hat_predict = self.discriminator(
                        x_hat,
                        step=self.step,
                        alpha=self.alpha
                    )

                    grad_x_hat = grad(
                        outputs=hat_predict.sum(),
                        inputs=x_hat,
                        create_graph=True
                    )[0]

                    grad_penalty = grad_x_hat.reshape(grad_x_hat.size(0), -1)
                    grad_penalty = grad_penalty.norm(2, dim=1) - 1
                    grad_penalty = grad_penalty ** 2
                    grad_penalty = grad_penalty.mean()
                    grad_penalty *= 10
                    grad_penalty.backward()

                    self.grad_loss_val += grad_penalty.item()
                    self.disc_loss_val += (real_predict - fake_predict).item()

                    self.d_optimizer.step()

                    # Train Generator
                    if (global_iteration + 1) % n_critic == 0:
                        self.generator.zero_grad()
                        self.discriminator.zero_grad()

                        predict = self.discriminator(
                            fake_image,
                            step=self.step,
                            alpha=self.alpha
                        )

                        loss = -predict.mean()
                        self.gen_loss_val += loss.item()

                        loss.backward()
                        self.g_optimizer.step()
                        accumulate(self.g_running, self.generator)

                    if (global_iteration + 1) % (sample_freq // 3) == 0:
                        self.checkpoint(global_iteration, self.fixed_point_for_gif, '/gif')

                    if (global_iteration + 1) % sample_freq == 0:
                        self.checkpoint(global_iteration, self.generate_latent_points(), '/sample')

                    if (global_iteration + 1) % save_freq == 0:
                        self.save_models(global_iteration)

                    # Update progress bar
                    bar.text = '-> d_loss: %.3f, g_loss: %.3f, grad_loss: %.3f, alpha: %.3f' \
                               % (self.disc_loss_val,
                                  self.gen_loss_val,
                                  self.grad_loss_val,
                                  self.alpha)
                    bar()

                    self.disc_loss_val, self.gen_loss_val, self.grad_loss_val = 0, 0, 0
                    global_iteration += 1

            # Update image pool for next learning cycle.
            iteration = 0
            self.step += 1

            step = min(self.step, self.max_available_steps)

            print("Required resolution on step: %d" % (4 * (2 ** step)))
            self.image_pool = ImagePool(data_directory,
                                        resolution=4 * (2 ** step),
                                        batch_size=self.batch_size[self.step - 1])

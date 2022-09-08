import os
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
from utils.data_utils import tensor2im
import torchvision.utils as utils


class MultiIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb, image_name=None, label_path=None, projector_type='ide3d', initial_w=None, total_steps=1000):
        super().__init__(data_loader, use_wandb, image_name=image_name, label_path=label_path, projector_type=projector_type, initial_w=initial_w)
        self.total_steps = total_steps

    def train(self):
        self.G.synthesis.train()
        self.G.mapping.train()

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        use_ball_holder = True
        w_pivots = []
        labels = []
        images = []

        for fname, image in tqdm(self.data_loader):
            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            image_name = fname[0]
            if hyperparameters.first_inv_type == 'w+':
                embedding_dir = f'{w_path_dir}/{paths_config.e4e_results_keyword}/{image_name}'
            else:
                embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)
            w_pivot, label = self.get_inversion(w_path_dir, image_name, image)
            w_pivots.append(w_pivot)
            labels.append(label)
            images.append((image_name, image))
            self.image_counter += 1

        for i in tqdm(range(hyperparameters.max_pti_steps)):
            self.image_counter = 0

            for data, w_pivot, label in zip(images, w_pivots, labels):
                image_name, image = data

                if self.image_counter >= hyperparameters.max_images_to_invert:
                    break

                real_images_batch = image.to(global_config.device)

                generated_images = self.forward(w_pivot, label)
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
                                    self.G, use_ball_holder, w_pivot)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                global_config.training_step += 1
                self.image_counter += 1

                if hyperparameters.save_interval is not None and global_config.training_step % hyperparameters.save_interval == 0:
                    os.makedirs(f'{paths_config.experiments_output_dir}/images', exist_ok=True)
                    for idx in range(generated_images.shape[0]):
                        new_image_path = f'{paths_config.experiments_output_dir}/images/step_{global_config.training_step}.png'
                        out_image = tensor2im(generated_images[idx]).resize((512, 512))
                        target_image = tensor2im(image[idx]).resize((512, 512))
                        coupled_image = np.concatenate([np.array(target_image), np.array(out_image)], axis=1)
                        Image.fromarray(coupled_image).save(new_image_path)

                if hyperparameters.model_save_interval is not None and global_config.training_step > 0 and global_config.training_step % hyperparameters.model_save_interval == 0:
                    torch.save(self.G,
                   f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_multi_id_step_{global_config.training_step}.pt')


        if self.use_wandb:
            log_images_from_w(w_pivots, self.G, [image[0] for image in images])

        torch.save(self.G,
                   f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_multi_id.pt')

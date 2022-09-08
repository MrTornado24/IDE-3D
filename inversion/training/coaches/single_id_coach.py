import os
import torch
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
import torchvision.utils as utils



class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb, image_name=None, label_path=None, projector_type='ide3d', initial_w=None):
        super().__init__(data_loader, use_wandb, image_name=image_name, label_path=label_path, projector_type=projector_type, initial_w=initial_w)

    def train(self):

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        use_ball_holder = True

        for fname, image in tqdm(self.data_loader):

            ### prepare for flip
            image_flip = torch.flip(image, (3,))
            image_name = fname[0]
            if self.image_name is not None:
                if image_name != self.image_name:
                    continue
            self.restart_training()

            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            w_pivot = None

            if hyperparameters.use_last_w_pivots:
                w_pivot = self.load_inversions(w_path_dir, image_name)

            elif not hyperparameters.use_last_w_pivots or w_pivot is None:
                w_pivot, label = self.calc_inversions(image, image_name)
                if label is not None:
                    label_flip = label.clone()
                    label_flip[:, [1,2,3,4,8]] *= -1

            # w_pivot = w_pivot.detach().clone().to(global_config.device)
            w_pivot = w_pivot.to(global_config.device)

            torch.save(w_pivot, f'{embedding_dir}/0.pt')
            torch.save(label, f'{embedding_dir}/0_label.pt')
            log_images_counter = 0
            real_images_batch = image.to(global_config.device)
            real_images_flip_batch = image_flip.to(global_config.device)

            for i in tqdm(range(hyperparameters.max_pti_steps)):

                generated_images = self.forward(w_pivot, label) if self.projector_type.startswith('ide3d') else self.forward(w_pivot, label)
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
                                                               self.G, use_ball_holder, w_pivot)
                
                if self.projector_type == 'ide3d_join_view':
                    generated_images_flip = self.forward(w_pivot, label_flip)
                    loss_flip, _, loss_lpips_flip = self.calc_loss(generated_images_flip, real_images_flip_batch, image_name,
                                                               self.G, use_ball_holder, w_pivot)
                    loss = loss + loss_flip
                    loss_lpips = loss_lpips + loss_lpips_flip

                self.optimizer.zero_grad()

                if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    break

                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                # if log_images_counter % global_config.image_rec_result_log_snapshot == 0:
                #     # log_images_from_w([w_pivot], self.G, [image_name])
                #     with torch.no_grad():
                #         utils.save_image(generated_images, f'tmp/{image_name}/opt_z_{log_images_counter:04d}.png', normalize=True, range=(-1,1))

                global_config.training_step += 1
                log_images_counter += 1

            self.image_counter += 1

            torch.save(self.G,
                       f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{image_name}.pt')
from fileinput import filename
import sys, argparse
from cv2 import normalize
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
import torch
import qdarkstyle
import numpy as np
import datetime
import pickle,time
from PIL import Image
import os
import math
import time
import json
import pickle
from torchvision import transforms, utils

sys.path.append('D:/projects/IDE-3D')
import dnnlib
import legacy
from inversion.BiSeNet import BiSeNet
from ui.ui import Ui_Form
from ui.mouse_event import GraphicsScene
from ui.util import number_color, number_object
from training.volumetric_rendering import sample_camera_positions, create_cam2world_matrix


def init_deep_model(g_ckpt, e_ckpt):
    if g_ckpt.endswith('pt'):
            with open(g_ckpt, 'rb') as f:
                G = torch.load(f).cuda().eval()
    else:
        with dnnlib.util.open_url(g_ckpt) as fp:
            network = legacy.load_network_pkl(fp)
            G = network['G_ema'].requires_grad_(False).cuda()

    with dnnlib.util.open_url(e_ckpt) as fp:
            network = legacy.load_encoder_pkl(fp)
            E = network['E'].requires_grad_(True).cuda()

    torch.cuda.empty_cache()

    return G.cuda().eval(), E.cuda().eval()

class ExWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.EX = Ex(args)


class Ex(QWidget, Ui_Form):
    def __init__(self, args):

        super().__init__()

        self.args = args
        self.current_style = 0
        self.generator, self.encoder = init_deep_model(args.g_ckpt, args.e_ckpt)
        self.ws_avg = self.generator.mapping.w_avg[None, None, :]
        self.noise = None
        self.setupUi(self)
        self.show()
        self.w = torch.load(args.target_code) if args.target_code is not None else None
        self.modes = 0
        self.alpha = 0.5
        self.yaw = math.pi * 0.5
        self.pitch = math.pi * 0.5
        self.yaw_orig = math.pi * 0.5
        self.pitch_orig = math.pi * 0.5
        self.fov = 18
        self.traj_type = 'orbit'
        self.trajectory = self.set_trajectory()
        with open('D:/projects/IDE-3D/data/ffhq/images512x512/dataset.json', 'rb') as f:
            labels = json.load(f)['labels']
        self.labels = dict(labels)
        self.label = torch.load(args.target_label) if args.target_label is not None else None
        self.fake_img = None
        self.synthetic_label = None
        self.inversion = args.inversion
        self.color_map = {
                0: [0, 0, 0],
                1: [204, 0, 0], # #CC0000
                2: [76, 153, 0], 
                3: [204, 204, 0], 
                4: [51, 51, 255], 
                5: [204, 0, 204], 
                6: [0, 255, 255], 
                7: [255, 204, 204], 
                8: [102, 51, 0], 
                9: [255, 0, 0], 
                10: [102, 204, 0], 
                11: [255, 255, 0], 
                12: [0, 0, 153], 
                13: [0, 0, 204], 
                14: [255, 51, 153], 
                15: [0, 204, 204], 
                16: [0, 51, 0], 
                17: [255, 153, 51], 
                18: [0, 204, 0]}

        self.bisNet, to_tensor = self.initFaceParsing(path='pretrained_models')

        self.mouse_clicked = False
        self.scene = GraphicsScene(self.modes, self)
        self.scene.setSceneRect(0, 0, 512, 512)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setAlignment(Qt.AlignCenter)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.GT_scene = QGraphicsScene()
        self.graphicsView_GT.setScene(self.GT_scene)
        self.graphicsView_GT.setAlignment(Qt.AlignCenter)
        self.graphicsView_GT.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_GT.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.dlg = QColorDialog(self.graphicsView)

        self.init_screen()

    def init_screen(self):
        self.image = QPixmap(QSize(512, 512))
        self.image.fill(QColor('#000000'))
        self.mat_img = np.zeros([512, 512], np.uint8)

        self.mat_img_org = self.mat_img.copy()

        self.GT_img_path = None
        GT_img = self.mat_img.copy()
        self.GT_img = Image.fromarray(GT_img)
        self.GT_img = self.GT_img.convert('RGB')

        self.last = time.time()

        self.scene.reset()
        if len(self.scene.items()) > 0:
            self.scene.reset_items()

        self.scene_image_pts = self.scene.addPixmap(self.image)
        self.GT_scene_image_pts = self.GT_scene.addPixmap(self.image)

        self.image = np.zeros([512, 512, 3], np.uint8)
        self.image_raw = self.image.copy()
        self.update_segmap_vis(self.mat_img)

        ###############
        self.recorded_img_names = []

        self.frameLog = {}
        self.starTime = datetime.datetime.now().strftime('%H_%M_%S_%f')

    def initFaceParsing(self, n_classes=20, path=None):
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        net.load_state_dict(torch.load(path+'/segNet-20Class.pth'))
        net.eval()
        to_tensor = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

        ])
        return net, to_tensor

    def run_deep_model(self, label=None):
        ""
        with torch.no_grad():
            seg_label = self.mask_labels(self.id_remap(self.mat_img))
            print("mat image size:", self.mat_img.shape)
            print("seg label size:", seg_label.shape)
            seg_label = torch.from_numpy(seg_label).view(1,19,512,512).float().cuda()
            seg_label = seg_label * 2 - 1
            if self.inversion:
                w = self.w
            else:
                self.set_random_seed(self.current_style)
                z = torch.from_numpy(np.random.RandomState(self.current_style).randn(1, self.generator.z_dim).astype('float32')).cuda()
                c_cond = torch.tensor([1,0,0,0, 0,1,0,0, 0,0,1,2.7, 0,0,0,1, 4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).float().cuda().reshape(1,-1)
                w = self.generator.mapping(z, c_cond)
            ### camera pose 
            if label is None or label is False:
                camera_points, phi, theta = sample_camera_positions('cuda', n=1, r=2.7, horizontal_mean=self.yaw, vertical_mean=self.pitch, mode=None)
                c = create_cam2world_matrix(-camera_points, camera_points, device='cuda')
                c = c.reshape(1,-1)
                c = torch.cat((c, torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).reshape(1, -1).to(c)), -1)
            else:
                c = label
            render_params = {
                'num_steps': 96
            }
            gen_img = self.generator.synthesis(w, c=c, render_params=render_params, noise_mode='const')
            # rec_ws = self.encoder(gen_img, seg_label, num_view=1)
            rec_ws = self.encoder(gen_img, seg_label)
            rec_ws = self.ws_avg + rec_ws 
            if self.w is not None and self.inversion:
                rec_ws[:, 8:] = self.w[:, 8:]
            fake_img = self.generator.synthesis(ws=rec_ws, c=c, noise_mode='const')      
            self.fake_img = fake_img                             
            fake_img = ((fake_img[0].permute(1,2,0).cpu()+ 1) / 2 * 255).clamp_(0,255).numpy().astype('uint8')
            fake_img = cv2.resize(fake_img, (512, 512))
        self.w = rec_ws
        self.GT_scene_image_pts.setPixmap(QPixmap.fromImage(QImage(fake_img.data.tobytes(), \
                                                            fake_img.shape[1], fake_img.shape[0],
                                                            QImage.Format_RGB888)))

    def run_deep_model_fake(self, label=None, yaw=None, pitch=None):
        """only for free-view rendering"""
        with torch.no_grad():
            seg_label = self.mask_labels(self.id_remap(self.mat_img))
            print("mat image size:", self.mat_img.shape)
            print("seg label size:", seg_label.shape)
            seg_label = torch.from_numpy(seg_label).view(1,19,512,512).float().cuda()
            seg_label = seg_label * 2 - 1
            self.set_random_seed(self.current_style)
            z = torch.from_numpy(np.random.RandomState(self.current_style).randn(1, self.generator.z_dim).astype('float32')).cuda()
            c_cond = torch.tensor([1,0,0,0, 0,1,0,0, 0,0,1,2.7, 0,0,0,1, 4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).float().cuda().reshape(1,-1)
            w = self.generator.mapping(z, c_cond)
            ### camera pose 
            if label is None or label is False:
                camera_points, phi, theta = sample_camera_positions('cuda', n=1, r=2.7, horizontal_mean=yaw, vertical_mean=pitch, mode=None)
                c = create_cam2world_matrix(-camera_points, camera_points, device='cuda')
                c = c.reshape(1,-1)
                c = torch.cat((c, torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).reshape(1, -1).to(c)), -1)
            else:
                c = label
            render_params = {
                'num_steps': 96
            }
            gen_img = self.generator.synthesis(w, c=c, render_params=render_params, noise_mode='const')
            # rec_ws = self.encoder(gen_img, seg_label, num_view=1)
            rec_ws = self.encoder(gen_img, seg_label)
            rec_ws = self.ws_avg + rec_ws 
            self.w = rec_ws
            fake_img = self.generator.synthesis(ws=rec_ws, c=c, noise_mode='const')      
            self.fake_img = fake_img                             
            fake_img = ((fake_img[0].permute(1,2,0).cpu()+ 1) / 2 * 255).clamp_(0,255).numpy().astype('uint8')
            fake_img = cv2.resize(fake_img, (512, 512))
        self.GT_scene_image_pts.setPixmap(QPixmap.fromImage(QImage(fake_img.data.tobytes(), \
                                                            fake_img.shape[1], fake_img.shape[0],
                                                            QImage.Format_RGB888)))

    def set_trajectory(self):
        traj = []
        images = []
        if self.traj_type == 'front':
            for frame_idx in range(240):
                h = math.pi*(0.5+0.1*math.cos(2*math.pi*frame_idx/(0.5 * 240)))
                v = math.pi*(0.5-0.05*math.sin(2*math.pi*frame_idx/(0.5 * 240)))
                traj.append((h, v))
        elif self.traj_type == 'orbit':
            for t in np.linspace(0.5, 0.3, 15):
                pitch = math.pi/2
                yaw = t * math.pi
                traj.append((yaw, pitch))
            for t in np.linspace(0.3, 0.5, 15):
                pitch = math.pi/2
                yaw = t * math.pi
                traj.append((yaw, pitch))
            for t in np.linspace(0.5, 0.7, 15):
                pitch = math.pi/2
                yaw = t * math.pi
                traj.append((yaw, pitch))
            for t in np.linspace(0.7, 0.5, 15):
                pitch = math.pi/2
                yaw = t * math.pi
                traj.append((yaw, pitch))
            for t in np.linspace(0.5, 0.4, 15):
                yaw = math.pi/2
                pitch = t * math.pi
                traj.append((yaw, pitch))
            for t in np.linspace(0.4, 0.5, 15):
                yaw = math.pi/2
                pitch = t * math.pi
                traj.append((yaw, pitch))
            for t in np.linspace(0.5, 0.6, 15):
                yaw = math.pi/2
                pitch = t * math.pi
                traj.append((yaw, pitch))
            for t in np.linspace(0.6, 0.5, 15):
                yaw = math.pi/2
                pitch = t * math.pi
                traj.append((yaw, pitch))
            
            
        
        return traj
        
    def mask_labels(self, mask_np):
        label_size = len(self.color_map.keys())
        labels = np.zeros((label_size, mask_np.shape[0], mask_np.shape[1]))
        for i in range(label_size):
            labels[i][mask_np==i] = 1.0
        return labels

    def change_style(self):
        self.current_style += 1
        self.run_deep_model()
    
    def back_style(self):
        self.current_style -= 1
        self.run_deep_model()

    @pyqtSlot()
    def freeview_render(self):
        for (h, v) in self.trajectory:
            self.run_deep_model_fake(yaw=h, pitch=v)
            QApplication.processEvents()

    @pyqtSlot()
    def reset_view(self):
        if self.label is not None:
            label = self.label
        else:
            label = self.get_label(self.mat_img_path)
        self.run_deep_model(label)


    def set_random_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    @pyqtSlot()
    def open(self):

        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", 'F:/Lab/samples')
        if fileName:

            self.mat_img_path = os.path.join(fileName)
            self.fileName = fileName
            print(self.mat_img_path)

            # USE CV2 read images, because of using gray scale images, no matter the RGB orders
            mat_img = cv2.imread(self.mat_img_path, -1)
            if mat_img is None:
                QMessageBox.information(self, "Image Viewer",
                                        "Cannot load %s." % fileName)
                return

            if mat_img.ndim == 2:
                self.mat_img = cv2.resize(mat_img, (512, 512), interpolation=cv2.INTER_NEAREST)
                print("mat image shape", self.mat_img.shape)
                self.image = self.segmap2rgb(self.id_remap(self.mat_img))
                self.mat_img_org = self.mat_img.copy()
            else:
                self.image = cv2.resize(mat_img[...,::-1], (512, 512))

            self.image_raw = self.image.copy()
            self.image = np.round(self.alpha*self.image).astype('uint8')
            image = self.image + (self.segmap2rgb(self.id_remap(self.mat_img)) * int(1000 * (1.0 - self.alpha)) // 1000).astype('uint8')
            image = QPixmap.fromImage(QImage(image.data.tobytes(), self.image.shape[1], self.image.shape[0], QImage.Format_RGB888))


            self.scene.reset()
            if len(self.scene.items()) > 0:
                self.scene.reset_items()

            self.scene_image_pts = self.scene.addPixmap(image)

            if mat_img.ndim == 2: # template
                self.update_segmap_vis(self.mat_img)

    @pyqtSlot()
    def open_random(self):

        style = np.random.randint(1000000)
        z = torch.from_numpy(np.random.RandomState(style).randn(1, self.generator.z_dim).astype('float32')).cuda()
        c_cond = torch.tensor([1,0,0,0, 0,1,0,0, 0,0,1,2.7, 0,0,0,1, 4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).float().cuda().reshape(1,-1)
        w = self.generator.mapping(z, c_cond)
        ### camera pose 
        camera_points, phi, theta = sample_camera_positions('cuda', n=1, r=2.7, horizontal_mean=0.5*math.pi, 
                                                        vertical_mean=0.5*math.pi, horizontal_stddev=0.3, vertical_stddev=0.155, mode='gaussian')
        self.yaw = theta
        self.pitch = phi
        self.yaw_orig = theta
        self.pitch_orig = phi
        c = create_cam2world_matrix(-camera_points, camera_points, device='cuda')
        c = c.reshape(1,-1)
        c = torch.cat((c, torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).reshape(1, -1).to(c)), -1)
        # self.synthetic_label = c
        gen_img = self.generator.synthesis(w, c=c, noise_mode='const')
        _, rec_seg = self.parsing_img(self.bisNet, gen_img, remap=True, return_mask=False)
        mat_img = rec_seg[0, 0].detach().cpu().numpy().astype(np.uint8)
        fileName = f"synthetic_seed_{style}.png"

        self.mat_img_path = os.path.join(fileName)
        self.fileName = fileName
        print(self.mat_img_path)

        # USE CV2 read images, because of using gray scale images, no matter the RGB orders
        if mat_img.ndim == 2:
            self.mat_img = cv2.resize(mat_img, (512, 512), interpolation=cv2.INTER_NEAREST)
            print("mat image shape", self.mat_img.shape)
            self.image = self.segmap2rgb(self.id_remap(self.mat_img))
            self.mat_img_org = self.mat_img.copy()
        else:
            self.image = cv2.resize(mat_img[...,::-1], (512, 512))

        self.image_raw = self.image.copy()
        self.image = np.round(self.alpha*self.image).astype('uint8')
        image = self.image + (self.segmap2rgb(self.id_remap(self.mat_img)) * int(1000 * (1.0 - self.alpha)) // 1000).astype('uint8')
        image = QPixmap.fromImage(QImage(image.data.tobytes(), self.image.shape[1], self.image.shape[0], QImage.Format_RGB888))


        self.scene.reset()
        if len(self.scene.items()) > 0:
            self.scene.reset_items()
        self.scene_image_pts = self.scene.addPixmap(image)

        if mat_img.ndim == 2: # template
            self.update_segmap_vis(self.mat_img)

    @pyqtSlot()
    def open_reference(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath()+'/samples')
        if fileName:

            self.mat_img_path = os.path.join(fileName)
            self.fileName = fileName

            mat_img = cv2.imread(self.mat_img_path, 1)

            self.image_raw = cv2.resize(mat_img[...,::-1], (512, 512))
            self.change_alpha_value()

    def update_segmap_vis(self, segmap):
        ""

        # if not self.args.load_network:
        #     self.GT_scene_image_pts.setPixmap(QPixmap.fromImage(QImage((10 * segmap).data.tobytes(), \
        #                                                      segmap.shape[1], segmap.shape[0],
        #                                                      QImage.Format_Grayscale8)))

        out = self.image + (self.segmap2rgb(self.id_remap(self.mat_img))*int(1000*(1.0-self.alpha))//1000).astype('uint8')
        self.scene_image_pts.setPixmap(QPixmap.fromImage(QImage(out.data.tobytes(), \
                                                         out.shape[1], out.shape[0],
                                                         QImage.Format_RGB888)))

        print('FPS: %s'%(1.0/(time.time()-self.last)))
        self.last = time.time()


    @pyqtSlot()
    def change_brush_size(self):
        self.scene.brush_size = self.brushSlider.value()
        self.brushsizeLabel.setText('Brush size: %d' % self.scene.brush_size)


    @pyqtSlot()
    def change_alpha_value(self):
        self.alpha = self.alphaSlider.value() / 20
        self.alphaLabel.setText('Alpha: %.2f' % self.alpha)

        self.image = np.round(self.image_raw*self.alpha).astype('uint8')
        out = self.image + (self.segmap2rgb(self.id_remap(self.mat_img))*int(1000*(1.0-self.alpha))//1000).astype('uint8')

        self.scene_image_pts.setPixmap(QPixmap.fromImage(QImage(out.data.tobytes(), \
                                                         out.shape[1], out.shape[0],
                                                         QImage.Format_RGB888)))
    @pyqtSlot()
    def change_yaw_value(self):
        self.yaw = self.yawSlider.value() / 50 + math.pi *0.5
        self.yawLabel.setText('Yaw: %.2f' % self.yaw)
        self.run_deep_model()
    
    @pyqtSlot()
    def change_pitch_value(self):
        self.pitch = self.pitchSlider.value() / 50 + math.pi *0.5
        self.pitchLabel.setText('Pitch: %.2f' % self.pitch)
        self.run_deep_model()

    @pyqtSlot()
    def mode_select(self, mode):
        self.modes = mode
        self.scene.modes = mode

        if mode == 0:
            self.brushButton.setStyleSheet("background-color: #85adad")
            self.recButton.setStyleSheet("background-color:")
            self.fillButton.setStyleSheet("background-color:")
            QApplication.setOverrideCursor(Qt.ArrowCursor)
        elif mode == 1:
            self.recButton.setStyleSheet("background-color: #85adad")
            self.brushButton.setStyleSheet("background-color:")
            self.fillButton.setStyleSheet("background-color:")
            QApplication.setOverrideCursor(Qt.ArrowCursor)
        elif mode == 2:
            self.fillButton.setStyleSheet("background-color: #85adad")
            self.brushButton.setStyleSheet("background-color:")
            self.recButton.setStyleSheet("background-color:")
            QApplication.setOverrideCursor(Qt.PointingHandCursor)

    def segmap2rgb(self, img):
        part_colors = np.array([[0, 0, 0], [127, 212, 255], [255, 255, 127], [255, 255, 170],  # 'skin',1 'eye_brow'2,  'eye'3
                       [240, 157, 240], [255, 212, 255],  # 'r_nose'4, 'l_nose'5
                       [89, 64, 92], [237, 102, 99], [181, 43, 101],  # 'mouth'6, 'u_lip'7,'l_lip'8
                       [0, 255, 85], [0, 255, 170],  # 'ear'9 'ear_r'10
                       [255, 255, 170],
                       [127, 170, 255], [85, 0, 255], [255, 170, 127],  # 'neck'11, 'neck_l'12, 'cloth'13
                       [212, 127, 255], [0, 170, 255],  # , 'hair'14, 'hat'15
                       [255, 255, 0], [255, 255, 85], [255, 255, 170],
                       [255, 0, 255], [255, 85, 255], [255, 170, 255],
                       [0, 255, 255], [85, 255, 255], [170, 255, 255], [100, 150, 200]]).astype('int')

        part_colors = np.array([
                [0, 0, 0],
                [204, 0, 0],
                [76, 153, 0], 
                [204, 204, 0], 
                [51, 51, 255], 
                [204, 0, 204], 
                [0, 255, 255], 
                [255, 204, 204], 
                [102, 51, 0], 
                [255, 0, 0], 
                [102, 204, 0], 
                [255, 255, 0], 
                [0, 0, 153], 
                [0, 0, 204], 
                [255, 51, 153], 
                [0, 204, 204], 
                [0, 51, 0], 
                [255, 153, 51], 
                [0, 204, 0]
        ]).astype('int')

        condition_img_color = part_colors[img]
        return condition_img_color


    def id_remap(self, seg, type=None):
        # remap_list = np.array([0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 15, 16]).astype('uint8')
        if type == 'celeba':
            remap_list = np.array([0, 1, 6, 7, 4, 5, 2, 2, 10, 11, 12, 8, 9, 15, 3, 17, 16, 18, 13, 14]).astype('uint8')
        remap_list = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]).astype('uint8')
        return remap_list[seg.astype('int')]

    def id_remap_tensor(self, seg):
        remap_list = torch.tensor([0, 1, 6, 7, 4, 5, 2, 2, 10, 11, 12, 8, 9, 15, 3, 17, 16, 18, 13, 14]).float()
        return remap_list[seg.long()].to(seg.device)


    def get_label(self, idx):
        # if self.synthetic_label is None:
        #     print("True")
        #     return self.synthetic_label
        try:
            label = [self.labels[idx[-21:]]]
            label = np.array(label)
            label[:, [1,2,5,6,9,10]] *= -1
            label = label.astype({1: np.int64, 2: np.float32}[label.ndim])
            return torch.tensor(label).cuda()
        except:
            camera_points, phi, theta = sample_camera_positions('cuda', n=1, r=2.7, horizontal_mean=self.yaw_orig, 
                                                        vertical_mean=self.pitch_orig, horizontal_stddev=0., vertical_stddev=0., mode=None)
            c = create_cam2world_matrix(-camera_points, camera_points, device='cuda')
            c = c.reshape(1,-1)
            c = torch.cat((c, torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).reshape(1, -1).to(c)), -1)
            return c

    @pyqtSlot()
    def save_img(self):

        ui_result_folder = './ui_results/' + os.path.basename(self.fileName)[:-4]

        os.makedirs(ui_result_folder,exist_ok=True)

        outName = os.path.join(ui_result_folder,datetime.datetime.now().strftime('%m%d%H%M%S') + '_segmap.png')
        cv2.imwrite(outName, self.mat_img)
        print('===> save segmap to %s'%outName)

        if self.fake_img is not None:
            outName = os.path.join(ui_result_folder,datetime.datetime.now().strftime('%m%d%H%M%S') + '_render.png')
            utils.save_image(self.fake_img.detach().cpu(), outName, normalize=True, range=(-1,1))

        if self.w is not None:
            outName = os.path.join(ui_result_folder,datetime.datetime.now().strftime('%m%d%H%M%S') + '_ws.pt')
            torch.save(self.w, outName)
            


    @pyqtSlot()
    def switch_labels(self, label):
        self.scene.label = label
        self.scene.color = number_color[label]
        _translate = QCoreApplication.translate
        self.color_Button.setText(_translate("Form", "%s" % number_object[label]))
        self.color_Button.setStyleSheet("background-color: %s;" % self.scene.color+ " color: black")

    def parsing_img(self, bisNet, img, argmax=True, return_mask=True, with_grad=False, remap=True):
        if not with_grad:
            with torch.no_grad():
                segmap = bisNet(img)[0]
                if argmax:
                    segmap = segmap.argmax(1, keepdim=True)
                if remap:
                    segmap = self.id_remap_tensor(segmap)
        else:
                segmap = bisNet(img)[0]
                if argmax:
                    segmap = segmap.argmax(1, keepdim=True)
                if remap:
                    segmap = self.id_remap_tensor(segmap)
        if return_mask:
                segmap = self.scatter(segmap)
        return img, segmap

    def scatter(self, condition_img, classSeg=19, label_size=(512, 512)):
        batch, c, height, width = condition_img.size()
        if height != label_size[0] or width != label_size[1]:
            condition_img= torch.nn.functional.interpolate(condition_img, size=label_size, mode='nearest')
        input_label = torch.zeros(batch, classSeg, *label_size, device=condition_img.device)
        return input_label.scatter_(1, condition_img.long(), 1)


    @pyqtSlot()
    def undo(self):
        self.scene.undo()

    def startScreening(self):
        self.isScreening,self.frameLog = True,{}
        self.starTime = datetime.datetime.now().strftime('%H_%M_%S_%f')

    def saveScreening(self):
        os.makedirs('./frameLog', exist_ok=True)
        name = './frameLog/%s.pkl'%self.starTime
        with open(name, 'wb') as f:
            pickle.dump(self.frameLog, f)
        print('====> saved frame log to %s'%name)

    def cleanForground(self):
        self.mat_img[:] = 0
        self.update_segmap_vis(self.mat_img)
        self.frameLog[datetime.datetime.now().strftime('%H:%M:%S:%f')] = {'undo': len(self.frameLog.keys())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--g_ckpt', default='pretrained_models/ide3d/network-snapshot-014480.pkl', type=str)
    parser.add_argument('--e_ckpt', default='pretrained_models/hybrid_encoder_w_real/network-snapshot-000220.pkl', type=str)
    parser.add_argument('--target_code', default=None, type=str)
    parser.add_argument('--target_label', default=None, type=str)
    parser.add_argument('--inversion', action='store_true')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ex = ExWindow(args)
    sys.exit(app.exec_())
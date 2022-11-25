from numpy.lib.arraysetops import isin
import torch
from torch._C import device
import torch.nn as nn
import numpy as np
from torch.nn.modules import module

# import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    RasterizationSettings, 
    MeshRasterizer
)
from pytorch3d.transforms.rotation_conversions import (
    quaternion_to_matrix,
    _copysign,
    quaternion_apply
)
from utils.utils import transform_mesh


class BFM(nn.Module):
    def __init__(self, fmodel='./models/bfm_models/expression_blendshape.npz',
                 chosen_file='./models/bfm_models/index_33744pts_bs.txt',
                 face_file='./models/bfm_models/Face_2106.txt',
                 param_dist_path='./models/bfm_models/param_dist.npz',
                 ):
        super(BFM, self).__init__()
        
        # load data
        data = np.load(fmodel)
        param_dist = np.load(param_dist_path)
        vids = np.loadtxt(chosen_file).astype(np.int32)
        faces = np.loadtxt(face_file,np.str_)[:,-3:].astype(np.int32)-1
        
        # parse data
        mean_shape = data['mean_shape']  # N*3
        w_exp = data['w_exp'] # 3N * 50
        w_shape = data['w_shape'] #3N * 199
        
        # parse data shape
        N = mean_shape.shape[0]
        self.n_sp = w_shape.shape[1]
        self.n_ep = w_exp.shape[1]
        
        # parse param_dist data
        self.xs_mean = nn.Parameter(torch.from_numpy(param_dist['xs_mean'].reshape(1,-1)).float(),
                                       requires_grad=False)
        self.xs_std = nn.Parameter(torch.from_numpy(param_dist['xs_std'].reshape(1,-1)).float(),
                                  requires_grad=False)
        self.xe_mean = nn.Parameter(torch.from_numpy(param_dist['xe_mean'].reshape(1,-1)).float(),
                                  requires_grad=False)
        self.xe_std = nn.Parameter(torch.from_numpy(param_dist['xe_std'].reshape(1,-1)).float(),
                                  requires_grad=False)
        
        # pick desired points
        mean_shape = mean_shape[vids,:]
        w_exp = w_exp.reshape((N, 3, -1))
        w_exp = w_exp[vids,:,:].reshape((-1, self.n_ep))
        w_shape = w_shape.reshape((N, 3, -1))
        w_shape = w_shape[vids,:,:].reshape((-1, self.n_sp))
        
        # register parameters
        self.mean_shape = nn.Parameter(torch.from_numpy(mean_shape).float(),
                                       requires_grad=False)  # N*3
        self.w_exp = nn.Parameter(torch.from_numpy(w_exp).float(),
                                  requires_grad=False) # 3N * 50
        self.w_shape = nn.Parameter(torch.from_numpy(w_shape).float(),
                                    requires_grad=False) #3N * 199
        self.faces = nn.Parameter(torch.from_numpy(faces).int(),
                                  requires_grad=False)

    def set_device(self, device=None):
        self.device = device

    def forward(self, params, batch=True, denorm_3dmm=True):
        
        if not batch:
            params = params.unsqueeze(0)
        
        # synthesis meshes
        N = params.size(0)
        # print(f'denorm_3dmm in bfm: {denorm_3dmm}')
        
        # if denorm_3dmm:
        xs = params[:, :self.n_sp] * self.xs_std + self.xs_mean
        xe = params[:, self.n_sp:(self.n_sp+self.n_ep)] * self.xe_std + self.xe_mean

        xs = xs.view(N, -1, 1)
        xe = xe.view(N, -1, 1)
        mesh_pre = self.mean_shape + \
                    torch.matmul(self.w_shape, xs).view(N, -1, 3) + \
                    torch.matmul(self.w_exp, xe).view(N, -1, 3)

        if not batch:
            mesh_pre = mesh_pre.squeeze()
        
        return mesh_pre # Bs, Vert_num, 3

    def param2mesh(self, params, batch=True):
        
        if not batch:
            params = params.unsqueeze(0)
        
        # synthesis meshes
        N = params.size(0)
        xs = params[:, :self.n_sp]
        xs = xs.view(N, -1, 1)
        xe = params[:, self.n_sp:(self.n_sp+self.n_ep)]
        xe = xe.view(N, -1, 1)
        mesh_pre = self.mean_shape + \
                    torch.matmul(self.w_shape, xs).view(N, -1, 3) + \
                    torch.matmul(self.w_exp, xe).view(N, -1, 3)

        if not batch:
            mesh_pre = mesh_pre.squeeze()
        
        return mesh_pre # Bs, Vert_num, 3
 

class DepthRendder(nn.Module):
    def __init__(self, faces):
        super(DepthRendder, self).__init__()
        self.faces = faces
        # self.meshes = Meshes(torch.rand(1,3000,3).to(device), self.faces.expand(1,*list(self.faces.shape)).to(device))
    
    def set_device(self, device=None):
        self.device = device

    def forward(self, verts, pitch=None, yaw=None, img_size=128):
        self.device = verts.device
        # print('get_depth.device:',self.device)
        # print('verts.requires_grad:',verts.requires_grad)
        bs = verts.shape[0]
        
        # world2cam_matrix = torch.inverse(cam2world_matrix)
        # elev = torch.linspace(0, 180, bs)
        # azim = torch.linspace(-180, 180, bs)
        # R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
        # print('pitch.shape:',pitch.shape)
        # print('yaw.shape:',yaw.shape)
        
        if pitch is not None and yaw is not None:
            elev = np.pi/2 - pitch.flatten()
            azim = np.pi/2 - yaw.flatten()
            # print('elev.shape:',elev.shape)
            # print('azim.shape:',azim.shape)
            R, T = look_at_view_transform(dist=1.0, elev=elev, azim=azim, degrees=False)
        else:
            R, T = look_at_view_transform(dist=1.0, elev=0, azim=0, degrees=False)

        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=12, znear=0.88, zfar=1.12)
        # if self.meshes is None:
        self.meshes = Meshes(verts, self.faces.expand(bs,*list(self.faces.shape)))
        
        raster_settings = RasterizationSettings(
                                image_size=img_size, 
                                blur_radius=0.0, 
                                faces_per_pixel=1,
                                # z_clip_value=0.01
                            )
        rasterizer = MeshRasterizer(
                        cameras=cameras,
                        raster_settings=raster_settings
                    )
        fragments = rasterizer(self.meshes)
        verts_ndc = transform_mesh(self.meshes,cameras)
        # verts_ndc = None
        # torch.save(verts_ndc, './verts_ndc.pth')
        depth_maps = fragments.zbuf
        # print('depth_maps.shape:',depth_maps.shape)
        # depth_maps = depth_maps.permute(0, 3, 1, 2)
        masks = depth_maps>=0
        bg_masks = depth_maps<0
        depth_maps[bg_masks] = depth_maps[masks].median()
        # depth_maps[bg_masks] = depth_maps[masks].max()

        # print('verts.device:',verts.device)
        # print('meshes.device:',meshes.device)
        # print('depth_maps.device:',depth_maps.device)
        # print('masks.device:',masks.device)

        # if False:
        #     depth_maps = torch.rand(bs,img_size,img_size).to(device)
        #     masks = depth_maps>0
        #     masks = masks.to(device)
        return depth_maps, masks, verts_ndc

class Transform(nn.Module):
    def __init__(self, rotation_mode='Euler', rotation_trainable=True):
        super(Transform, self).__init__()
        
        rotation_modes = ['Euler', 'Quaternion']
        assert rotation_mode in rotation_modes, f"Mode {rotation_mode} not supported!"
        self.rotation_mode = rotation_mode
        self.rotation_trainable = rotation_trainable
        
        if self.rotation_mode == 'Euler':
            # register parameters
            self.p_y_r = nn.Parameter(
                torch.rand(3)*1e-9, requires_grad=self.rotation_trainable
            )
        elif self.rotation_mode == 'Quaternion':
            self.q = nn.Parameter(
                torch.from_numpy(np.array([1.1863831e+00, 5.8277890e-02, 2.7547943e-04, 2.4426032e-03]).astype(np.float32)), requires_grad=self.rotation_trainable
            )

        self.s = nn.Parameter(
                torch.from_numpy(np.array([0.09549354]).astype(np.float32)), requires_grad=self.rotation_trainable
            )
        self.t = nn.Parameter(
                torch.from_numpy(np.array([[0.00132487, -0.02791605, -0.08785161]]).astype(np.float32)).T, requires_grad=self.rotation_trainable
            )
        # self.debug_mesh = torch.nn.Parameter(torch.from_numpy(np.array(
        #             [[[0,0,0],[1,0,0],[0,1,0]]]
        #         )).float(), requires_grad=False)
            
    def set_device(self, device=None):
        self.device = device

    def get_Euler_R(self, _type='zyx'):
        if len(self.p_y_r.shape) == 1:
            batch = 1; reshape = True
            angle = self.p_y_r.view(1,-1)
        else:
            reshape = False
            angle = self.p_y_r
        c = torch.cos(angle)
        s = torch.sin(angle)
        one = torch.ones(len(c),1,dtype = c.dtype,device = c.device)
        zero= torch.zeros(len(c),1,dtype = c.dtype,device = c.device)

        Rx = torch.cat(( \
            one,      zero,       zero, \
            zero, c[:, 0:1], s[:, 0:1], \
            zero, -s[:, 0:1], c[:, 0:1]), -1).view(-1,3,3)

        Ry = torch.cat(( \
            c[:, 1:2], zero, s[:, 1:2], \
                zero, one,      zero, \
            -s[:, 1:2], zero,c[:, 1:2]), -1).view(-1,3,3)

        Rz = torch.cat(( \
            c[:,2:3],s[:,2:3],zero, \
            -s[:,2:3], c[:,2:3],zero, \
                zero,       zero, one),-1).view(-1,3,3)
        R = torch.matmul(Rz, torch.matmul(Ry, Rx))

        if reshape:
            return R.view(3,3)
        else:
            return R

    def get_Quaternion_R(self):
        return quaternion_to_matrix(self.q)

    def get_R(self):
        if self.rotation_mode == 'Euler':
            return self.get_Euler_R()
        elif self.rotation_mode == 'Quaternion':
            return self.get_Quaternion_R()
        
    def normalize_q(self):
        if self.rotation_mode == 'Quaternion':
            # self.q = self.q / _copysign(torch.sqrt(s), self.q[0])
            self.q = nn.Parameter(
                self.q / _copysign(torch.norm(self.q), self.q[0]),
                requires_grad=True
            )
            # self.q.copy_(self.q / _copysign(torch.sqrt(s), self.q[0]))

    def forward(self, mesh, use_batch=True):
        # self.normalize_q()
        if not use_batch:
            mesh = mesh.unsqueeze(0)
        
        bs, v_num, d = mesh.shape
        if self.rotation_mode == 'Euler':
            # print('self.sR.device:',self.sR.device)
            # print('self.T.device:',self.T.device)
            # print('self.device:',self.device)
            sR = self.get_Euler_R() * self.s
            assert d==sR.shape[0] and d==sR.shape[1] and d==self.t.shape[0]
            # If self.sR is defined in the __init__,
            # the connection between the sR and p_y_r will be released after back propagation.
            # And you will get an Error reading "Trying to backward through the graph a second time".
            mesh_trans = torch.matmul(mesh,sR.T)+self.t.T
            # mesh_trans = mesh+self.t.T
            # mesh_trans = mesh
        elif self.rotation_mode == 'Quaternion':
            q = self.q / (torch.norm(self.q) + torch.finfo(self.q.dtype).eps)
            # q.requires_grad = True
            # print("self.q:",self.q)
            # print("torch.norm(self.q):",torch.norm(self.q))
            # print("q:",q)
            
            # print("q:",q)
            # print("q.requires_grad:",q.requires_grad)
            mesh_trans = quaternion_apply(q, mesh)
            # mesh_trans.requires_grad = True
            # print("mesh_trans:",mesh_trans)
            mesh_trans = mesh_trans*self.s+self.t.T
            # mesh_trans.requires_grad = True
            # print("mesh_trans.requires_grad:",mesh_trans.requires_grad)
            # print('================')
        
        if not use_batch:
            mesh_trans = mesh_trans.squeeze(0)
            # mesh_trans.requires_grad = True

        return mesh_trans

class LandmarkTable(nn.Module):
    """docstring for LandmarkTable"""
    def __init__(self, fldmk_table='./models/bfm_models/landmark_table_3060pts.npz'):
        super(LandmarkTable, self).__init__()
        ldmk_table = np.load(fldmk_table)
        
        # 这里的min max 是角度, 按照opengl ,yaw, pitch
        self.n_y_p = nn.Parameter(
            torch.from_numpy(ldmk_table["n_y_p"]).float(), requires_grad=False
        )
        
        self.min_v = nn.Parameter(
            torch.from_numpy(np.radians(ldmk_table["min_v"])).float(), requires_grad=False
        )
        
        self.max_v = nn.Parameter(
            torch.from_numpy(np.radians(ldmk_table["max_v"])).float(), requires_grad=False
        )
        
        self.vids = nn.Parameter(
            torch.from_numpy(ldmk_table["vids"]).int(), requires_grad=False
        )
        
        self.wets = nn.Parameter(
            torch.from_numpy(ldmk_table["wets"]).float(), requires_grad=False
        )
                
    def set_device(self, device=None):
        self.device = device

    def get_id(self, val, n, v_min, v_max):
        val = val.clone()
        val[val<v_min] = v_min
        val[val>v_max] = v_max
        if n <= 1:
            raise ZeroDivisionError(
                "Divide the interval into at least two parts!")
        interval = (v_max - v_min) / (n - 1)
        return ((val - v_min) / interval + 0.5).int()

    def get_rect_id(self, y_p):
        x = self.get_id(y_p[0], self.n_y_p[0], self.min_v[0], self.max_v[0])
        y = self.get_id(y_p[1], self.n_y_p[1], self.min_v[1], self.max_v[1])
        idx = y * self.n_y_p[0] + x
        return idx

    def forward(self, yaw, pitch, is_opencv=True):
        p = np.pi/2 - pitch
        y = np.pi/2 - yaw
        y = -y # yaw: left is negtive; pitch: up is negtive
        idx = self.get_rect_id([y, p]).long()
        # print('idx:',idx)
        # print('idx.shape:',idx.shape)
        # print('self.vids[idx].shape:',self.vids[idx].shape)
        # print('self.wets[idx].shape:',self.wets[idx].shape)
        
        return self.vids[idx].squeeze(1).long(), self.wets[idx].squeeze(1), idx


if __name__ == '__main__':
    test_module = 'DepthRendder'
    if test_module == 'BFM':
        bs = 2
        param_s_e = np.random.randn(bs,199+50)
        params_pred = torch.FloatTensor(param_s_e)

        bfm = BFM(fmodel='./bfm_models/expression_blendshape.npz',
                chosen_file='./bfm_models/index_2106pts_bs.txt',
                face_file='./bfm_models/Face_2106.txt',
                param_dist_path='./bfm_models/param_dist.npz')
        renderer = DepthRendder(bfm.faces,bfm.device)
        transformer = Transform(bfm.device)
        mesh_pre = bfm(params_pred)
        mesh_pre = transformer(mesh_pre).detach()
        mesh_numpy = mesh_pre.numpy()
        for i,mesh in enumerate(mesh_numpy):
            with open(f"test_bfm_{i}.obj","w") as w:
                l,c = mesh.shape
                for i in range(l):
                    w.write("v ")
                    for j in range(c):
                        w.write(str(mesh[i,j])+' ')
                    w.write('\n')
        # print(mesh_pre.shape, bfm.faces.shape)
        depths, masks = renderer(mesh_pre)
        depths = depths.permute(0, 3, 1, 2)
        masks = masks.permute(0, 3, 1, 2)
        import matplotlib.pyplot as plt
        import cv2
        for i,depth in enumerate(depths):
            mask = masks[i]
            depth = depth*mask
            foreground_entries = depth[mask].cpu().numpy()
            f_min = foreground_entries.min()
            f_max = foreground_entries.max()
            output = depth.permute(1,2,0)[...,0].cpu().numpy()
            print(f"range of depth: {f_min}, {f_max}")
            cv2.imwrite(f'depth_{i}.png',(output-f_min)/(f_max-f_min)*255)
    elif test_module == 'Transform':
        mesh = torch.rand(4,1000,3)
        transformer = Transform()
        mesh_trans = transformer(mesh)
    elif test_module == 'DepthRendder':
        bs = 2
        param_s_e = np.random.randn(bs,199+50)
        params_pred = torch.FloatTensor(param_s_e)

        bfm = BFM(fmodel='./models/bfm_models/expression_blendshape.npz',
                chosen_file='./models/bfm_models/index_2106pts_bs.txt',
                face_file='./models/bfm_models/Face_2106.txt',
                param_dist_path='./models/bfm_models/param_dist.npz')
        renderer = DepthRendder(bfm.faces,bfm.device)
        transformer = Transform(bfm.device)
        
        mesh_pre = bfm(params_pred)
        mesh_pre = transformer(mesh_pre)

        depths, masks, verts_ndc = renderer(mesh_pre)
        depths = depths.permute(0, 3, 1, 2)
        masks = masks.permute(0, 3, 1, 2)
        target = torch.rand(depths.shape)
        
        mse = torch.nn.MSELoss()
        opt = torch.optim.Adam(transformer.parameters(), lr=0.01, betas=(0,0.9), weight_decay=0)
        steps = 1000
        for i in range(steps):
            mesh_t = transformer(mesh)
            print(mesh_t.requires_grad)
            loss = mse(mesh_t, target)
            print(f'--------{i}--------')
            print("loss:",loss.item())
            loss.backward()
            # print("transformer.q:",transformer.q)
            # print("grad:",transformer.q.grad)
            opt.step()
            # print("q after step:",transformer.q)
            opt.zero_grad()
            
            # transformer.normalize_q()
            # print("q after normalization:",transformer.q)
            break
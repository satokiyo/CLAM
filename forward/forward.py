from models.nuclei_detection.models.dmcount_model import DMCountModel
import argparse
import torch
import os
import numpy as np
#import datasets.crowd as crowd
import time
import utils.utils as utils
import h5py
import math
from torchvision import transforms
import cv2
from datasets.dataset_h5 import Whole_Slide_Bag_FP
from pathlib import Path
from wsi_core.wsi_utils import save_hdf5

class Config():
    def __init__(self):
        self.device: int = 0
        self.gpu: int = 1
        self.input_size: int = 1024
        self.crop_size: int = 1024
        self.encoder_name: str = 'se_resnext50_32x4d'
        self.classes: int = 1
        self.scale_pyramid_module: int = 1
        self.use_attention_branch: int = 0
        self.downsample_ratio: int = 1
        self.deep_supervision: int = 1
        self.use_ssl: int = 1
        self.pred_density_map_path: str = '/media/prostate/20210331_PDL-1/CLAM/result/nuclei_detection'
        #self.pred_density_map_path: str = ''

class Transform():
    def __init__(self):
        self.trans = transforms.Compose([
            transforms.ToTensor(),
#            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __call__(self, patch):
        return self.trans(patch)


def forward_nuclei_detect(file_path, wsi_object, model_path): # forward using trained model and save info to .h5 file.
    start = time.time()
    args = Config()
    transform = Transform()
    args.gpu = 0 # cpu
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)  # set vis gpu
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')
    
    model = DMCountModel(args)
    model.to(device)
    model.load_state_dict(torch.load(model_path, device))
    model.eval()

    dataset = Whole_Slide_Bag_FP(file_path, wsi_object.getOpenSlide(), pretrained=False, custom_transforms=False, custom_downsample=1, target_patch_size=-1)
    total = len(dataset)
    slide_id = dataset.slide_id
    dataloader = torch.utils.data.DataLoader(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True if args.gpu==1 else False)

    verbose=1
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)
        
    xy_locations = []
    dab_intensity_map = []
    tc_positive_indices = []
    for patch_id, (patch, coord) in enumerate(dataloader):
        if verbose > 0:
            if patch_id % ten_percent_chunk == 0:
                print('progress: {}/{} forward finished'.format(patch_id, total))
        
        inputs = patch.to(device)
        with torch.set_grad_enabled(False):
            if args.deep_supervision:
                outputs, intermediates = model(inputs)
                del intermediates
            else:
                outputs = model(inputs)

        print(f'estimated count {torch.sum(outputs).item()} cells')

        if args.pred_density_map_path: # save ROI overlay images
            if not Path(args.pred_density_map_path).exists():
                Path(args.pred_density_map_path).mkdir(parents=True)
            vis_img = outputs[0].detach().cpu().numpy()
            del outputs
            # normalize density map values from 0 to 1, then map it to 0-255.
            vis_img = (vis_img - np.min(vis_img)) / np.ptp(vis_img)
            vis_img2 = vis_img.copy()
            vis_img = (vis_img*255).astype(np.uint8)
            vis_img = vis_img.transpose(1,2,0) # channel last
            vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_VIRIDIS)
            if args.downsample_ratio > 1:
                vis_img = cv2.resize(vis_img, dsize=(args.input_size, args.input_size), interpolation=cv2.INTER_NEAREST)
                vis_img2 = cv2.resize(vis_img2, dsize=(args.input_size, args.input_size), interpolation=cv2.INTER_NEAREST)
    
            vis_img = vis_img[:,:,::-1] # convert to RGB
            #vis_img = cv2.resize(vis_img, dsize=(int(self.args.input_size), int(self.args.input_size)), interpolation=cv2.INTER_NEAREST)
            org_img = inputs[0].detach().cpu().numpy()
            org_img = (org_img - np.min(org_img)) / np.ptp(org_img)
            org_img = (org_img*255).astype(np.uint8)
            org_img = org_img.transpose(1,2,0) # channel last
            if (vis_img.shape[:2]) != (org_img.shape[:2]):
                vis_img = vis_img.resize(org_img.shape[:2])
            # overlay
            overlay = np.uint8((org_img/2) + np.uint8(vis_img/2)) # RGB
            coord = coord[0].detach().cpu().numpy()
            cv2.imwrite(os.path.join(args.pred_density_map_path, str('_'.join([slide_id, str(coord[0]), str(coord[1])]) + '.jpg')), overlay.astype(np.uint8)[:,:,::-1])
     
            # detect/draw center point
            paint_center(args, overlay, vis_img2.transpose(1,2,0), taus=[-1], org_img=org_img, name=str('_'.join([slide_id, str(coord[0]), str(coord[1])]) + '.jpg'))

        # TODO return (x,y) location arry, dab_intensity_map, tc_positive_indices for each patch
        # TODO no need predition map
        n_cells = 100
        xy_locations.append(tuple((np.zeros(n_cells), np.zeros(n_cells))))
        dab_intensity_map.append(np.zeros(dataset.patch_size, dataset.patch_size, 1))
        tc_positive_indices.append(np.zeros(n_cells))

    #results = np.array([result for result in results if result is not None])
    xy_locations = np.array([result for result in xy_locations if result is not None])
    dab_intensity_map = np.array([result for result in dab_intensity_map if result is not None])
    tc_positive_indices = np.array([result for result in tc_positive_indices if result is not None])

#    file = h5py.File(file_path, 'w')
#    dset = file['coords']
#    coords = dset[:]
#    dset.attrs['nuclei_detection']['nuclei_coords'] = results # 1:n = roi coords : nuclei coords
#    print('Extracted {} coordinates'.format(len(results)))

#    save_path_hdf5 = os.path.join(save_path, str(self.name) + '.h5')
            
    if len(xy_locations)>1:
        asset_dict = {'nuclei_detection' : xy_locations} # locations of detected center of nuclei
        
        attr = {'patch_size'           : dataset.patch_size,  # patch_size. Not patch size in reference frame(level 0)
                'patch_level'          : dataset.patch_level, # patch_level. Not ref level(level 0)
                'intensity_thres'      : 175,                 # for BN method to detect DAB+ cell
                'area_thres'           : 0.1,                 # over 10% area
                'dab_intensity_map'    : dab_intensity_map,   # [0,255] heatmap for representing DAB intensity
                'tc_positive_indices'  : tc_positive_indices, # TC(+)となる核位置のインデクス。核位置毎に1or0のフラグを与える。 # 閾値が変わったらdab_intensity_mapを元に再計算してアップデートする
                #TODO contourごとに計算できるように、各contourに対応するpatchのインデクスを持たせ、assetからインデクシングして取得できるようにする。全体のsummaryの計算は全てのインデクスを使えばいい。
                }
                #'downsample':             self.level_downsamples[patch_level],
                #'downsampled_level_dim' : tuple(np.array(self.level_dim[patch_level])),
                #'level_dim':              self.level_dim[patch_level],
                #'name':                   self.name,
                #'save_path':              save_path}

        attr_dict = { 'nuclei_detection' : attr}

    else:
        asset_dict, attr_dict = {}, {}


    if len(asset_dict) > 0:
        #TODO this overwrite "coords" asset. fix to save new asset additionally.
        save_hdf5(file_path, asset_dict, attr_dict, mode='a')

    return file_path






#
#    if len(results)>1:
#        asset_dict = {'coords' :          results}
#        
#        attr = {'patch_size' :            patch_size, # patch_size. Not patch size in reference frame(level 0)
#                'patch_level' :           patch_level, # patch_level. Not ref level(level 0)
#                'downsample':             self.level_downsamples[patch_level],
#                'downsampled_level_dim' : tuple(np.array(self.level_dim[patch_level])),
#                'level_dim':              self.level_dim[patch_level],
#                'name':                   self.name,
#                'save_path':              save_path}
#
#        attr_dict = { 'coords' : attr}
#        return asset_dict, attr_dict
#
#    file.close()
#    return heatmap


def forward_segmentation(file_path, wsi_object, model_path): # forward using trained model and save info to .h5 file.
    pass
 


def paint_center(args, img, liklihoodmap, taus=[-1], org_img=None, name=None):
    """ 'tau=-1 means dynamic Otsu thresholding. '
        'tau=-2 means Beta Mixture Model-based thresholding.') 
        RGB img org_img
        """
    # The estimated map must be thresholded to obtain estimated points
    for t, tau in enumerate(taus):
        if tau != -2:
            mask, _ = utils.threshold(liklihoodmap, tau)
        else:
            mask, _, mix = utils.threshold(liklihoodmap, tau)
        # Save thresholded map to disk
        #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'mask_tau_{round(tau, 4)}.jpg'), mask)
        cv2.imwrite(os.path.join(args.pred_density_map_path, str(name) + f'mask_tau_{round(tau, 4)}.jpg'), mask)
        #est_count_int = int(torch.sum(outputs).item())

        # method 1. detect center by GMM fitting
        #centroids_wrt_orig = utils.cluster(mask, est_count_int, max_mask_pts=500)

        # meghod 2. detect center by labeling
        # 膨張・収縮処理
        #kernel = np.ones((2, 2), np.uint8)
        ##mask = cv2.dilate(mask, kernel)
        ##mask = cv2.erode(mask, kernel)
        ##mask = cv2.erode(mask, kernel)
        #mask = cv2.erode(mask, kernel)
        ##mask = cv2.erode(mask, kernel)
        ##mask = cv2.dilate(mask, kernel)
        #nLabels, labelImages, data, center = cv2.connectedComponentsWithStats(mask)
        #centroids_wrt_orig = center[:,[1,0]].astype(np.uint32)

        # method 3. find local maxima
        kernel = np.ones((2, 2), np.uint8)
        mask_copy = mask.copy()
        dilate = cv2.dilate(mask_copy, kernel)
        erode = cv2.erode(mask_copy, kernel)
        #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'dilate.jpg'), dilate)
        #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'erode.jpg'), erode)
        peak = dilate - mask_copy
        flat = mask_copy - erode
        #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'mask_tau_{round(tau, 4)}_peak.jpg'), cv2.bitwise_not(peak))
        #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'mask_tau_{round(tau, 4)}_flat.jpg'), cv2.bitwise_not(flat))
        peak[flat > 0] = 255
        #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'mask_tau_{round(tau, 4)}_peak2.jpg'), cv2.bitwise_not(peak))
        con, hierarchy = cv2.findContours(peak,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # compute the center of the contour
        center = []
        for c in con:
            M = cv2.moments(c)
            cX = int(M["m10"] / (M["m00"] + 1e-5))
            cY = int(M["m01"] / (M["m00"] + 1e-5))
            center.append(np.array([cX, cY]))
        center = np.array(center).astype(np.uint32)
        if len(center) == 0:
            print("count zero!!")
            print("continue")
            continue
        centroids_wrt_orig = center[:,[1,0]]
        print(f'ncenters: {len(center)}')



        # Paint a cross at the estimated centroids
        img_with_x_n_map = utils.paint_circles(img=img,
                                               points=centroids_wrt_orig,
                                               color='red',
                                               crosshair=True)
        # Save to disk
        cv2.imwrite(os.path.join(args.pred_density_map_path, str(name) + f'painted_on_estmap_tau_{round(tau, 4)}.jpg'), img_with_x_n_map[:,:,::-1])

        # voronoi
        rect = (0, 0, img.shape[0], img.shape[1])
        subdiv = cv2.Subdiv2D(rect)
        for p in center: # center[:,[1,0]]?
            subdiv.insert((int(p[0]), int(p[1])))
        facets, centers = subdiv.getVoronoiFacetList([])
        img_draw = img.copy()
        img_draw_DAB = org_img.copy()
        org_img_copy = org_img.copy().astype(np.float32)

        R = org_img_copy[:,:,0] # R-ch
        G = org_img_copy[:,:,1] # G-ch
        B = org_img_copy[:,:,2] # B-ch
        BN = 255*np.divide(B, (B+G+R), out=np.zeros_like(B), where=(B+G+R)!=0) # ref.paper : Automated Selection of DAB-labeled Tissue for Immunohistochemical Quantification
        DAB = 255 - BN
        #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'BN{round(tau, 4)}.jpg'), BN)
        #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'DAB{round(tau, 4)}.jpg'), DAB)

        #cv2.polylines(img_draw, [f.astype(int) for f in facets], True, (255, 255, 255), thickness=2) # draw voronoi
        #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'painted_on_estmap_tau_{round(tau, 4)}_volonoi0.jpg'), img_draw)

        # voronoi with restricted radius
        mat = np.zeros((img_draw.shape[0], img_draw.shape[1]), np.uint8)
        facets = [f.astype(int) for f in facets]
        radius = 25
        cells = []
        for i,(center, points) in enumerate(zip(centers, facets)):
            mask1 = cv2.fillPoly(mat.copy(), [points], (255)) # make binary mask
            mask2 = cv2.circle(mat.copy(),(int(center[0]), int(center[1])), radius, (255), -1)
            intersection = mask1 & mask2
            con, hierarchy = cv2.findContours(intersection,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            # Create a mask image that contains the contour filled in
            mask3 = np.zeros_like(mat, np.uint8)
            mask3 = cv2.drawContours(mask3, con, -1, 255, 1)
            contour_area = np.count_nonzero(mask3) # 後で使うpixel単位面積
            mask4 = (mask3==255).astype(np.uint8) # contour region mask
            contour_DAB = DAB * mask4
            intensity_thres=175
            area_thres=0.1 # over 10% area
            over_thres_area = np.count_nonzero(contour_DAB > intensity_thres)

            if (over_thres_area / contour_area) > area_thres: # 1-NonNucleusArea > 0.1
                #描画
                img_draw_DAB = cv2.drawContours(img_draw_DAB , con, -1, (255,0,0), 1) # red
            else:
                img_draw_DAB = cv2.drawContours(img_draw_DAB , con, -1, (0,180,180), 1) # cyan

            img_draw = cv2.drawContours(img_draw, con, -1, (0,255,0), 1) # draw voronoi with restricted redius

        #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'painted_on_estmap_tau_{round(tau, 4)}_volonoi.jpg'), img_draw) 
        #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'painted_on_estmap_tau_{round(tau, 4)}_dab.jpg'), img_draw_DAB)
        cv2.imwrite(os.path.join(args.pred_density_map_path, str(name) + f'painted_on_estmap_tau_{round(tau, 4)}_volonoi.jpg'), img_draw[:,:,::-1]) 
        cv2.imwrite(os.path.join(args.pred_density_map_path, str(name) + f'painted_on_estmap_tau_{round(tau, 4)}_dab.jpg'), img_draw_DAB[:,:,::-1])
        


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
        #self.pred_density_map_path: str = '/media/prostate/20210331_PDL-1/CLAM/result/nuclei_detection'
        self.pred_density_map_path: str = ''

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

    file = h5py.File(file_path, 'r')
    dset = file['coords']
    coords = dset[:]

    #if 'downsampled_level_dim' in dset.attrs.keys(): # patch level. Not level 0
    #    w, h = dset.attrs['downsampled_level_dim'] # patch size at patch level. Not at level 0
    #else:
    #    w, h = dset.attrs['level_dim']
    #print('original size: {} x {}'.format(w, h))

    #w, h = wsi.level_dimensions[0] # image size at level0
    #print('start stitching {}'.format(dset.attrs['name']))
    #print('original size: {} x {}'.format(w, h))

    #w, h = wsi.level_dimensions[vis_level] # image size at 'heatmap level' for stitching. (Not level0 nor patch level)
    #print('downscaled size for stiching: {} x {}'.format(w, h))
    
    patch_size = dset.attrs['patch_size']
    patch_level = dset.attrs['patch_level']
    slide_id = dset.attrs['name']
    print('patch size: {}x{} patch level: {}'.format(patch_size, patch_size, patch_level)) # patch levelでのpatch size
    #patch_size = tuple((np.array((patch_size, patch_size)) * wsi.level_downsamples[patch_level]).astype(np.int32)) # level0でのpatch size
    #print('ref patch size: {}x{}'.format(patch_size, patch_size))



#    if w*h > Image.MAX_IMAGE_PIXELS: 
#        raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)
#    
#    if alpha < 0 or alpha == -1:
#        heatmap = Image.new(size=(w,h), mode="RGB", color=bg_color)
#    else:
#        heatmap = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
#    
#    heatmap = np.array(heatmap)
#    heatmap = DrawMap(heatmap, dset, coords, downscaled_shape, indices=None, draw_grid=draw_grid)

    print('number of patches: {}'.format(len(coords)))
    indices = np.arange(len(coords))
    total = len(indices)
    verbose=1
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)
        
#    patch_size = tuple(np.ceil((np.array(patch_size)/np.array(downsamples))).astype(np.int32)) # convert patch_size from level 0 to vis_level.
#    print('downscaled patch size: {}x{}'.format(patch_size[0], patch_size[1]))
    
    results = []
    for patch_id in range(total):
        if verbose > 0:
            if patch_id % ten_percent_chunk == 0:
                print('progress: {}/{} forward finished'.format(patch_id, total))
        
        coord = coords[patch_id] # coord at level0
        patch = np.array(wsi_object.wsi.read_region(tuple(coord), patch_level, (patch_size, patch_size)).convert("RGB")) # coord is the location (x, y) tuple giving the top left pixel in the level 0 reference frame
#        coord = np.ceil(coord / downsamples).astype(np.int32) # convert coord for vis_level
#        canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3].shape[:2]
#        canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
#        if draw_grid:
#            DrawGrid(canvas, coord, patch_size)
#
#    return Image.fromarray(canvas)


        inputs = transform(patch)
        inputs = inputs.to(device)
        inputs = torch.unsqueeze(inputs, 0) # add axis 0
#        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            if args.deep_supervision:
                outputs, intermediates = model(inputs)
                del intermediates
            else:
                outputs = model(inputs)

        print(f'estimated count {torch.sum(outputs).item()} cells')

        if args.pred_density_map_path:
            print("###################")
            vis_img = outputs[0].detach().cpu().numpy()
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
            cv2.imwrite(os.path.join(args.pred_density_map_path, str('_'.join([slide_id, str(coord[0]), str(coord[1])]) + '.jpg')), overlay.astype(np.uint8)[:,:,::-1])
     
            # detect/draw center point
            paint_center(args, overlay, vis_img2.transpose(1,2,0), taus=[-1], org_img=org_img, name=str('_'.join([slide_id, str(coord[0]), str(coord[1])]) + '.jpg'))

        results.append(np.array(outputs[0]).transpose(1,2,0))

    results = np.array([result for result in results if result is not None])
        
    dset.attrs['nuclei_detection']['nuclei_coords'] = results # 1:n = roi coords : nuclei coords
    print('Extracted {} coordinates'.format(len(results)))
    import pdb;pdb.set_trace()

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
        #cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + f'painted_on_estmap_tau_{round(tau, 4)}.jpg'), img_with_x_n_map.transpose(1,2,0)[:,:,::-1])
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
        #e=1e-6
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
        


from models.nuclei_detection.models.dmcount_model import DMCountModel
from models.segmentation.models.segmentation_model import SegModel
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
from utils.file_utils import create_hdf5_group, create_hdf5_dataset, create_hdf5_attrs, open_hdf5_file
from PIL import Image

class Config():
    device: int = 0
    gpu: int = 0
#    gpu: int = 1
    input_size: int = 1024
    crop_size: int = 1024
    encoder_name: str = 'se_resnext50_32x4d'
    classes: int = 1
    scale_pyramid_module: int = 1
    use_attention_branch: int = 0
    downsample_ratio: int = 1
    deep_supervision: int = 1
    use_ssl: int = 1
    save: bool = False

class SegConfig():
    device: int = 0
    gpu: int = 0
#    gpu: int = 1
    input_size: int = 1024
    crop_size: int = 1024
    encoder_name: str = 'se_resnext50_32x4d'
    classes: int = 4                               # class数が異なる
    scale_pyramid_module: int = 1
    use_attention_branch: int = 0
    downsample_ratio: int = 1
    deep_supervision: int = 1
    use_ocr: int = 0
    use_ssl: int = 1
    activation: str = 'identity'
    save: bool = True
    palette: list = [0,0,0,
                     0,255,0,
                     255,0,0,
                     0,0,255]
 



def forward_detection(file_path, wsi_object, patch_size, model_path): # forward using trained model and save info to .h5 file.
    '''
    file_pathからHDF5ファイルを読み込み、/detection以下の階層から各パッチの座標、detectionのforwardは完了しているか等の情報を参照する。
    detectionの結果がないパッチのみ新たにforwardを実行し、detection結果をdatasetとして/detection/contourxx/patchxx/detection_loc以下に追加する

    Return:
      file_path
    '''
    start = time.time()
    conf = Config()
    conf.input_size=patch_size
    conf.crop_size=patch_size
    if conf.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(conf.device)  # set vis gpu
        device = torch.device(conf.device)
    else:
        device = torch.device('cpu')
    
    model = DMCountModel(conf)
    model.to(device)
    model.load_state_dict(torch.load(model_path, device))
    model.eval()

    dataset = Whole_Slide_Bag_FP(file_path, wsi_object.getOpenSlide(), pretrained=False, custom_transforms=False, custom_downsample=1, target_patch_size=-1, target='detection')
    total = len(dataset)
    slide_id = dataset.slide_id
    dataloader = torch.utils.data.DataLoader(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True if conf.gpu==1 else False)

    dset_name = 'detection_loc'
    f = open_hdf5_file(file_path, mode='a')

    verbose=1
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)

    if conf.save: # save ROI overlay images
        save_dir = Path(file_path).parent.parent / 'detection'
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
    else:
        save_dir = None
 
    for patch_id, (patch, coord, grp_name_parent) in enumerate(dataloader):
        if verbose > 0:
            if patch_id % ten_percent_chunk == 0:
                print('progress: {}/{} forward finished'.format(patch_id, total))

        # 既に結果があるパッチは飛ばす
        if dset_name in f[grp_name_parent[0]]:
            continue
        
        inputs = patch.to(device)
        with torch.set_grad_enabled(False):
            if conf.deep_supervision:
                outputs, intermediates = model(inputs)
                del intermediates
            else:
                outputs = model(inputs)
    
        #print(f'estimated count {torch.sum(outputs).item()} cells')
    
        vis_img = outputs[0].detach().cpu().numpy()
        del outputs
        # normalize density map values from 0 to 1, then map it to 0-255.
        vis_img = (vis_img - np.min(vis_img)) / np.ptp(vis_img)
        vis_img2 = vis_img.copy()
        vis_img = (vis_img*255).astype(np.uint8)
        vis_img = vis_img.transpose(1,2,0) # channel last
        vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_VIRIDIS)
        if conf.downsample_ratio > 1:
            vis_img = cv2.resize(vis_img, dsize=(conf.input_size, conf.input_size), interpolation=cv2.INTER_NEAREST)
            vis_img2 = cv2.resize(vis_img2, dsize=(conf.input_size, conf.input_size), interpolation=cv2.INTER_NEAREST)
        
        vis_img = vis_img[:,:,::-1] # convert to RGB
        #vis_img = cv2.resize(vis_img, dsize=(int(self.conf.input_size), int(self.conf.input_size)), interpolation=cv2.INTER_NEAREST)
        org_img = inputs[0].detach().cpu().numpy()
        org_img = (org_img - np.min(org_img)) / np.ptp(org_img)
        org_img = (org_img*255).astype(np.uint8)
        org_img = org_img.transpose(1,2,0) # channel last
        if (vis_img.shape[:2]) != (org_img.shape[:2]):
            vis_img = vis_img.resize(org_img.shape[:2])
        # overlay
        overlay = np.uint8((org_img/2) + np.uint8(vis_img/2)) # RGB
        coord = coord[0].detach().cpu().numpy()
    
        if save_dir: # save ROI overlay images
            cv2.imwrite(os.path.join(save_dir, str('_'.join([slide_id, str(coord[0]), str(coord[1])]) + '.jpg')), overlay.astype(np.uint8)[:,:,::-1])
    
        # detect/draw center point
        centroids_per_patch = get_centroids(overlay, vis_img2.transpose(1,2,0),
                                            tau=-1, org_img=org_img, name=str('_'.join([slide_id, str(coord[0]), str(coord[1])])),
                                            save_dir=save_dir)

        # save nuclei locations as dataset
        create_hdf5_dataset(f[grp_name_parent[0]], dset_name=dset_name, data=centroids_per_patch[:, [1,0]]) # reverse x,y

    f.close()

    return file_path




def forward_segmentation(file_path, wsi_object, patch_size, model_path): # forward using trained model and save info to .h5 file.
    '''
    file_pathからHDF5ファイルを読み込み、/segmentation以下の階層から各パッチの座標、segmentationのforwardは完了しているか等の情報を参照する。
    segmentationの結果がないパッチのみ新たにforwardを実行し、segmentation結果をdatasetとして/segmentation/contourxx/patchxx/segmap以下に追加する

    Return:
      file_path
    '''
    start = time.time()
    conf = SegConfig()
    conf.input_size=patch_size
    conf.crop_size=patch_size
    if conf.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(conf.device)  # set vis gpu
        device = torch.device(conf.device)
    else:
        device = torch.device('cpu')
    
    model = SegModel(conf)
    model.to(device)
    model.load_state_dict(torch.load(model_path, device))
    model.eval()

    dataset = Whole_Slide_Bag_FP(file_path, wsi_object.getOpenSlide(), pretrained=False, custom_transforms=False, custom_downsample=1, target_patch_size=-1, target='segmentation')
    total = len(dataset)
    slide_id = dataset.slide_id
    dataloader = torch.utils.data.DataLoader(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True if conf.gpu==1 else False)

    dset_name = 'segmap'
    f = open_hdf5_file(file_path, mode='a')

    verbose=1
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)

    if conf.save: # save ROI overlay images
        save_dir = Path(file_path).parent.parent / 'segmentation'
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
    else:
        save_dir = None


    for patch_id, (patch, coord, grp_name_parent) in enumerate(dataloader):
        if verbose > 0:
            if patch_id % ten_percent_chunk == 0:
                print('progress: {}/{} forward finished'.format(patch_id, total))

        # 既に結果があるパッチは飛ばす
        if dset_name in f[grp_name_parent[0]]:
            continue
        
        inputs = patch.to(device)
        with torch.set_grad_enabled(False):
            if conf.deep_supervision:
                outputs, intermediates = model(inputs)
                del intermediates
            else:
                outputs = model(inputs)

            # save segmentation map as dataset
            vis_img = outputs[0].detach().cpu().numpy()
#            create_hdf5_dataset(f[grp_name_parent[0]], dset_name=dset_name, data=vis_img)

            # save image
            if save_dir:
                PALETTE = conf.palette
                # normalize density map values from 0 to 1, then map it to 0-255.
                vis_img = (vis_img - np.min(vis_img)) / np.ptp(vis_img)
                vis_img = (vis_img*255).astype(np.uint8)
                vis_map = np.argmax(vis_img, axis=0)
                vis_map = Image.fromarray(vis_map.astype(np.uint8), mode="P")
                vis_map.putpalette(PALETTE)
                org_img = inputs[0].detach().cpu().numpy().transpose(1,2,0)
                del outputs
                del inputs
                org_img = (org_img - np.min(org_img)) / np.ptp(org_img)
                org_img = (org_img*255).astype(np.uint8)
                if (vis_map.size) != (org_img.shape[:1]):
                    vis_map = vis_map.resize(org_img.shape[:2])
                vis_map = np.array(vis_map.convert("RGB"))
                create_hdf5_dataset(f[grp_name_parent[0]], dset_name=dset_name, data=vis_map)
     
                # overlay
                overlay = np.uint8((org_img/2) + (vis_map/2))#.transpose(2,0,1)

                # save
                coord = coord[0].detach().cpu().numpy()
                cv2.imwrite(os.path.join(save_dir, str('_'.join([slide_id, str(coord[0]), str(coord[1])]) + '.jpg')), overlay.astype(np.uint8)[:,:,::-1])

    f.close()

    return file_path


def get_centroids(img, liklihoodmap, tau=-1, org_img=None, name=None, save_dir=None):
    """ 'tau=-1 means dynamic Otsu thresholding. '
        'tau=-2 means Beta Mixture Model-based thresholding.') 
        RGB img org_img
        """
    # The estimated map must be thresholded to obtain estimated points
    if tau != -2:
        mask, _ = utils.threshold(liklihoodmap, tau)
    else:
        mask, _, mix = utils.threshold(liklihoodmap, tau)
    # Save thresholded map to disk
    if save_dir:
        cv2.imwrite(os.path.join(save_dir, str(name + f'_mask_tau_{round(tau, 4)}.jpg')), mask)

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
    #if save_dir:
        #cv2.imwrite(os.path.join(save_dir, str(name + f'_dilate.jpg')), dilate)
        #cv2.imwrite(os.path.join(save_dir, str(name + f'_erode.jpg')), erode)
    peak = dilate - mask_copy
    flat = mask_copy - erode
    #if save_dir:
        #cv2.imwrite(os.path.join(save_dir, str(name) + f'_mask_tau_{round(tau, 4)}_peak.jpg')), cv2.bitwise_not(peak))
        #cv2.imwrite(os.path.join(save_dir, str(name) + f'_mask_tau_{round(tau, 4)}_flat.jpg')), cv2.bitwise_not(flat))
    peak[flat > 0] = 255
    #if save_dir:
        #cv2.imwrite(os.path.join(save_dir, str(name) + f'_mask_tau_{round(tau, 4)}_peak2.jpg')), cv2.bitwise_not(peak))
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
        print(f'ncenters: {len(center)}')
        centroids_wrt_orig = np.array([[-1, -1]]).astype(np.uint32)
    else:
        centroids_wrt_orig = center[:,[1,0]]
        print(f'ncenters: {len(center)}')

    if save_dir:
        # Paint a cross at the estimated centroids
        img_with_x_n_map = utils.paint_circles(img=img,
                                               points=centroids_wrt_orig,
                                               color='red',
                                               crosshair=True)
        # Save to disk
        cv2.imwrite(os.path.join(save_dir, str(name + f'_painted_on_estmap_tau_{round(tau, 4)}.jpg')), img_with_x_n_map[:,:,::-1])

    return centroids_wrt_orig



def detect_tc_positive_nuclei(file_path, wsi_object, intensity_thres=175, area_thres=0.1, radius=25):
    '''
    file_pathのHDF5ファイルの各パッチをイテレートし、TC(+)の結果が無かった場合だけ処理を実行し、該当のpatchのhdf5 groupに新たなdatasetを追加する
    もしくは結果はあるが、閾値が変更された場合に、既にある結果計算時の閾値が違うpatchに対してのみ処理を実行し、datasetの値を更新する

    file_pathからHDF5ファイルを読み込み、/detection以下の階層から各パッチの座標、detection結果を元にしたのTC(+)の計算は完了しているか等の情報を参照する。
    TC(+)の結果がないパッチのみ処理し、結果をdatasetとして/detection/contourxx/patchxx/detection_dab_intensity および /detection/contourxx/patchxx/detection_tc_positive_indices以下に追加する

    Return:
      file_path
    '''

    dataset = Whole_Slide_Bag_FP(file_path, wsi_object.getOpenSlide(), 
                                   pretrained=False, custom_transforms=False, custom_downsample=1, target_patch_size=-1,
                                   target='detection', detection_loc=True) # detection_loc=True returns detected nuclei locations
    total = len(dataset)
    slide_id = dataset.slide_id
    dataloader = torch.utils.data.DataLoader(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=False)
    conf = Config()
    if conf.save: # save ROI overlay images
        save_dir = Path(file_path).parent.parent / 'detection'
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
    else:
        save_dir = None

    attr_dict = {
                   'intensity_thres' : intensity_thres,
                   'area_thres'      : area_thres,
                   'radius'          : radius,
                 }

    f = open_hdf5_file(file_path, mode='a')

    # 閾値のattrがない場合(一回目の処理)、attrに設定する
    for attr_name, thres in attr_dict.items():
        if not attr_name in f['/detection'].attrs:
            create_hdf5_attrs(f['/detection'], attr_name, thres)

    verbose=1
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)

    for patch_id, (patch, coord, grp_name_parent, detection_loc) in enumerate(dataloader):
        verbose = 1
        if verbose > 0:
            if patch_id % ten_percent_chunk == 0:
                print('progress: {}/{} TC(+)count finished'.format(patch_id, total))

        coord = coord[0].detach().cpu().numpy()
        detection_locs = np.array(detection_loc)[:]

        # 核の検出数が0だったパッチは処理しない
        if (len(detection_locs)==1) and (np.all(detection_loc[0] == np.uint32(-1))): # 核の数が0だった場合、np.array([-1, -1], dtype=np.uint32)が入っている
            continue

        # 閾値が変更されていない場合、かつ既に結果があるパッチは飛ばす
        if (intensity_thres == f['/detection'].attrs.get('intensity_thres')) and (area_thres == f['/detection'].attrs.get('area_thres')) and (radius == f['/detection'].attrs.get('radius')):
            if ('detection_dab_intensity' in f[grp_name_parent[0]]) and ('detection_tc_positive_indices' in f[grp_name_parent[0]]):
                continue

        with torch.set_grad_enabled(False):
            img = patch.to('cpu').numpy().squeeze(0).transpose(1,2,0)
            img = (img - np.min(img)) / np.ptp(img)
            img = (img*255).astype(np.uint8)

            img_draw_volonoi = img.copy()
            img_draw_DAB = img.copy()
            org_img_copy = img.copy().astype(np.float32)

            R = org_img_copy[:,:,0] # R-ch
            G = org_img_copy[:,:,1] # G-ch
            B = org_img_copy[:,:,2] # B-ch
            BN = 255*np.divide(B, (B+G+R), out=np.zeros_like(B), where=(B+G+R)!=0) # ref.paper : Automated Selection of DAB-labeled Tissue for Immunohistochemical Quantification
            DAB = 255 - BN
            #if save_dir:
            #    cv2.imwrite(os.path.join(save_dir, str('_'.join([slide_id, str(coord[0]), str(coord[1]), 'BN.jpg']))), BN.astype(np.uint8))
            #    cv2.imwrite(os.path.join(save_dir, str('_'.join([slide_id, str(coord[0]), str(coord[1]), 'DAB.jpg']))), DAB.astype(np.uint8))

            # voronoi
            rect = (0, 0, img.shape[0], img.shape[1])
            subdiv = cv2.Subdiv2D(rect)
            for p in detection_locs:
                subdiv.insert((int(p[0]), int(p[1])))
            facets, centers = subdiv.getVoronoiFacetList([])
            #if save_dir:
            #    cv2.polylines(img_draw_volonoi, [f.astype(int) for f in facets], True, (255, 255, 255), thickness=2) # draw voronoi
            #    cv2.imwrite(os.path.join(save_dir, str('_'.join([slide_id, str(coord[0]), str(coord[1]), 'volonoi.jpg']))), img_draw_volonoi.astype(np.uint8)[:,:,::-1])
    
            # voronoi with restricted radius
            mat = np.zeros((img.shape[0], img.shape[1]), np.uint8)
            facets = [f.astype(int) for f in facets]
            detection_tc_positive_indices = []
            for i, (center, points) in enumerate(zip(centers, facets)):
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
                over_thres_area = np.count_nonzero(contour_DAB > intensity_thres)
    
                if (over_thres_area / contour_area) > area_thres: # 1-NonNucleusArea > 0.1
                    #描画
                    img_draw_DAB = cv2.drawContours(img_draw_DAB , con, -1, (255,0,0), 1) # red
                    detection_tc_positive_indices.append(i)
                else:
                    img_draw_DAB = cv2.drawContours(img_draw_DAB , con, -1, (0,180,180), 1) # cyan
    
                img_draw_volonoi = cv2.drawContours(img_draw_volonoi, con, -1, (0,255,0), 1) # draw voronoi with restricted redius

            if save_dir:
                #cv2.imwrite(os.path.join(save_dir, str('_'.join([slide_id, str(coord[0]), str(coord[1]), 'volonoi.jpg']))), img_draw_volonoi.astype(np.uint8)[:,:,::-1]) 
                cv2.imwrite(os.path.join(save_dir, str('_'.join([slide_id, str(coord[0]), str(coord[1]), 'DAB.jpg']))), img_draw_DAB.astype(np.uint8)[:,:,::-1])

            create_hdf5_dataset(f[grp_name_parent[0]], dset_name='detection_dab_intensity', data=DAB.astype(np.uint8))
            create_hdf5_dataset(f[grp_name_parent[0]], dset_name='detection_tc_positive_indices', data=np.array(detection_tc_positive_indices))

    f.close()

    return file_path
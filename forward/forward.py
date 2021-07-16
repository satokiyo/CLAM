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
    gpu: int = 1
    input_size: int = 1024
    crop_size: int = 1024
    encoder_name: str = 'se_resnext50_32x4d'
    classes: int = 1
    scale_pyramid_module: int = 1
    use_attention_branch: int = 0
    downsample_ratio: int = 1
    deep_supervision: int = 1
    use_ssl: int = 1
    save: bool = True

class SegConfig():
    device: int = 0
    gpu: int = 1
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
    detectionの結果がないパッチのみ新たにforwardを実行し、detection結果をdatasetとして/detection/contourxx/detection_loc_x or _y以下に追加する

    Return:
      file_path
    '''
    start = time.time()
    conf = Config()
    conf.input_size=patch_size
    conf.crop_size=patch_size
    if conf.gpu and torch.cuda.is_available():
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
                                               num_workers=8,
                                               pin_memory=True if conf.gpu==1 else False)

    dset_name_x = 'detection_loc_x'
    dset_name_y = 'detection_loc_y'

    with open_hdf5_file(file_path, mode='a') as f:

        verbose=1
        if verbose > 0:
            ten_percent_chunk = math.ceil(total * 0.1)

        if conf.save: # save ROI overlay images
            save_dir = Path(file_path).parent.parent / 'detection' / f.attrs.get('name')
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
        else:
            save_dir = None
 
        buf = {}
        for patch_id, (patch, coord, grp_name_parent) in enumerate(dataloader):
            if verbose > 0:
                if patch_id % ten_percent_chunk == 0:
                    print('progress: {}/{} forward finished'.format(patch_id, total))

            # 既に結果があるcontourのパッチは飛ばす(->途中でforwardを止めた場合でもcontourに対して一つでも結果があるとスキップしてしまう。その場合、.h5を削除してから再実行する必要がある)
            if (dset_name_x in f[grp_name_parent[0]]) and (dset_name_y in f[grp_name_parent[0]]):
                continue
            
            inputs = patch.to(device)
            with torch.set_grad_enabled(False):
                '''
                #TODO tmp for onnx
                if patch_id == 0:
                    import pdb;pdb.set_trace()
                    # Export the model
                    torch.onnx.export(model,               # model being run
                                      inputs,                         # model input (or a tuple for multiple inputs)
                                      "detection_model.onnx",   # where to save the model (can be a file or file-like object)
                                      export_params=True,        # store the trained parameter weights inside the model file
                                      opset_version=9,          # the ONNX version to export the model to
                                      do_constant_folding=True,  # whether to execute constant folding for optimization
                                      input_names = ['input'],   # the model's input names
                                      output_names = ['output'], # the model's output names
                                      dynamic_axes={'input' : {0 : 'batch_size',
                                                               2 : 'height',
                                                               3 : 'width'},    # variable length axes
                                                    'output' : {0 : 'batch_size',
                                                                2 : 'height',
                                                                3 : 'width'},    # variable length axes
                                                    },
                                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
                                      #operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN
                                      #operator_export_type=torch.onnx.OperatorExportTypes.RAW
                                      #operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH
                                      #operator_export_type=torch.onnx.OperatorExportTypes.ONNX
                                                    )
                '''

                if conf.deep_supervision:
                    ## onnx
                    #import onnxruntime as ort
                    #import pdb;pdb.set_trace()
                    #ort_session = ort.InferenceSession('/media/prostate/20210331_PDL1/CLAM/detection_model.onnx')
                    #outputs = ort_session.run(None, {'actual_input_1': inputs})
                    outputs, intermediates = model(inputs)
                    del intermediates
                else:
                    ## onnx
                    #outputs = ort_session.run(None, {'actual_input_1': inputs})
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

#            # save nuclei locations as dataset
#            create_hdf5_dataset(f[grp_name_parent[0]], dset_name=dset_name, data=centroids_per_patch[:, [1,0]]) # reverse x,y
            if grp_name_parent[0] not in buf.keys():
                buf.setdefault(grp_name_parent[0], [])
            buf[grp_name_parent[0]].append(centroids_per_patch[:, [1,0]]) # reverse x,y

        f.close()

    # save nuclei locations as dataset
    for i_cont, v_cont in buf.items():
        #v = np.array([(e.astype(np.int32)) for e in v])
        vx = []
        vy = []
        for i_patch, v_patch in enumerate(v_cont):
            vx.append([])
            vy.append([])
            for v_loc in v_patch:
                vx[i_patch].append(v_loc[0])
                vy[i_patch].append(v_loc[1])
            vx[i_patch] = np.array(vx[i_patch]).astype(np.int32)
            vy[i_patch] = np.array(vy[i_patch]).astype(np.int32)
#           vx = np.array([e.T[0].astype(np.int32) for e in v])
#           vy = np.array([e.T[1].astype(np.int32) for e in v])
        dt = h5py.vlen_dtype(np.dtype('int32'))
        with open_hdf5_file(file_path, mode='a') as f:
            create_hdf5_dataset(f[i_cont], dset_name=dset_name_x, data=np.array(vx), data_type=dt) # contour毎にdataset作成
            create_hdf5_dataset(f[i_cont], dset_name=dset_name_y, data=np.array(vy), data_type=dt) # contour毎にdataset作成

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
    if conf.gpu and torch.cuda.is_available():
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
                                               num_workers=8,
                                               pin_memory=True if conf.gpu==1 else False)

    dset_name = 'segmap'
    with open_hdf5_file(file_path, mode='a') as f:

        verbose=1
        if verbose > 0:
            ten_percent_chunk = math.ceil(total * 0.1)

        if conf.save: # save ROI overlay images
            save_dir = Path(file_path).parent.parent / 'segmentation' / f.attrs.get('name')
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
        else:
            save_dir = None


        buf = {}
        for patch_id, (patch, coord, grp_name_parent) in enumerate(dataloader):
            if verbose > 0:
                if patch_id % ten_percent_chunk == 0:
                    print('progress: {}/{} forward finished'.format(patch_id, total))

            # 既に結果があるcontourのパッチは飛ばす(->途中でforwardを止めた場合でもcontourに対して一つでも結果があるとスキップしてしまう。その場合、.h5を削除してから再実行する必要がある)
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

                # normalize density map values from 0 to 1, then map it to 0-255.
                vis_img = (vis_img - np.min(vis_img)) / np.ptp(vis_img)
                vis_img = (vis_img*255).astype(np.uint8)
                vis_map = np.argmax(vis_img, axis=0)
#                create_hdf5_dataset(f[grp_name_parent[0]], dset_name=dset_name, data=vis_map)
                if grp_name_parent[0] not in buf.keys():
                    buf.setdefault(grp_name_parent[0], [])
                buf[grp_name_parent[0]].append(vis_map)

                # save image
                if save_dir:
                    PALETTE = conf.palette
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
         
                    # overlay
                    overlay = np.uint8((org_img/2) + (vis_map/2))

                    # save
                    coord = coord[0].detach().cpu().numpy()
                    cv2.imwrite(os.path.join(save_dir, str('_'.join([slide_id, str(coord[0]), str(coord[1])]) + '.jpg')), overlay.astype(np.uint8)[:,:,::-1])
        f.close()


    # save segmap as dataset
    for k, v in buf.items():
        with open_hdf5_file(file_path, mode='a') as f:
            create_hdf5_dataset(f[k], dset_name=dset_name, data=np.array(v).astype(np.uint8)) # contour毎にdataset作成
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
    center = np.array(center).astype(np.int32)
    if len(center) == 0:
        print("count zero!!")
        print(f'ncenters: {len(center)}')
        centroids_wrt_orig = np.array([[-1, -1]]).astype(np.int32)
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

    file_pathからHDF5ファイルを読み込み、/detection以下の階層から各パッチの座標、detection結果を元にしたTC(+)の計算は完了しているか等の情報を参照する。
    TC(+)の結果がないパッチのみ処理し、結果をdatasetとして/detection/contourxx/patchxx/detection_dab_intensity および /detection/contourxx/patchxx/detection_tc_positive_indices以下に追加する

    Return:
      file_path
    '''

    with open_hdf5_file(file_path, mode='a') as f:

        attr_dict = {
                       'intensity_thres' : intensity_thres,
                       'area_thres'      : area_thres,
                       'radius'          : radius,
                     }
    
    
        # 閾値のattrがない場合(一回目の処理)、attrに設定する
        count = 0
        for attr_name, thres in attr_dict.items():
            if not attr_name in f['/detection'].attrs:                  # 一回目
                create_hdf5_attrs(f['/detection'], attr_name, thres)
            else:                                                       # 一回目以外
                if f['/detection'].attrs.get(attr_name) == thres:       # 既にattrがあり変更されていない
                    count+=1
                else:                                                   # modified 
                    # update attr
                    #if type(data) == str:
                    #    f['/detection'].attrs.modify(attr_name, data.encode("utf-8"))
                    #else:
                    f['/detection'].attrs.modify(attr_name, thres)
                    print(f"update attr {attr_name} to : {thres}")
     
                    #create_hdf5_attrs(f['/detection'], attr_name, thres)
        if count == len(attr_dict.keys()): # 閾値が一つも変更されていない
            skip_flag=True
        else:
            skip_flag=False
        print(f"skip flag   {skip_flag}")
    
    
        dataset = Whole_Slide_Bag_FP(file_path, wsi_object.getOpenSlide(), 
                                       pretrained=False, custom_transforms=False, custom_downsample=1, target_patch_size=-1,
                                       target='detection', detection_loc=True, skip_flag=skip_flag) # detection_loc=True returns detected nuclei locations
        total = len(dataset)
        slide_id = dataset.slide_id
        dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=8,
                                                   pin_memory=False)
    
        conf = Config()
        if conf.save: # save ROI overlay images
            save_dir = Path(file_path).parent.parent / 'detection' / f.attrs.get('name')
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
        else:
            save_dir = None
    
        verbose=1
        if verbose > 0:
            ten_percent_chunk = math.ceil(total * 0.1)
    
        #for patch_id, (patch, coord, grp_name_parent, detection_loc) in enumerate(dataloader):
        buf1 = {} # detection_dab_intensity
        buf2 = {} # detection_tc_positive_indices
        for patch_id, (data) in enumerate(dataloader):
            if verbose > 0:
                if patch_id % ten_percent_chunk == 0:
                    print('progress: {}/{} TC(+)count finished'.format(patch_id, total))
            if data:
                patch, coord, grp_name_parent, detection_loc = data
            else: # 閾値が一つも変更されていない場合、かつ既に結果があるcontourのパッチは0が返ってくる
                continue
    
    #        # 閾値が一つも変更されていない場合、かつ既に結果があるパッチは飛ばす
    #        if skip_flag > 0:
    #            if ('detection_dab_intensity' in f[grp_name_parent[0]]) and ('detection_tc_positive_indices' in f[grp_name_parent[0]]):
    #                 continue
    
            # 核の検出数が0だったパッチは処理しない
            detection_locs = detection_loc.squeeze(0).numpy()
            if (len(detection_locs)==1) and (np.all(detection_locs[0] == np.int32(-1))): # 核の数が0だった場合、np.array([-1, -1], dtype=np.int32)が入っている
                img = patch.to('cpu').numpy().squeeze(0).transpose(1,2,0)
                if grp_name_parent[0] not in buf1.keys():
                    buf1.setdefault(grp_name_parent[0], [])
                if grp_name_parent[0] not in buf2.keys():
                    buf2.setdefault(grp_name_parent[0], [])
                buf1[grp_name_parent[0]].append(np.zeros((img.shape[:2])).astype(np.uint8))
                buf2[grp_name_parent[0]].append(np.array([]))
    #            create_hdf5_dataset(f[grp_name_parent[0]], dset_name='detection_dab_intensity', data=np.zeros((img.shape[:2])).astype(np.uint8))
    #            create_hdf5_dataset(f[grp_name_parent[0]], dset_name='detection_tc_positive_indices', data=np.array([]))
                continue
    
            coord = coord[0].detach().cpu().numpy()
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
    
    #            create_hdf5_dataset(f[grp_name_parent[0]], dset_name='detection_dab_intensity', data=DAB.astype(np.uint8))
    #            create_hdf5_dataset(f[grp_name_parent[0]], dset_name='detection_tc_positive_indices', data=np.array(detection_tc_positive_indices))
                if grp_name_parent[0] not in buf1.keys():
                    buf1.setdefault(grp_name_parent[0], [])
                if grp_name_parent[0] not in buf2.keys():
                    buf2.setdefault(grp_name_parent[0], [])
                buf1[grp_name_parent[0]].append(DAB.astype(np.uint8))
                buf2[grp_name_parent[0]].append(np.array(detection_tc_positive_indices))

        f.close()
    
    # save dataset
    for (k1, v1), (k2, v2) in zip(buf1.items(), buf2.items()):
        with open_hdf5_file(file_path, mode='a') as f:
            create_hdf5_dataset(f[k1], dset_name='detection_dab_intensity', data=np.array(v1)) # contour毎にdataset作成
            v2 = np.array([e.astype(np.int32) for e in v2])
            dt = h5py.vlen_dtype(np.dtype('int32'))
            v2 = v2.astype(dt)
            create_hdf5_dataset(f[k2], dset_name='detection_tc_positive_indices', data=v2, data_type=dt) # contour毎にdataset作成

            f.close()

    return file_path
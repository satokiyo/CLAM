import h5py
import numpy as np
import os
import pdb
from wsi_core.util_classes import Mosaic_Canvas
from PIL import Image
import math
import cv2
from utils.file_utils import open_hdf5_file
import utils.utils as utils


class HDFVisitor():
    '''
    h5pyのgroup objectのvisititem()から呼ばれると、各階層毎にqueryと一致するdatasetを探索する
    '''
    def __init__(self, *queries):
        self.container = {}
        for query in queries:
            self.container[query] = []

    def __call__(self, name, obj):
        for query in self.container.keys():
            if isinstance(obj, h5py.Dataset) and (name.split("/")[-1] == query):
                #if query == 'segmap':
                #    self.container[name.split("/")[-1]].append(obj.ref) # ref
                #else:
                #    self.container[name.split("/")[-1]].append(obj[:])
                self.container[name.split("/")[-1]].append(obj[:])


def count_TC(mask, segmap, locs):
    '''
    Args:
        mask: 注目している囲み領域の輪郭(0 or 255のバイナリマスク)
        segmap: セグメンテーション結果のマスク
        locs: 検出された核の位置座標
    '''
    tc_count = 0
    tc_count_on_segmap = 0
    locs_inside_contour_segmap = []

    for loc in locs: # loop over all tc cells
        if mask[loc[0], loc[1]] > 0: # 囲みの中に含まれる点か?
            tc_count+=1
            if segmap[loc[0], loc[1]] == 2: # segmentation結果でcancer = class2の点か?
                tc_count_on_segmap+=1
                locs_inside_contour_segmap.append(loc)

    return tc_count, tc_count_on_segmap, locs_inside_contour_segmap


# slow
#    mask_region = np.where(mask==255) # 注目している囲みの領域
#    mask_region_arr = np.array([mask_region[0], mask_region[1]]).T.astype(np.int64)
#
#    cancer_region = np.where(segmap==2) # segmentation結果でcancer(=class2)の領域
#    cancer_region_arr = np.array([cancer_region[0], cancer_region[1]]).T.astype(np.int64)
#
#    # 注目している囲みの領域の中の核
#    mask_region_strarr = [f'{x[0]}_{x[1]}' for x in  mask_region_arr] # make rows as a single value to use np.intersect1d()
#    locs_strarr = [f'{x[0]}_{x[1]}' for x in  locs] # make rows as a single value to use np.intersect1d()
#    locs_inside_contour_strarr = np.intersect1d(mask_region_strarr, locs_strarr)
#    locs_inside_contour = np.array([x.split('_') for x in locs_inside_contour_strarr]).astype(np.int32)
#    tc_count = len(locs_inside_contour)
#
#    # 注目している囲みの領域の中でかつsegmentation結果のcancerの領域にある核
#    cancer_region_strarr = [f'{x[0]}_{x[1]}' for x in  cancer_region_arr] # make rows as a single value to use np.intersect1d()
#    locs_inside_contour_segmap_strarr = np.intersect1d(cancer_region_strarr, locs_inside_contour_strarr)
#    locs_inside_contour_segmap = np.array([x.split('_') for x in locs_inside_contour_segmap_strarr]).astype(np.int32)
#    tc_count_on_segmap = len(locs_inside_contour_segmap)
#
#    return tc_count, tc_count_on_segmap, locs_inside_contour_segmap



def isWhitePatch(patch, satThresh=5):
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    return True if np.mean(patch_hsv[:,:,1]) < satThresh else False

def isBlackPatch(patch, rgbThresh=40):
    return True if np.all(np.mean(patch, axis = (0,1)) < rgbThresh) else False

def isBlackPatch_S(patch, rgbThresh=20, percentage=0.05):
    num_pixels = patch.size[0] * patch.size[1]
    return True if np.all(np.array(patch) < rgbThresh, axis=(2)).sum() > num_pixels * percentage else False

def isWhitePatch_S(patch, rgbThresh=220, percentage=0.2):
    num_pixels = patch.size[0] * patch.size[1]
    return True if np.all(np.array(patch) > rgbThresh, axis=(2)).sum() > num_pixels * percentage else False

def coord_generator(x_start, x_end, x_step, y_start, y_end, y_step, args_dict=None):
    for x in range(x_start, x_end, x_step):
        for y in range(y_start, y_end, y_step):
            if args_dict is not None:
                process_dict = args_dict.copy()
                process_dict.update({'pt':(x,y)})
                yield process_dict
            else:
                yield (x,y)

def savePatchIter_bag_hdf5(patch):
    x, y, cont_idx, patch_level, downsample, downsampled_level_dim, level_dim, img_patch, name, save_path= tuple(patch.values())
    img_patch = np.array(img_patch)[np.newaxis,...]
    img_shape = img_patch.shape

    file_path = os.path.join(save_path, name)+'.h5'
    file = h5py.File(file_path, "a")

    dset = file['imgs']
    dset.resize(len(dset) + img_shape[0], axis=0)
    dset[-img_shape[0]:] = img_patch

    if 'coords' in file:
        coord_dset = file['coords']
        coord_dset.resize(len(coord_dset) + img_shape[0], axis=0)
        coord_dset[-img_shape[0]:] = (x,y)

    file.close()

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

def initialize_hdf5_bag(first_patch, save_coord=False):
    x, y, cont_idx, patch_level, downsample, downsampled_level_dim, level_dim, img_patch, name, save_path = tuple(first_patch.values())
    file_path = os.path.join(save_path, name)+'.h5'
    file = h5py.File(file_path, "w")
    img_patch = np.array(img_patch)[np.newaxis,...]
    dtype = img_patch.dtype

    # Initialize a resizable dataset to hold the output
    img_shape = img_patch.shape
    maxshape = (None,) + img_shape[1:] #maximum dimensions up to which dataset maybe resized (None means unlimited)
    dset = file.create_dataset('imgs', 
                                shape=img_shape, maxshape=maxshape,  chunks=img_shape, dtype=dtype)

    dset[:] = img_patch
    dset.attrs['patch_level'] = patch_level
    dset.attrs['wsi_name'] = name
    dset.attrs['downsample'] = downsample
    dset.attrs['level_dim'] = level_dim
    dset.attrs['downsampled_level_dim'] = downsampled_level_dim

    if save_coord:
        coord_dset = file.create_dataset('coords', shape=(1, 2), maxshape=(None, 2), chunks=(1, 2), dtype=np.int32)
        coord_dset[:] = (x,y)

    file.close()
    return file_path

def sample_indices(scores, k, start=0.48, end=0.52, convert_to_percentile=False, seed=1):
    np.random.seed(seed)
    if convert_to_percentile:
        end_value = np.quantile(scores, end)
        start_value = np.quantile(scores, start)
    else:
        end_value = end
        start_value = start
    score_window = np.logical_and(scores >= start_value, scores <= end_value)
    indices = np.where(score_window)[0]
    if len(indices) < 1:
        return -1 
    else:
        return np.random.choice(indices, min(k, len(indices)), replace=False)

def top_k(scores, k, invert=False):
    if invert:
        top_k_ids=scores.argsort()[:k]
    else:
        top_k_ids=scores.argsort()[::-1][:k]
    return top_k_ids

def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average')/len(scores) * 100   
    return scores

def screen_coords(scores, coords, top_left, bot_right):
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
    scores = scores[mask]
    coords = coords[mask]
    return scores, coords

def sample_rois(scores, coords, k=5, mode='range_sample', seed=1, score_start=0.45, score_end=0.55, top_left=None, bot_right=None):

    if len(scores.shape) == 2:
        scores = scores.flatten()

    scores = to_percentiles(scores)
    if top_left is not None and bot_right is not None:
        scores, coords = screen_coords(scores, coords, top_left, bot_right)

    if mode == 'range_sample':
        sampled_ids = sample_indices(scores, start=score_start, end=score_end, k=k, convert_to_percentile=False, seed=seed)
    elif mode == 'topk':
        sampled_ids = top_k(scores, k, invert=False)
    elif mode == 'reverse_topk':
        sampled_ids = top_k(scores, k, invert=True)
    else:
        raise NotImplementedError
    coords = coords[sampled_ids]
    scores = scores[sampled_ids]

    asset = {'sampled_coords': coords, 'sampled_scores': scores}
    return asset

def DrawGrid(img, coord, shape, thickness=1, color=(0,0,0,128)):
    cv2.rectangle(img, tuple(np.maximum([0, 0], coord-thickness//2)), tuple(coord - thickness//2 + np.array(shape)), color, thickness=thickness)
    return img

def DrawMap(canvas, patch_dset, coords, patch_size, indices=None, verbose=1, draw_grid=True, file=None):
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)
        #print('start stitching {}'.format(patch_dset.attrs['wsi_name']))
    
    for idx in range(total):
        if verbose > 0:
            if idx % ten_percent_chunk == 0:
                print('progress: {}/{} stitched'.format(idx, total))
        
        patch_id = indices[idx]
        patch_ref = patch_dset[patch_id]
        patch = file[patch_ref]
        #patch = cv2.resize(patch, patch_size)
        coord = coords[patch_id]
        canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3].shape[:2]
        canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size)

    return Image.fromarray(canvas)


#TODO slow
def DrawMapGray(canvas, patch_dset, coords, patch_size, step_size, indices=None, verbose=1):
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)
        #print('start stitching {}'.format(patch_dset.attrs['wsi_name']))
    
    for idx in range(total):
        if verbose > 0:
            if idx % ten_percent_chunk == 0:
                print('progress: {}/{} stitched segmap'.format(idx, total))
        
        patch_id = indices[idx]
        patch = patch_dset[patch_id]
        #patch = file[patch_ref]
        coord = coords[patch_id]
        canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].shape[:2]
        if patch_size > step_size: # overlap tile
            offset = (int((patch_size[0] - step_size[0]) // 2), int((patch_size[0] - step_size[0]) // 2))
            canvas_crop_shape = canvas[coord[1]:coord[1]+step_size[1], coord[0]:coord[0]+step_size[0]].shape[:2]
            canvas[coord[1]+offset[1]:coord[1]+offset[1]+step_size[1], coord[0]+offset[0]:coord[0]+offset[0]+step_size[0]] = patch[offset[1]:offset[1]+canvas_crop_shape[0], offset[0]:offset[0]+canvas_crop_shape[1]]
        else:
            canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1]]

    return canvas


def DrawMapFromCoords(canvas, wsi_object, coords, patch_size, vis_level, indices=None, verbose=1, draw_grid=True):
    '''
    Args:
      coords : top-left (x,y) coordinates at level 0
      patch_size : patch size at level 0
      vis_level : disired vis level
    '''
    downsamples = wsi_object.wsi.level_downsamples[vis_level]
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)
        
    patch_size = tuple(np.ceil((np.array(patch_size)/np.array(downsamples))).astype(np.int32)) # convert patch_size from level 0 to vis_level.
    print('downscaled patch size: {}x{}'.format(patch_size[0], patch_size[1]))
    
    canvas_size = canvas.shape[:2]
    canvas[:,:,:] = np.array(wsi_object.wsi.read_region((0,0), vis_level, (canvas_size[1], canvas_size[0])).convert("RGB")) # coord is the location (x, y) tuple giving the top left pixel in the level 0 reference frame
    for patch_id in range(total):
        if verbose > 0:
            if patch_id % ten_percent_chunk == 0:
                print('progress: {}/{} stitched'.format(patch_id, total))
        
        coord = coords[patch_id] # coord at level0
        #patch = np.array(wsi_object.wsi.read_region(tuple(coord), vis_level, patch_size).convert("RGB")) # coord is the location (x, y) tuple giving the top left pixel in the level 0 reference frame
        coord = np.ceil(coord / downsamples).astype(np.int32) # convert coord for vis_level
        #canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3].shape[:2]
        #canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size)

    return canvas


def StitchPatches(hdf5_file_path, downscale=16, draw_grid=False, bg_color=(0,0,0), alpha=-1):
    file = h5py.File(hdf5_file_path, 'r')
    dset = file['imgs']
    coords = file['coords'][:]
    if 'downsampled_level_dim' in dset.attrs.keys(): # patch level. Not level 0
        w, h = dset.attrs['downsampled_level_dim'] # patch size at patch level. Not at level 0
    else:
        w, h = dset.attrs['level_dim']
    print('original size: {} x {}'.format(w, h))
    w = w // downscale
    h = h //downscale
    coords = (coords / downscale).astype(np.int32)
    print('downscaled size for stiching: {} x {}'.format(w, h))
    print('number of patches: {}'.format(len(dset)))
    img_shape = dset[0].shape
    print('patch shape: {}'.format(img_shape))
    downscaled_shape = (img_shape[1] // downscale, img_shape[0] // downscale)

    if w*h > Image.MAX_IMAGE_PIXELS: 
        raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)
    
    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w,h), mode="RGB", color=bg_color)
    else:
        heatmap = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
    
    heatmap = np.array(heatmap)
    heatmap = DrawMap(heatmap, dset, coords, downscaled_shape, indices=None, draw_grid=draw_grid)
    
    file.close()
    return heatmap

def StitchCoords(hdf5_file_path, wsi_object, save_dir, downscale=16, draw_grid=False, draw_contour=False, bg_color=(0,0,0), alpha=-1,
                   overlaymap=None, all_locs=[], tc_positive_locs=[]): 
    '''
    パッチ座標をもとにパッチを切り出したヒートマップを作成する
    Args:
        overlaymap : segmentation heatmap of size at vis_level
        all_locs  : coordinates of detected nuclei location at vis_level
        tc_positive_locs  : coordinates of detected DAB(+) nuclei location at vis_level
    '''
    # hdf5の読み込み
    file = open_hdf5_file(hdf5_file_path, mode='r')

    wsi = wsi_object.getOpenSlide()
    vis_level = wsi.get_best_level_for_downsample(downscale)
    w, h = wsi.level_dimensions[0] # image size at level0

    print('start stitching {}'.format(file.attrs['name']))
    print('original size: {} x {}'.format(w, h))
    w, h = wsi.level_dimensions[vis_level] # image size at 'heatmap level' for stitching. (Not level0 nor patch level)
    print('downscaled size for stiching: {} x {}'.format(w, h))

    targets = ['detection', 'segmentation']
    for target in targets:
        grp_target = file[target]
        patch_size = grp_target.attrs['patch_size']
        patch_level = grp_target.attrs['patch_level']
        print('patch size: {}x{} patch level: {}'.format(patch_size, patch_size, patch_level)) # patch levelでのpatch size
        patch_size = tuple((np.array((patch_size, patch_size)) * wsi.level_downsamples[patch_level]).astype(np.int32)) # level0でのpatch size
        print('ref patch size: {}x{}'.format(patch_size, patch_size))

        coords_all_patch = [] # 全てのcontourのパッチ座標保存用
        coords_all_contour = {} # 全てのcontourの輪郭座標保存用

        # contour毎に処理する
        for i, cont in enumerate(sorted(grp_target)): # sortedしないと順番がおかしくなる
#            grp_cont = file[f'{target}/{cont}']

            # dataset読み込み。以下の階層にdatasetがある。
            # /segmentation/contourxx/coords_patches
#            queries = ['coords_patches']
#            v = HDFVisitor(*queries)
#            grp_cont.visititems(v)
#            coords_all_patch.extend(v.container['coords_patches'])
            coords_contour = file[f'{target}/{cont}/coords_contour']
            coords_all_contour.setdefault(i, coords_contour)

        queries = ['coords_patches']
        v = HDFVisitor(*queries)
    #        grp_cont.visititems(v)
        grp_target.visititems(v)
        if not v.container['coords_patches']:
            print(f'There is no patch generated for {target}.')
            print('exit.')
            import sys
            sys.exit()

        coords_all_patch = np.vstack(v.container['coords_patches'])
 

        # 全てのcontoursに対して処理する
        all_coords_patch = np.unique(coords_all_patch, axis=0) # 重複するpatchは除く
        print(f'start stitching all contours of {target}')
        print('number of patches: {}'.format(len(all_coords_patch)))



        if w*h > Image.MAX_IMAGE_PIXELS: 
            raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)
        
        if alpha < 0 or alpha == -1:
            heatmap = Image.new(size=(w,h), mode="RGB", color=bg_color)
        else:
            heatmap = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
    
        heatmap = np.array(heatmap)
    
        DrawMapFromCoords(heatmap, wsi_object, all_coords_patch, patch_size, vis_level, indices=None, draw_grid=draw_grid)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    
        if draw_contour:
            color = (0,255,0)
            hole_color = (255,0,0)
            line_thickness=200
            number_contours = True
            downsample = wsi_object.level_downsamples[vis_level] # image size at 'heatmap level' for stitching. (Not level0 nor patch level)
            scale = [1/downsample[0], 1/downsample[1]]
            top_left = (0,0)
            offset = tuple(-(np.array(top_left) * scale).astype(int))
            line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))


            for i, coords_contour in coords_all_contour.items():
                if not number_contours:
                    cv2.drawContours(heatmap, wsi_object.scaleContourDim(coords_contour, scale), 
                                     -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)
        
                else: # add numbering to each contour
                    contour = np.array(wsi_object.scaleContourDim(coords_contour, scale))
                    M = cv2.moments(contour)
                    cX = int(M["m10"] / (M["m00"] + 1e-9))
                    cY = int(M["m01"] / (M["m00"] + 1e-9))
                    # draw the contour and put text next to center
                    cv2.drawContours(heatmap,  [contour], -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)
                    cv2.putText(heatmap, "{}".format(i), (cX, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
            for holes in wsi_object.holes_tissue:
                cv2.drawContours(heatmap, wsi_object.scaleContourDim(holes, scale), 
                                 -1, hole_color, line_thickness, lineType=cv2.LINE_8)

        save_path = os.path.join(save_dir, f'{wsi_object.name}_{target}.jpg')
        cv2.imwrite(save_path, heatmap)

        if target == 'segmentation':
            if overlaymap is not None:
                if not isinstance(overlaymap, Image.Image):
                    overlay = Image.fromarray(overlaymap)
                print('start overlay segmap')
                PALETTE = [0,0,0,
                           0,128,0,
                           128,0,0,
                           0,0,128]
                overlaymap_p = overlay.convert("P")
                overlaymap_p.putpalette(PALETTE)
                overlaymap_rgb = np.array(overlaymap_p.convert('RGB')).astype(np.uint8)
                overlaymap_bgr = overlaymap_rgb[:,:,::-1]
                mask = np.zeros_like(overlaymap_bgr).astype(np.uint8)
                mask[np.any(overlaymap_bgr>0, axis=2)] = 255
                heatmap = heatmap.astype(np.uint8)
                heatmap_inner = np.bitwise_and(heatmap, mask)
                heatmap_outer = np.bitwise_and(heatmap, (255 - mask))
                heatmap = heatmap_outer + (heatmap_inner / 2) + (overlaymap_bgr / 2)
                print('end')
            if all_locs:
                print('start overlay loc')
                #heatmap = utils.paint_circles(img=np.array(heatmap), points=np.vstack(all_locs), color='cyan', crosshair=True, markerSize=0) # slow
                ctr = np.array(all_locs).reshape((-1,1,2)).astype(np.int32)[:,:,[1,0]] # reverse xy
                heatmap = cv2.drawContours(np.array(heatmap), ctr, -1, (255,255,0), -10, 8) # index=-1:all contours
                print('end')
            if tc_positive_locs:
                print('start overlay loc tc(+)')
                #heatmap = utils.paint_circles(img=np.array(heatmap), points=np.vstack(tc_positive_locs), color='pink', crosshair=True, markerSize=0) # slow
                ctr = np.array(tc_positive_locs).reshape((-1,1,2)).astype(np.int32)[:,:,[1,0]] # reverse xy
                heatmap = cv2.drawContours(np.array(heatmap), ctr, -1, (0,255,255), -10, 8) # index=-1:all contours
                print('end')
 
            # save
            save_path = os.path.join(save_dir, f'{wsi_object.name}_{target}_overlay.jpg')
            cv2.imwrite(save_path, heatmap)

    file.close()



def SamplePatches(coords_file_path, save_file_path, wsi_object, 
    patch_level=0, custom_downsample=1, patch_size=256, sample_num=100, seed=1, stitch=True, verbose=1, mode='w'):
    file = h5py.File(coords_file_path, 'r')
    dset = file['coords']
    coords = dset[:]

    h5_patch_size = dset.attrs['patch_size']
    h5_patch_level = dset.attrs['patch_level']
    
    if verbose>0:
        print('in .h5 file: total number of patches: {}'.format(len(coords)))
        print('in .h5 file: patch size: {}x{} patch level: {}'.format(h5_patch_size, h5_patch_size, h5_patch_level))

    if patch_level < 0:
        patch_level = h5_patch_level

    if patch_size < 0:
        patch_size = h5_patch_size

    np.random.seed(seed)
    indices = np.random.choice(np.arange(len(coords)), min(len(coords), sample_num), replace=False)

    target_patch_size = np.array([patch_size, patch_size])
    
    if custom_downsample > 1:
        target_patch_size = (np.array([patch_size, patch_size]) / custom_downsample).astype(np.int32)
        
    if stitch:
        canvas = Mosaic_Canvas(patch_size=target_patch_size[0], n=sample_num, downscale=4, n_per_row=10, bg_color=(0,0,0), alpha=-1)
    else:
        canvas = None
    
    for idx in indices:
        coord = coords[idx]
        patch = wsi_object.wsi.read_region(coord, patch_level, tuple([patch_size, patch_size])).convert('RGB')
        if custom_downsample > 1:
            patch = patch.resize(tuple(target_patch_size))

        # if isBlackPatch_S(patch, rgbThresh=20, percentage=0.05) or isWhitePatch_S(patch, rgbThresh=220, percentage=0.25):
        #     continue

        if stitch:
            canvas.paste_patch(patch)

        asset_dict = {'imgs': np.array(patch)[np.newaxis,...], 'coords': coord}
        save_hdf5(save_file_path, asset_dict, mode=mode)
        mode='a'

    return canvas, len(coords), len(indices)



def calculate_TPS(file_path, wsi_object):
    '''
    パッチ座標をもとにパッチを切り出したヒートマップを作成する
    '''
    wsi = wsi_object.getOpenSlide()

    # hdf5の読み込み
    file = open_hdf5_file(file_path, mode='r')

    print('start calculating TPS {}'.format(file.attrs['name']))

    #-----------------------------------------------------------#
    # segmentationの結果をindex color画像として取得する #
    #-----------------------------------------------------------#
    grp_target = file['segmentation']

    # 全contourの情報を取得
    coords_all_patch = [] # 全てのcontourのパッチ座標保存用
    segmap_all_patch = [] # 全てのcontourのsegmap保存用

    #TODO
    #datasetとしてcontour毎に計算済みTPSを記憶しておく。もし全てのcontourが計算済みのdatasetを持っていたら、summaryも含めて前回の値を返すだけ。
    #もし計算済みのdatasetを持っていないcontourがあったら、そのcontourだけTPSを計算し、後のcontourは値を使いまわす。ただしsummaryは再計算して更新する必要がある。
    # -> もし一つでも変わっていたらsummary再計算のためにマップを準備しないといけない。なので時間短縮効果は、一つも変化ない場合のみ。なのであまり意味がない可能性が高い

#    for i, cont in enumerate(sorted(grp_target)): # sortedしないと順番がおかしくなる
#        grp_cont = file[f'segmentation/{cont}']

        # dataset読み込み。以下の階層にdatasetがある。
        # /segmentation/contourxx/coords_patches
        # /segmentation/contourxx/segmap
    queries = ['coords_patches', 'segmap']
    v = HDFVisitor(*queries)
#        grp_cont.visititems(v)
    grp_target.visititems(v)
    if not v.container['coords_patches']:
        print('There is no patch generated for segmentation.')
        print('exit.')
        import sys
        sys.exit()

    coords_all_patch = np.vstack(v.container['coords_patches'])
    segmap_all_patch = np.vstack(v.container['segmap'])

    target_level_seg = file['segmentation'].attrs.get('patch_level')
    step_size = file['segmentation'].attrs.get('step_size')
    step_size_seg = (step_size, step_size)
    coords = coords_all_patch # coord at level0
    downsamples = wsi.level_downsamples[target_level_seg]
    coords_seg_level = np.ceil(np.array(coords) / downsamples).astype(np.int32) # convert coord for target level
    patch_size_seg = segmap_all_patch[0].shape[:2]
    dset = segmap_all_patch

    w, h = wsi.level_dimensions[target_level_seg] # image size at 'heatmap level' for calculating TPS. (Not level0 nor patch level)
    canvas = np.zeros((h,w))
    print('downscaled size for calculating TPS: {} x {}'.format(w, h))

    # 全パッチ分のsegmentationの結果を一枚にまとめる
    segmap = DrawMapGray(canvas, dset, coords_seg_level, patch_size_seg, step_size=step_size_seg, indices=None)
#    segmap = np.array(segmap)

    #------------------------------------------------------------------------#
    # detectionの結果の座標を読み込んでsegmentationのlevelでの座標に変換する #
    #------------------------------------------------------------------------#
    grp_target = file['detection']

    # 全contourの情報を取得
#    coords_all_patch = [] # 全てのcontourのパッチ座標保存用
#    detection_loc_all_patch = [] # 全てのcontourのdetection_loc保存用
#    detection_indices_all_patch = [] # 全てのcontourのdetection_indices保存用

#    for i, cont in enumerate(sorted(grp_target)): # sortedしないと順番がおかしくなる
#        grp_cont = file[f'detection/{cont}']

        # dataset読み込み。以下の階層にdatasetがある。
        # /detection/contourxx/coords_patches
        # /detection/contourxx/detection_loc_x
        # /detection/contourxx/detection_loc_y
        # /detection/contourxx/detection_tc_positive_indices
    queries = ['coords_patches', 'detection_loc_x', 'detection_loc_y', 'detection_tc_positive_indices']
    v = HDFVisitor(*queries)
    grp_target.visititems(v)
    if not v.container['coords_patches']:
        print('There is no patch generated for detection.')
        print('exit.')
        import sys
        sys.exit()
        #grp_cont.visititems(v)
    coords_all_patch = np.vstack(v.container['coords_patches'])
#    detection_loc = np.array([np.array(patch_x, patch_y).T for cont_x, cont_y in zip(v.container['detection_loc_x'], v.container['detection_loc_y']) for patch_x, patch_y in zip(cont_x, cont_y) ])
    #tmp = np.array([(patch_x, patch_y) for cont_x, cont_y in zip(v.container['detection_loc_x'], v.container['detection_loc_y']) for patch_x, patch_y in zip(cont_x, cont_y)]) # tuple of xy, by contour to by patch 
    buf = []
    for cont_id, (cont_x, cont_y) in enumerate(zip(v.container['detection_loc_x'], v.container['detection_loc_y'])):
        buf.append([])
        for patch_x, patch_y in zip(cont_x, cont_y): # tuple of xy, by contour to by patch 
            buf[cont_id].append((patch_x, patch_y))
    tmp1 = np.vstack(buf)
    buf = []
#    tmp2 = np.array([(x,y) for i, (patch_x, patch_y) in enumerate(tmp) for x, y in zip(patch_x, patch_y)])
    for patch_id, (patch_tuple) in enumerate(tmp1):
        patch_x, patch_y = patch_tuple
        buf.append([])
        for x, y in zip(patch_x, patch_y):
            buf[patch_id].append((x,y))
        buf[patch_id] = np.array(buf[patch_id])
    tmp2 = np.array(buf)

    #detection_loc_all_patch.extend(detection_loc)
    detection_loc_all_patch = tmp2
#    detection_indices_all_patch.extend(v.container['detection_tc_positive_indices'])
    buf = []
    for cont_id, (tc_ind) in enumerate(v.container['detection_tc_positive_indices']):
#        buf.append(tc_ind)
        for patch_ind in tc_ind: # by contour to by patch 
            #buf[cont_id].append(patch_ind)
            buf.append(patch_ind)
    tmp1 = np.array(buf)
#    for patch_id, (patch_tuple) in enumerate(tmp1):
#        patch_x, patch_y = patch_tuple
#        buf.append([])
#        for x, y in zip(patch_x, patch_y):
#            buf[patch_id].append((x,y))
#        buf[patch_id] = np.array(buf[patch_id])
#    tmp2 = np.array(buf)
    detection_indices_all_patch = tmp1


 

    target_level_detection = file['detection'].attrs.get('patch_level')
    patch_downsample = int(wsi.level_downsamples[target_level_detection])
    patch_downsample_seg = int(wsi.level_downsamples[target_level_seg])

    detection_loc_all_patch_ref_level = []
    detection_indices_all_patch_notempty = []
    for coord, detection_loc, detection_ind in zip(coords_all_patch, detection_loc_all_patch, detection_indices_all_patch):
        if np.all(detection_loc == np.int32(-1)): # 核の数が0だった場合、np.array([-1, -1], dtype=np.int32)が入っている
            continue
        detection_loc = coord + detection_loc * patch_downsample  # level0での絶対座標に変換. coordはlevel0でのtopleft
        detection_loc_all_patch_ref_level.append(detection_loc)
        detection_indices_all_patch_notempty.append(detection_ind)

    # seg levelでの座標に変換
    detection_loc_all_patch_seg_level = [(loc/patch_downsample_seg).astype(np.int32) for loc in detection_loc_all_patch_ref_level]


    #------------------------------------------#
    # contour毎にループしてTPSを算出 @seglevel #
    #------------------------------------------#
    all_locs = detection_loc_all_patch_seg_level
    tc_positive_locs = [all_locs[i_patch][detection_indices_all_patch_notempty[i_patch][:].tolist()] for i_patch in range(len(all_locs))]
    all_locs = np.vstack(all_locs).astype(np.int32)[:,[1,0]] # reverse y,x
    tc_positive_locs = np.vstack(tc_positive_locs).astype(np.int32)[:,[1,0]] # reverse y,x
#    all_locs = np.unique(np.vstack(all_locs), axis=0).astype(np.int32)[:,[1,0]] # reverse y,x
#    tc_positive_locs = np.unique(np.vstack(tc_positive_locs), axis=0).astype(np.int32)[:,[1,0]] # reverse y,x
    #print(f"all_locs {len(all_locs)}")
    #print(f"tc_positive_locs {len(tc_positive_locs)}")

    grp_target = file['segmentation']

    coords_all_contour = [] # 全てのcontourの輪郭座標保存用
    w, h = wsi.level_dimensions[target_level_seg]
    mask_all = np.zeros((h,w), dtype=np.int32)

    # contour毎に処理する
    scale = [1/patch_downsample_seg, 1/patch_downsample_seg]
    for i, cont in enumerate(sorted(grp_target)): # sortedしないと順番がおかしくなる
        coords_contour = file[f'segmentation/{cont}/coords_contour']
        coords_all_contour.append(coords_contour[:])

        cont_mask = np.zeros((h,w), dtype=np.float32)
        cont_mask = cv2.fillPoly(cont_mask, wsi_object.scaleContourDim([coords_contour[:]], scale), (1)).astype(np.uint8) # binary mask of a contour
        mask_all += cont_mask

        # count tc
        tc_count, tc_count_on_segmap, _ = count_TC(cont_mask, segmap, all_locs)
        print(f'TC count contour{i}              : {tc_count}')
        print(f'TC count contour{i} with segmap  : {tc_count_on_segmap}')

        # count tc(+)
        tc_pos_count, tc_pos_count_on_segmap, _ = count_TC(cont_mask, segmap, tc_positive_locs)
        print(f'TC(+) count contour{i}             : {tc_pos_count}')
        print(f'TC(+) count contour{i}with segmap  : {tc_pos_count_on_segmap}')

        # TPS
        print(f'TPS contour{i}           : {tc_pos_count/(tc_count+1e-6)}')
        print(f'TPS cont{i} with segmap  : {tc_pos_count_on_segmap/(tc_count_on_segmap+1e-6)}')

    # 全てのcontoursに対して処理する
    # count summary tc
    tc_count, tc_count_on_segmap, locs_inside_contour_segmap = count_TC(mask_all, segmap, all_locs)
    print(f'TC count summary               : {tc_count}')
    print(f'TC count summary with segmap   : {tc_count_on_segmap}')

    # count summary tc(+)
    tc_pos_count, tc_pos_count_on_segmap, locs_pos_inside_contour_segmap = count_TC(mask_all, segmap, tc_positive_locs)
    print(f'TC(+) count summary              : {tc_pos_count}')
    print(f'TC(+) count summary with segmap  : {tc_pos_count_on_segmap}')

    # TPS summary
    print(f'TPS summary              : {tc_pos_count/(tc_count+1e-6)}')
    print(f'TPS summary with segmap  : {tc_pos_count_on_segmap/(tc_count_on_segmap+1e-6)}')


    # segmapを囲み領域に限定する
    mask_all[mask_all>0] = 1 #TODO slow
#    mask_all = (mask_all*255).astype(np.uint8) # binary mask 0 or 255
#    mask_all = (mask_all/255).astype(np.uint8) # binary mask 0 or 1
    #mask_all = np.where((mask_all != 0), 1, 0).astype(np.uint8) # binary mask 0 or 1
    segmap = mask_all * segmap

    del mask_all

    file.close()

    return segmap, target_level_seg, locs_inside_contour_segmap, locs_pos_inside_contour_segmap
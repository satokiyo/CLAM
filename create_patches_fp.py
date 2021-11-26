# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords, StitchPatches, calculate_TPS
from wsi_core.batch_process_utils import initialize_df
from forward.forward import forward_detection, detect_tc_positive_nuclei, forward_segmentation
# other imports
import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd
from PIL import Image
import cv2
import gc
from pathlib import Path
#from datetime import datetime
from logging import getLogger, DEBUG, INFO
from utils.logger_setting import set_logger

logger = getLogger('pdl1_module')

def seg_and_patch(source_slides, source_annotations, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, 
                  patch_size = 256, step_size = 256, 
                  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'},
                  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}, 
                  vis_params = {'vis_level': -1, 'line_thickness': 500},
                  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
                  patch_resolution_detection = 40,
                  patch_resolution_segmentation = 10,
                  use_default_params = False, 
                  seg = False, save_mask = True, 
                  stitch= False, 
                  patch = False, auto_skip=True, segmentation=False, detection=False, process_list = None,
                  model_path_detection='', model_path_segmentation='',
                  intensity_thres=175, area_thres=0.1, radius=25):

    slides = sorted(os.listdir(source_slides))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(source_slides, slide)) and slide.split('.')[-1] in ['ndpi', 'svs']]
    xmls = sorted(os.listdir(source_annotations))
    target_names = [xml.split('.')[0] for xml in xmls if os.path.isfile(os.path.join(source_annotations, xml)) and xml.split('.')[-1] in ['ndpa', 'xml']]
    tmp = []
    for slide in slides[:]:
        if slide.split('.')[0] in target_names:
            tmp.append(slide)
    slides = tmp
    logger.debug(slides)

    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

    mask = df['process'] == 1
    process_stack = df[mask]

    total = len(process_stack)

    seg_times = 0.
    patch_times = 0.
    stitch_times = 0.

    for i in range(total):
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        logger.debug("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
        logger.debug('processing {}'.format(slide))
        
        df.loc[idx, 'process'] = 0
        slide_id, _ = os.path.splitext(slide)

        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            logger.debug('{} already exist in destination location, skipped'.format(slide_id))
            df.loc[idx, 'status'] = 'already_exist'
            continue

        # Inialize WSI
        full_path = os.path.join(source_slides, slide)
        WSI_object = WholeSlideImage(full_path)

        # patch level(openslide mag_idx)の決定
        obj_pow = int(WSI_object.objective_power)
        if (obj_pow < patch_resolution_detection) or (obj_pow < patch_resolution_segmentation):
            logger.debug("Desired patch resolution is higher than wsi's highest available resolution i.e. objective power")
            logger.debug("skip slide ...")
            continue
        downscale_detection = int(obj_pow / patch_resolution_detection)
        downscale_segmentation = int(obj_pow / patch_resolution_segmentation)
        patch_level_detection = WSI_object.getOpenSlide().get_best_level_for_downsample(downscale_detection)
        patch_level_segmentation = WSI_object.getOpenSlide().get_best_level_for_downsample(downscale_segmentation)
        assert patch_level_detection in [0, 1, 2, 3]
        assert patch_level_segmentation in [0, 1, 2, 3]

        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()
        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}
            current_patch_params = {}

            for key in vis_params.keys():
                current_vis_params.update({key: df.loc[idx, key]})

            for key in filter_params.keys():
                current_filter_params.update({key: df.loc[idx, key]})

            for key in seg_params.keys():
                current_seg_params.update({key: df.loc[idx, key]})

            for key in patch_params.keys():
                current_patch_params.update({key: df.loc[idx, key]})

        if current_vis_params['vis_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_vis_params['vis_level'] = 0
            else:    
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_vis_params['vis_level'] = best_level
                del wsi

        if current_seg_params['seg_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_seg_params['seg_level'] = 0
            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_seg_params['seg_level'] = best_level
                del wsi

        keep_ids = str(current_seg_params['keep_ids'])
        if keep_ids != 'none' and len(keep_ids) > 0:
            str_ids = current_seg_params['keep_ids']
            current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['keep_ids'] = []

        exclude_ids = str(current_seg_params['exclude_ids'])
        if exclude_ids != 'none' and len(exclude_ids) > 0:
            str_ids = current_seg_params['exclude_ids']
            current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['exclude_ids'] = []

        w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
        if w * h > 1e8:
            logger.debug('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
            df.loc[idx, 'status'] = 'failed_seg'
            continue

        df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
        df.loc[idx, 'seg_level'] = current_seg_params['seg_level']


        # start segment tissue and background.
        seg_time_elapsed = -1
        if seg:
            start_time = time.time()

            # Segment
            WSI_object.segmentTissue(**current_seg_params, filter_params=current_filter_params)

            seg_time_elapsed = time.time() - start_time   
            logger.debug("segmenttissue took  {:>10.5f} seconds".format(seg_time_elapsed))

        # save segmented tissue mask(contour line)
        if save_mask:
            mask = WSI_object.visWSI(**current_vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
            cv2.imwrite(mask_path, mask)
            del mask

        # save patch coordinates of each contour as .h5 file
        patch_time_elapsed = -1 # Default time
        if patch:
            start_time = time.time()

            patch_levels = {"detection" : patch_level_detection, "segmentation" : patch_level_segmentation}
            current_patch_params.update({'patch_level': patch_levels, 'patch_size': patch_size, 'step_size': step_size, 
                                         'save_path': patch_save_dir})
            # Patch and save in WSI_object.hdf5_file
            WSI_object.process_contours(**current_patch_params)
            
            patch_time_elapsed = time.time() - start_time
            logger.debug("patching took      {:>10.5f} seconds".format(patch_time_elapsed))

        # forward segmentation model.
        segmentation_time_elapsed = -1
        segmentation=True
        if segmentation:
            file_path = os.path.join(patch_save_dir, slide_id+'.h5') # add new attr to patched .h5 file.
            if os.path.isfile(file_path):
                start = time.time()

                # segmentation
                file_path = forward_segmentation(file_path, WSI_object, patch_size, model_path=model_path_segmentation) # forward using trained model and save info to .h5 file.

                segmentation_time_elapsed = time.time() - start
                logger.debug("segmentation took  {:>10.5f} seconds".format(segmentation_time_elapsed))


        # forward detection model.
        detection_time_elapsed = -1
        detection=True
        if detection:
            file_path = os.path.join(patch_save_dir, slide_id+'.h5') # add new attr to patched .h5 file.
            if os.path.isfile(file_path):
                start = time.time()

                # detect nuclei
#                file_path = forward_detection(file_path, WSI_object, patch_size, model_path=model_path_detection) # forward using trained model and save info to .h5 file.

                # detect TC(+)
                file_path = detect_tc_positive_nuclei(file_path, WSI_object, intensity_thres=intensity_thres, area_thres=area_thres, radius=radius, num_worker=12) 

                detection_time_elapsed = time.time() - start
                logger.debug("detection took     {:>10.5f} seconds".format(detection_time_elapsed))




        # TPS算出
        calc_tps_time_elapsed = -1
        if True:
            file_path = os.path.join(patch_save_dir, slide_id+'.h5') # add new attr to patched .h5 file.
            if os.path.isfile(file_path):
                start = time.time()
 
                # calculate TPS
                heatmap, heatmap_level, all_locs, tc_positive_locs = calculate_TPS(file_path, WSI_object)

                calc_tps_time_elapsed = time.time() - start
                logger.debug("calculate TPS took {:>10.5f} seconds".format(calc_tps_time_elapsed))


        # save stitching heatmap of patches
        stitch_time_elapsed = -1
        if stitch:
            # 視覚化用にヒートマップをリサイズする
            downscale = 16
            if obj_pow == 20:
                downscale = 8
            elif obj_pow == 10:
                downscale = 4
            wsi = WSI_object.getOpenSlide()
            vis_level = wsi.get_best_level_for_downsample(downscale)

            def resize_to_vis_level(img, level_from, level_to):
                assert level_from <= level_to
                w, h = wsi.level_dimensions[level_to]
                return cv2.resize(np.array(img).astype(np.float32), (w, h))
            heatmap_vis_level = resize_to_vis_level(heatmap, level_from=heatmap_level, level_to=vis_level)
            del wsi, heatmap

            def rescale_to_vis_level(locs, level_from, level_to):
                assert level_from <= level_to
                locs = [loc / 2**(level_to-level_from) for loc in locs]
                return locs
            all_locs = rescale_to_vis_level(all_locs, level_from=heatmap_level, level_to=vis_level)
            tc_positive_locs = rescale_to_vis_level(tc_positive_locs, level_from=heatmap_level, level_to=vis_level)

            file_path = WSI_object.hdf5_file
            if os.path.isfile(file_path):
                start = time.time()
                # Patch and save in WSI_object.hdf5_file
 
                # Stitch
                StitchCoords(file_path, WSI_object, stitch_save_dir, downscale=downscale, bg_color=(0,0,0), alpha=-1,
                             draw_grid=True, draw_contour=True, overlaymap=heatmap_vis_level, all_locs=all_locs, tc_positive_locs=tc_positive_locs) 

                stitch_time_elapsed = time.time() - start
                logger.debug("stitching took     {:>10.5f} seconds".format(stitch_time_elapsed))

        logger.debug("segmentation took  {:>10.5f} seconds".format(seg_time_elapsed))
        logger.debug("patching took      {:>10.5f} seconds".format(patch_time_elapsed))
        logger.debug("detection took     {:>10.5f} seconds".format(detection_time_elapsed))
        logger.debug("segmentation took  {:>10.5f} seconds".format(segmentation_time_elapsed))
        logger.debug("calculate TPS took {:>10.5f} seconds".format(calc_tps_time_elapsed))
        logger.debug("stitching took     {:>10.5f} seconds".format(stitch_time_elapsed))
        df.loc[idx, 'status'] = 'processed'

        seg_times += seg_time_elapsed
        patch_times += patch_time_elapsed
        stitch_times += stitch_time_elapsed
        del WSI_object, heatmap_vis_level, all_locs, tc_positive_locs
        gc.collect()

    seg_times /= total
    patch_times /= total
    stitch_times /= total

    df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
    logger.debug("average segmentation time in s per slide: {:>10.5f}".format(seg_times))
    logger.debug("average patching time in s per slide:     {:>10.5f}".format(patch_times))
    logger.debug("average stiching time in s per slide:     {:>10.5f}".format(stitch_times))
        
    return seg_times, patch_times




parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source_slides', type = str,
                    help='path to folder containing raw wsi image files')
parser.add_argument('--source_annotations', type = str,
                    help='path to folder containing annotation files')
parser.add_argument('--step_size', type = int, default=256,
                    help='step_size')
parser.add_argument('--patch_size', type = int, default=256,
                    help='patch_size')
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--segmentation', default=False, action='store_true')
parser.add_argument('--detection', default=False, action='store_true')
parser.add_argument('--save_dir', type = str,
                    help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str,
                    help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_resolution_detection', type=int, default=40, choices=[5, 10, 20, 40],
                    help='equivalent objective power at which to patch')
parser.add_argument('--patch_resolution_segmentation', type=int, default=10, choices=[5, 10, 20, 40],
                    help='equivalent objective power at which to patch')
parser.add_argument('--process_list',  type = str, default=None,
                    help='name of list of images to process with parameters (.csv)')
parser.add_argument('--ckpts_detection', type=str, default='')
parser.add_argument('--ckpts_segmentation', type=str, default='')
parser.add_argument('--intensity_thres', type=int, default=175)
parser.add_argument('--area_thres', type=float, default=0.1)
parser.add_argument('--radius', type=int, default=25)


if __name__ == '__main__':
    args = parser.parse_args()

    patch_save_dir = os.path.join(args.save_dir, 'patches')
    mask_save_dir = os.path.join(args.save_dir, 'masks')
    stitch_save_dir = os.path.join(args.save_dir, 'stitches')

    if args.process_list:
        process_list = os.path.join(args.save_dir, args.process_list)

    else:
        process_list = None

    # prepare logger
    #curr_date    = datetime.now().strftime('%Y%m%d_%H%M')
    log_file     = os.path.join(args.save_dir, "processing.log")
    if not Path(log_file).exists():
        Path(log_file).parent.mkdir(parents=True)
    set_logger(logger, log_file, level=DEBUG, level_fh=DEBUG, level_ch=DEBUG) 

    logger.debug(f'source_slides: {args.source_slides}')
    logger.debug(f'source_annotations: {args.source_annotations}')
    logger.debug(f'patch_save_dir: {patch_save_dir}')
    logger.debug(f'mask_save_dir: {mask_save_dir}')
    logger.debug(f'stitch_save_dir: {stitch_save_dir}')
    
    directories = {'source_slides': args.source_slides, 
                   'source_annotations': args.source_annotations,
                   'save_dir': args.save_dir,
                   'patch_save_dir': patch_save_dir, 
                   'mask_save_dir' : mask_save_dir, 
                   'stitch_save_dir': stitch_save_dir} 

    for key, val in directories.items():
        logger.debug("{} : {}".format(key, val))
        if key not in ['source_slides', 'source_annotations']:
            os.makedirs(val, exist_ok=True)

    seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'}
    filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
    vis_params = {'vis_level': -1, 'line_thickness': 200, 'number_contours': True}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'} # (choices between 'four_pt' - checks if one of four points need to be inside the contour, 'four_pt_hard' - checks if all four points need to be inside the contour, 'center' - checks if the center of the patch is inside the contour, 'basic' - checks if the top-left corner of the patch is inside the contour, default: 'four_pt')

    if args.preset:
        preset_df = pd.read_csv(os.path.join('presets', args.preset))
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]

        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]

        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]

        for key in patch_params.keys():
            patch_params[key] = preset_df.loc[0, key]
    
    seg_params["use_otsu"]=True
    filter_params["a_t"]=1
    filter_params["a_h"]=1
    filter_params["max_n_holes"]=2
    patch_params['contour_fn'] = '13_pt' #'five_pt' '13_pt' 'four_pt' # 'four_pt_hard' 'center' 'basic'

    parameters = {'seg_params': seg_params,
                  'filter_params': filter_params,
                   'patch_params': patch_params,
                  'vis_params': vis_params}

    logger.debug(parameters)

    seg_times, patch_times = seg_and_patch(**directories, **parameters,
                                            patch_size = args.patch_size, step_size=args.step_size, 
                                            seg = args.seg,  use_default_params=False, save_mask = True, 
                                            stitch= args.stitch,
                                            patch_resolution_detection=args.patch_resolution_detection, 
                                            patch_resolution_segmentation=args.patch_resolution_segmentation,
                                            patch = args.patch,
                                            process_list = process_list, auto_skip=args.no_auto_skip,
                                            segmentation = args.segmentation, detection=args.detection,
                                            model_path_detection=args.ckpts_detection,
                                            model_path_segmentation=args.ckpts_segmentation,
                                            intensity_thres=args.intensity_thres,
                                            area_thres=args.area_thres,
                                            radius=args.radius)

# TODO
# onnx -> detection model , seg model re learn without bilinearupsample, adaptive pooling
# log the results with conf settings.
# makefile
#  -> 学習用と推論用(deploy用)で分ける
# defaultでセーブしないようにし、後から必要に応じて画像を作るようにする
# detect_tc_positive_nucleiの高速化検討(TC(+)count)
# -> forwardのところでsave_dir をdefaultでnoneにして、スキップする条件をsave_dir=noneのときはスキップしないようにすれば、後からforwardをすることができる。stitchingはコマンド引数で--stitchを渡すかどうかで制御出来る
#    ただしとりあえず保留
# 一度forward完了していても、ただスキップするだけでもパッチ数が多いとそれなりに時間かかる。モデルのロードに1.5秒、あとはパッチ数に応じてループに時間かかる。パッチ数の多いdetectionの方が、segmentationよりも時間かかる
#  -> これはどうしようもない。onnxにしても変わらないだろう。dataloaderを工夫できればいいが、pytorchのは優秀なので、これ以上は厳しいのでは？
# 丸めの問題　np.ceil を使っているところ。少数をintにキャストした時の座標の問題確認
# contourの番号を描いてから点を上書きしているので汚い
# hole 対応
# 無駄な処理しているところないか
# hdf5 fileからROIをjpgにして保存する機能 or seg, detectionのときに元の画像のROIも一緒に保存してしまう
# pyspark
# hdf5 viewerをインストール
# c++移行
# gui
# docker compose?
# sidecar container?
# np.vstackで0の配列はエラーになる
# github copilot
# TODO slow　の箇所
# detectionの閾値の設定、変更方法およびデータの持たせ方検討
# 気管支腺の領域の予測が、それを含む間質をちゃんと学習しているので、間質をアノテーションしたら結構うまく分けられる気がする。が、アノテーションが大変。
# 学習時に特徴空間でクラスタリングされるような最適化をしたらどうか？->セグメンテーションだとラベル付けが難しい。


# DONE
# overlayの時間短縮(paint_circle)
# forwardを途中で止めても、再開は途中から開始できる
# patch 取得のルーチン化
# stitch coord de patch 取得のルーチン化
# pil -> cv (resize fast)
# mapのoverlayが囲みからはみ出ているのを直す
# contourをはみ出たものも描画・カウントされている?
# calculate TPSでもっとfasterなアルゴリズム -> dataframeにしてベクトル演算?
# overlay するときに、囲み領域だけoverlayしないと、全体的に暗くなってしまう
# 囲みが閉じてない場合でも大丈夫か?
# 閾値が変更された場合、囲みが追加削除変更された場合の処理
# auto gen csvの扱い
#   -> csvでprocessのフラグをチェックするのをやめる(no_auto_skip引数をデフォルトにする)
# overlap strategy
# マージ画像で核検出だけ、セグメンテーションだけ、元画像だけ、と分けて出力
# patchの階層をなくして、contourの階層直下にarrayでもたせる。→.h5の容量、速度が早くなるはず
# test when thres changed
# open -> with xx:  (to gracefully close the file)
# detection network -> learn at 20x
# onnx model -> 対応してないopsが多いので諦めた
# 5ptでROIチェックをしているところ、囲みが小さすぎる場合への対応 -> add 13pt
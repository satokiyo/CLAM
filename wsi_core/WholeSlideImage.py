import math
import os
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
import multiprocessing as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
from PIL import Image
import h5py
import math
from wsi_core.wsi_utils import savePatchIter_bag_hdf5, initialize_hdf5_bag, coord_generator, save_hdf5, sample_indices, screen_coords, isBlackPatch, isWhitePatch, to_percentiles, HDFVisitor
import itertools
from wsi_core.util_classes import isInContourV1, isInContourV2, isInContourV3_Easy, isInContourV3_Hard, isInContourV3_Easy_5pt, isInContourV3_Easy_13pt, Contour_Checking_fn
from utils.file_utils import load_pkl, save_pkl
from wsi_core.preprocessing import create_tile_generator, get_20x_zoom_level, process_slide, process_tile_index, optical_density, keep_tile, process_tile, normalize_staining, flatten_sample_tuple, flatten_sample, get_labels_df, preprocess, save_df, add_row_indices, sample, rdd_2_df, save_rdd_2_jpeg, save_2_jpeg, save_nonlabelled_sample_2_jpeg, save_labelled_sample_2_jpeg, save_jpeg_help
Image.MAX_IMAGE_PIXELS = 933120000
from pathlib import Path
from utils.file_utils import create_hdf5_group, create_hdf5_dataset, create_hdf5_attrs, open_hdf5_file
import re
import gc
from numba import jit, uint8, float64, boolean

import pdb
from logging import getLogger

logger = getLogger(f'pdl1_module.{__name__}')

class WholeSlideImage(object):
    def __init__(self, path):

        """
        Args:
            path (str): fullpath to WSI file
        """

        self.path = path
        self.name = ".".join(path.split("/")[-1].split('.')[:-1])
        self.wsi = openslide.open_slide(path) # Open Whole-Slide Image
        self.level_downsamples = self._assertLevelDownsamples()
        self.level_dim = self.wsi.level_dimensions
        self.vendor = self.wsi.properties.get('openslide.vendor')
        self.objective_power = self.wsi.properties.get('openslide.objective-power')

        self.contours_tissue = None
#        self.contours_tumor = None
        self.hdf5_file = None

    def getOpenSlide(self):
        return self.wsi

    def initXML(self, xml_path):
        def _createContour(coord_list):
            return np.array([[[int(float(coord.attributes['X'].value)), 
                               int(float(coord.attributes['Y'].value))]] for coord in coord_list], dtype = 'int32')

        xmldoc = minidom.parse(xml_path)
        annotations = [anno.getElementsByTagName('Coordinate') for anno in xmldoc.getElementsByTagName('Annotation')]

#        self.contours_tumor  = [_createContour(coord_list) for coord_list in annotations]
#        self.contours_tumor = sorted(self.contours_tumor, key=cv2.contourArea, reverse=True)
        self.contours_xml  = [_createContour(coord_list) for coord_list in annotations]
        self.contours_xml = sorted(self.contours_xml, key=cv2.contourArea, reverse=True)

        # TODO get hole contour
        hole_contours = []

        return self.contours_xml, hole_contours



    def initXML2(self, path, seg_level=0): # ?????????mag_idx != level
        ''' Parses the xml at the given path, assuming annotation format importable by ImageScope. '''

        mag_idx = seg_level
        # ndpi ????????????????????????????????????????????? (???????????????????????????????????????????????????/pixel??????)###########
        prp_mppx = float(self.wsi.properties.get('openslide.mpp-x') )
        prp_mppy = float(self.wsi.properties.get('openslide.mpp-y') )
        prp_offset_x = int(self.wsi.properties.get('hamamatsu.XOffsetFromSlideCentre'))
        prp_offset_y = int(self.wsi.properties.get('hamamatsu.YOffsetFromSlideCentre'))
        prp_hig = int(self.wsi.properties.get('openslide.level[' + str(mag_idx) + '].height'))
        prp_wid = int(self.wsi.properties.get('openslide.level[' + str(mag_idx) + '].width'))
        prp_downsample = int(self.wsi.properties.get('openslide.level[' + str(mag_idx) + '].downsample'))
        
        # ?????????????????????mag_idx???????????????????????? wid , hig ??? mag_idx 0 ????????????????????????
        wid_mag0 = int(self.wsi.properties.get('openslide.level[0].width'))
        hig_mag0 = int(self.wsi.properties.get('openslide.level[0].height'))
        ###################################################################################

        # ?????????????????????????????????????????????????????????????????????HE????????????mag_idx????????????????????????????????????????????????????????????
        draw_down_sampling_rate = 1
    
        def hex_to_rgb(value):
            value = value.lstrip('#')
            lv = len(value)
            np_RGB = np.array( [int(value[ i:i + lv //3] , 16) for i in range(0,lv,lv//3)] )
            #np_BGR = np.array( [ np_RGB[2] , np_RGB[1] , np_RGB[0] ] , dtype='uint8')
            np_BGR =  tuple( [int(np_RGB[2]) , int(np_RGB[1]) , int(np_RGB[0]) ] )
            return np_BGR
    
        def micrometer2pix_hamahoto( point , npp , offset , size_mag0 ):
            
            if(len(point ) != 2):
               logger.debug("micrometer2pix_hamahoto : input error")
               exit()
            if(len(npp   ) != 2):
               logger.debug("micrometer2pix_hamahoto : input error")
               exit()
            if(len(offset) != 2):
               logger.debug("micrometer2pix_hamahoto : input error")
               exit()
            
            x_nanometer  = point[0]
            y_nanometer  = point[1]
            x_npp        = npp[0]
            y_npp        = npp[1]
            x_offset     = offset[0]
            y_offset     = offset[1]
            wid_pix_mag0 = size_mag0[0]
            hig_pix_mag0 = size_mag0[1]
            
            # ????????????????????????????????????????????????????????????????????????:?????????????????????????????????
            # ??????????????????????????????????????????????????????????????????????????????:????????????????????????
            x_offset_from_imgLT = (wid_pix_mag0 / 2) * x_npp - x_offset
            y_offset_from_imgLT = (hig_pix_mag0 / 2) * y_npp - y_offset
            
            # ?????????????????????????????????????????????????????? point ??????????????????
            x_nanometer_from_pix0point = x_offset_from_imgLT + x_nanometer
            y_nanometer_from_pix0point = y_offset_from_imgLT + y_nanometer
            
            # nm ?????? ?????????????????????
            x_pix = x_nanometer_from_pix0point / x_npp
            y_pix = y_nanometer_from_pix0point / y_npp
            
            return x_pix , y_pix
    
    
        # xml ????????????
        tree = ET.parse(path)
        root = tree.getroot()
        
        # ?????????????????????????????????????????????
        anno_info_list_color = []
        anno_info_list_point = []
        anno_info_list_point_color_white = []
        anno_info_list_ccode = []
        anno_info_list_title = []
        anno_info_ccode_class_idx = []

        #????????????????????????????????????????????????
        for child in root:
        
            # xml?????????????????????????????????????????????
            plist           = child.find("annotation").find("pointlist")
            anno_color_code = child.find("annotation").attrib["color"]
            #annotation      = ColorCode2annotation(anno_color_code)
            annotation      = hex_to_rgb(anno_color_code)
            logger.debug(anno_color_code)
        
            title = child.find("title").text
            #logger.debug(title)
        
            xlist_pix = []
            ylist_pix = []
            for point in plist:
        
                point_x_micrometer = int(point.find("x").text)
                point_y_micrometer = int(point.find("y").text)
        
                # ndpa??????????????????????????????????????????mpp???npp(nanometer per pix)???????????????
                prp_nppx = prp_mppx * 1000
                prp_nppy = prp_mppy * 1000
                # npp ??? pixel???????????????
                point_x_pix , point_y_pix = micrometer2pix_hamahoto( (point_x_micrometer, point_y_micrometer) , (prp_nppx , prp_nppy) , (prp_offset_x , prp_offset_y) , (wid_mag0 , hig_mag0))
        
                point_x_pix_target_mag = int((point_x_pix / prp_downsample) / draw_down_sampling_rate)
                point_y_pix_target_mag = int((point_y_pix / prp_downsample) / draw_down_sampling_rate)
        
        
        
                xlist_pix += [point_x_pix_target_mag]
                ylist_pix += [point_y_pix_target_mag]
        
        
            #xy_array = np.array([np.array([xlist_pix , ylist_pix]).T])
            #xy_array = np.array([np.array([xlist_pix , ylist_pix])]).T
            #xy_array = np.array([[xlist_pix , ylist_pix]])
            xy_array = np.array([np.array([[x_pix , y_pix]]) for x_pix, y_pix in zip(xlist_pix, ylist_pix)])
    
            # ???????????????????????????????????????????????????????????????????????????????????????
            if(anno_color_code == '#ffffff'):
                anno_info_list_point_color_white.append(xy_array)
                continue
        
            anno_info_list_color.append(annotation)
            anno_info_list_ccode.append(anno_color_code)
            anno_info_list_point.append(xy_array)
            anno_info_list_title.append(title)

        #self.contours_xml  = [_createContour(coord_list) for coord_list in annotations]
        #self.contours_xml = sorted(self.contours_xml, key=cv2.contourArea, reverse=True)
        self.contours_xml  = anno_info_list_point
        self.contours_xml = sorted(self.contours_xml, key=cv2.contourArea, reverse=True)

        hole_contours = [[] for i in range(len(self.contours_xml))] # dummy hole
        self.contours_xml_white = anno_info_list_point_color_white

        return self.contours_xml, hole_contours


    def initTxt(self,annot_path):
        def _create_contours_from_dict(annot):
            all_cnts = []
            for idx, annot_group in enumerate(annot):
                contour_group = annot_group['coordinates']
                if annot_group['type'] == 'Polygon':
                    for idx, contour in enumerate(contour_group):
                        contour = np.array(contour).astype(np.int32).reshape(-1,1,2)
                        all_cnts.append(contour) 

                else:
                    for idx, sgmt_group in enumerate(contour_group):
                        contour = []
                        for sgmt in sgmt_group:
                            contour.extend(sgmt)
                        contour = np.array(contour).astype(np.int32).reshape(-1,1,2)    
                        all_cnts.append(contour) 

            return all_cnts
        
        with open(annot_path, "r") as f:
            annot = f.read()
            annot = eval(annot)
#        self.contours_tumor  = _create_contours_from_dict(annot)
#        self.contours_tumor = sorted(self.contours_tumor, key=cv2.contourArea, reverse=True)

#    def initSegmentation(self, mask_file):
#        # load segmentation results from pickle file
#        import pickle
#        asset_dict = load_pkl(mask_file)
#        self.holes_tissue = asset_dict['holes']
#        self.contours_tissue = asset_dict['tissue']

    def saveSegmentation(self, mask_file):
        # save segmentation results using pickle
        asset_dict = {'holes': self.holes_tissue, 'tissue': self.contours_tissue}
        save_pkl(mask_file, asset_dict)

    def segmentTissue(self, seg_level=0, sthresh=20, sthresh_up = 255, mthresh=7, close = 0, use_otsu=False, 
                            filter_params={'a_t':100}, ref_patch_size=512, exclude_ids=[], keep_ids=[]):
        """
            Segment the tissue via HSV  -> Binary threshold
        """
        
       
        img = np.array(self.wsi.read_region((0,0), seg_level, self.level_dim[seg_level]))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV space
        img_med = img_hsv[:,:,1]
        
       
        # Thresholding
        if use_otsu:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            img_med[img_gray[:,:]<65] = 0 # remove black 65> in gray pixels
            thres, _ = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
            img_bin = np.zeros((img_gray.shape), dtype=np.uint8)
            img_bin[img_med >= thres] = 255
            logger.debug(f"th_otsu_sat : {thres}")
            self.sat_thres = thres
        else:
            _, img_bin = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

        # Morphological closing
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)                 

        scale = self.level_downsamples[seg_level]
        scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))
       
        logger.debug(f"vendor : {self.vendor}")
        if self.vendor == 'hamamatsu':
            xml_path = self.path + '.ndpa'
        elif self.vendor == 'aperio':
            xml_path = ".".join(self.path.split('.')[:-1]) + '.xml' # aperio

        if Path(xml_path).exists(): # segment tissue based on xml
            # ndpa as contour, white as hole.
            def findContours_from_xml(xml_path, seg_level):
                #self.initXML(xml_path) # xml for svs ?
                return self.initXML2(xml_path, seg_level) # ndpa for ndpi

            contours,  hole_contours = findContours_from_xml(xml_path=xml_path, seg_level=seg_level) # Find contours 
            foreground_contours = contours

        else: # segment tissue automatically
            # Find and filter contours
            filter_params = filter_params.copy()
            filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area # area filter threshold for tissue (positive integer, the minimum size of detected foreground contours to consider, relative to a reference patch size of 512 x 512 at level 0, e.g. a value 10 means only detected foreground contours of size greater than 10 512 x 512 sized patches at level 0 will be processed, default: 100)
            filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area # area filter threshold for holes (positive integer, the minimum size of detected holes/cavities in foreground contours to avoid, once again relative to 512 x 512 sized patches at level 0, default: 16)
 
            contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # Find contours 
            hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
            if filter_params:
                def _filter_contours(contours, hierarchy, filter_params):
                    """
                        Filter contours by: area.
                    """
                    filtered = []
        
                    # find indices of foreground contours (parent == -1)
                    hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)
                    all_holes = []
                    
                    # loop through foreground contour indices
                    for cont_idx in hierarchy_1:
                        # actual contour
                        cont = contours[cont_idx]
                        # indices of holes contained in this contour (children of parent contour)
                        holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
                        # take contour area (includes holes)
                        a = cv2.contourArea(cont)
                        # calculate the contour area of each hole
                        hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
                        # actual area of foreground contour region
                        a = a - np.array(hole_areas).sum()
                        if a == 0: continue
                        if tuple((filter_params['a_t'],)) < tuple((a,)): 
                            filtered.append(cont_idx)
                            all_holes.append(holes)
        
                    foreground_contours = [contours[cont_idx] for cont_idx in filtered]
                    
                    hole_contours = []
        
                    for hole_ids in all_holes:
                        unfiltered_holes = [contours[idx] for idx in hole_ids ]
                        unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                        # take max_n_holes largest holes by area
                        unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
                        filtered_holes = []
                        
                        # filter these holes
                        for hole in unfilered_holes:
                            if cv2.contourArea(hole) > filter_params['a_h']:
                                filtered_holes.append(hole)
        
                        hole_contours.append(filtered_holes)
        
                    return foreground_contours, hole_contours
 
                foreground_contours, hole_contours = _filter_contours(contours, hierarchy, filter_params)  # Necessary for filtering out artifacts


        self.contours_tissue = self.scaleContourDim(foreground_contours, scale) # scale coord value to level0
        self.holes_tissue = self.scaleHolesDim(hole_contours, scale) # scale coord value to level0

        if len(keep_ids) > 0:
            contour_ids = set(keep_ids) - set(exclude_ids)
        else:
            contour_ids = set(np.arange(len(self.contours_tissue))) - set(exclude_ids)

        self.contours_tissue = [self.contours_tissue[i] for i in contour_ids]
        #self.holes_tissue = [self.holes_tissue[i] for i in contour_ids]

    def visWSI(self, vis_level=0, color = (0,255,0), hole_color = (255,0,0),
                    line_thickness=250, max_size=None, top_left=None, bot_right=None, custom_downsample=1, view_slide_only=False,
                    number_contours=False, seg_display=True):
        
        downsample = self.level_downsamples[vis_level]
        scale = [1/downsample[0], 1/downsample[1]]
        
        if top_left is not None and bot_right is not None:
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)
        else:
            top_left = (0,0)
            region_size = self.level_dim[vis_level]
 
        img = np.array(self.wsi.read_region(top_left, vis_level, region_size))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        
        if not view_slide_only:
            offset = tuple(-(np.array(top_left) * scale).astype(int))
            line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
            if self.contours_tissue is not None and seg_display:
                if not number_contours: # draw contour id 
                    cv2.drawContours(img, self.scaleContourDim(self.contours_tissue, scale), 
                                     -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)

                else: # add numbering to each contour
                    for idx, cont in enumerate(self.contours_tissue):
                        contour = np.array(self.scaleContourDim(cont, scale))
                        M = cv2.moments(contour)
                        cX = int(M["m10"] / (M["m00"] + 1e-9))
                        cY = int(M["m01"] / (M["m00"] + 1e-9))
                        # draw the contour and put text next to center
                        cv2.drawContours(img,  [contour], -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)
                        cv2.putText(img, "{}".format(idx), (cX, cY),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                for holes in self.holes_tissue:
                    cv2.drawContours(img, self.scaleContourDim(holes, scale), 
                                     -1, hole_color, line_thickness, lineType=cv2.LINE_8)
            
        h, w = img.shape[:2]
        if custom_downsample > 1:
            #img = img.resize((int(w/custom_downsample), int(h/custom_downsample)))
            img = cv2.resize(img, (int(h/custom_downsample), int(w/custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size/w if w > h else max_size/h
            #img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))
            img = cv2.resize(img, (int(h*resizeFactor), int(w*resizeFactor)))
       
        return img


    def createPatches_bag_hdf5(self, save_path, patch_level=0, patch_size=256, step_size=256, save_coord=True, **kwargs):
        contours = self.contours_tissue
        contour_holes = self.holes_tissue

        logger.debug(f"Creating patches for: {self.name} ...")
        elapsed = time.time()
        for idx, cont in enumerate(contours):
            patch_gen = self._getPatchGenerator(cont, idx, patch_level, save_path, patch_size, step_size, **kwargs)
            
            if self.hdf5_file is None:
                try:
                    first_patch = next(patch_gen)

                # empty contour, continue
                except StopIteration:
                    continue

                file_path = initialize_hdf5_bag(first_patch, save_coord=save_coord)
                self.hdf5_file = file_path

            for patch in patch_gen:
                savePatchIter_bag_hdf5(patch)

        return self.hdf5_file


    def _getPatchGenerator(self, cont, cont_idx, patch_level, save_path, patch_size=256, step_size=256, custom_downsample=1,
        white_black=True, white_thresh=15, black_thresh=50, contour_fn='four_pt', use_padding=True):
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])
        logger.debug(f"Bounding Box: {start_x} {start_y} {w} {h}")
        logger.debug(f"Contour Area: {cv2.contourArea(cont)}")
        
        if custom_downsample > 1:
            assert custom_downsample == 2 
            target_patch_size = patch_size
            patch_size = target_patch_size * 2
            step_size = step_size * 2
            logger.debug("Custom Downsample: {}, Patching at {} x {}, But Final Patch Size is {} x {}".format(custom_downsample, patch_size, patch_size, 
                target_patch_size, target_patch_size))

        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1])
        
        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]
        
        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt':
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'four_pt_hard':
                cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'center':
                cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
            elif contour_fn == 'basic':
                cont_check_fn = isInContourV1(contour=cont)
            elif contour_fn == 'five_pt':
                cont_check_fn = isInContourV3_Easy_5pt(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == '13_pt':
                cont_check_fn = isInContourV3_Easy_13pt(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn

        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y+h
            stop_x = start_x+w
        else:
            stop_y = min(start_y+h, img_h-ref_patch_size[1])
            stop_x = min(start_x+w, img_w-ref_patch_size[0])

        count = 0
        for y in range(start_y, stop_y, step_size_y):
            for x in range(start_x, stop_x, step_size_x):

                if not self.isInContours(cont_check_fn, (x,y), self.holes_tissue[cont_idx], ref_patch_size[0]): #point not inside contour and its associated holes
                    continue    
                
                count+=1
                patch_PIL = self.wsi.read_region((x,y), patch_level, (patch_size, patch_size)).convert('RGB')
                if custom_downsample > 1:
                    patch_PIL = patch_PIL.resize((target_patch_size, target_patch_size))
                
                if white_black:
                    if isBlackPatch(np.array(patch_PIL), rgbThresh=black_thresh) or isWhitePatch(np.array(patch_PIL), satThresh=white_thresh): 
                        continue

                patch_info = {'x':x // (patch_downsample[0] * custom_downsample), 'y':y // (patch_downsample[1] * custom_downsample), 'cont_idx':cont_idx, 'patch_level':patch_level, 
                'downsample': self.level_downsamples[patch_level], 'downsampled_level_dim': tuple(np.array(self.level_dim[patch_level])//custom_downsample), 'level_dim': self.level_dim[patch_level],
                'patch_PIL':patch_PIL, 'name':self.name, 'save_path':save_path}

                yield patch_info

        
        logger.debug("patches extracted: {}".format(count))

    @staticmethod
    def isInHoles(holes, pt, patch_size):
        for hole in holes:
            if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
                return 1
        
        return 0

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
        if cont_check_fn(pt):
            if holes is not None:
                return not WholeSlideImage.isInHoles(holes, pt, patch_size)
            else:
                return 1
        return 0

    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]

    def _assertLevelDownsamples(self):
        level_downsamples = []
        dim_0 = self.wsi.level_dimensions[0]
        
        for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) else level_downsamples.append((downsample, downsample))
        
        return level_downsamples

    def process_contours(self, save_path, patch_level={'detection':1, 'segmentation':2}, patch_size=256, step_size=256, **kwargs):
        '''
        wsi???????????????contours(????????????)???????????????????????????

        hdf5 file???dataset???????????????????????????????????????????????????????????????

        /target/contourxx/coords_contour : ???????????????????????????
        /target/contourxx/patchxx/coord  : ??????????????????
        (target = detection or segmentation)

        ???????????????????????????????????????
        / : name(???????????????)
        /target : patch_size, patch_level, downsample, downsampled_level_dim, level_dim

        '''
        assert patch_level['segmentation'] >= patch_level['detection']
        save_path_hdf5 = os.path.join(save_path, str(self.name) + '.h5')
        logger.debug(f"Creating patches for: {self.name} ...")
        n_contours = len(self.contours_tissue)
        logger.debug(f"Total number of contours to process: {n_contours}")
        fp_chunk_size = math.ceil(n_contours * 0.05)

        # group??????
        with open_hdf5_file(save_path_hdf5, mode='a') as f:
            create_hdf5_attrs(f, name='name', data=self.name) # create attrs at /

            for target in ['segmentation', 'detection']:
                if target in f.keys():
                    grp = f[target]
                else:
                    grp = create_hdf5_group(f, target) # create group at /target

                # /detection or /segmentation : patch_level or patch_size or step_size???????????????????????????????????????????????????????????????????????????????????????????????????
                if grp.attrs:
                    if (patch_level[target] != grp.attrs.get('patch_level')) or (patch_size != grp.attrs.get('patch_size') or (step_size != grp.attrs.get('step_size'))):
                        grp.clear()
                        grp.attrs.clear()

                # create attrs at /target
                attr = {'patch_size'            : patch_size, # patch_size. Not patch size in reference frame(level 0)
                        'step_size'             : step_size,  # step_size. Not ref level(level 0)
                        'patch_level'           : patch_level[target], # patch_level. Not ref level(level 0)
                        'downsample'            : self.level_downsamples[patch_level[target]],
                        'downsampled_level_dim' : tuple(np.array(self.level_dim[patch_level[target]])),
                        'level_dim'             : self.level_dim[patch_level[target]],}
                for name, data in attr.items():
                    create_hdf5_attrs(grp, name=name, data=data) 

                cont_idx_processed = [s for s in grp] # ex: ["contour0", "contour1", "contour2"]
                if cont_idx_processed:
                    cont_idx_offset = max([re.findall(r'\d+', s)[-1] for s in cont_idx_processed]) # contour number digit. ex: 2
                else:
                    cont_idx_offset = -1


                # dataset?????????????????????????????????dataset????????????
                queries = ['coords_contour']
                v = HDFVisitor(*queries)
                grp.visititems(v)
                processed_contours = v.container['coords_contour']

                def is_exist_ainb(a:list, b:list):
                    '''
                    list a ????????????list b??????????????????????????????list a ??????????????????????????????????????? 
                    '''
                    ret = []
                    for i, e1 in enumerate(a):
                        for e2 in b:
                            if np.all(e1 == e2):
                                ret.append(i)
                    return ret

                to_skip_list = is_exist_ainb(a=self.contours_tissue, b=processed_contours) # ????????????contour??????????????????contour??????????????????

                # processed_contours???????????????contour???.h5????????????(????????????or??????????????????)
                exist_both = is_exist_ainb(a=processed_contours, b=self.contours_tissue) # ???????????????contour
                no_longer_exist = [idx for idx in range(len(processed_contours)) if idx not in exist_both] # processed_contours???????????????contour???.h5????????????(????????????or??????????????????)
                for delete_idx in no_longer_exist:
                    to_delete = processed_contours[delete_idx]
                    for processed_cont in grp:
                        if np.all(grp[processed_cont]['coords_contour'][:] == to_delete):
                            grp[processed_cont].clear()
                            del grp[processed_cont]

                # contour????????????
                add_idx = 0
                for idx_cont, cont in enumerate(self.contours_tissue): # contour coords at level0
                    if (idx_cont + 1) % fp_chunk_size == fp_chunk_size:
                        logger.debug('Processing contour {}/{}'.format(idx_cont, n_contours))

                    if idx_cont in to_skip_list:
                        continue
                    add_idx += 1

                    grp_cont = create_hdf5_group(grp, f'contour{int(cont_idx_offset) + add_idx}') # /target/contourxx contour??????????????????????????????????????????
                    create_hdf5_dataset(grp_cont, 'coords_contour', data=cont) # dataset at /target/contourxx/coords_contour

                    if target == "segmentation":
                        # ??????????????????
                        asset_dict = self.process_contour(cont, self.holes_tissue[idx_cont], patch_level[target], save_path, patch_size, step_size, **kwargs)
                        # save 
                        if len(asset_dict) > 0:
                            # save assets per patch
                            create_hdf5_dataset(grp_cont, 'coords_patches', data=asset_dict['coords']) # dataset at /target/contourxx/coords_patches
                        else: # no patches for this cont
                            # TODO
                            pass
                    elif target == "detection":
                        ratio = 2**(patch_level['segmentation'] - patch_level['detection'])
                        seg_patch_coords = f['segmentation'][f'contour{int(cont_idx_offset) + add_idx}']['coords_patches'][:]
                        if ratio == 1:
                            detect_patch_coords = seg_patch_coords
                        if ratio >= 2:
                            patch_downsample_seg = int(self.level_downsamples[patch_level["segmentation"]][0])
                            ref_patch_size_seg = patch_size*patch_downsample_seg
                            increase_step_size = int(ref_patch_size_seg // ratio)
                            detect_patch_coords = []
                            for coord in seg_patch_coords:
                                for i_add_x in range(ratio):
                                    for i_add_y in range(ratio):
                                        add_coord = [coord[0]+increase_step_size*i_add_y, coord[1]+increase_step_size*i_add_x]
                                        detect_patch_coords.append(add_coord)

                        create_hdf5_dataset(grp_cont, 'coords_patches', data=np.array(detect_patch_coords)) # dataset at /target/contourxx/coords_patches
                    else:
                        pass
    
            debug=False
            if debug:
                def print_dataset(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        logger.debug(name)
                        # logger.debug('\t',obj)
                f.visititems(print_dataset)

            f.flush()
            f.close()
            gc.collect()
    
        self.hdf5_file = save_path_hdf5


    def process_contour(self, cont, contour_holes, patch_level, save_path, patch_size = 256, step_size = 256,
        contour_fn='four_pt', use_padding=True, top_left=None, bot_right=None):
        '''
        contour???????????????????????????????????????????????????
        '''
        # ????????????BBOX
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])

        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1])
        
        img_w, img_h = self.level_dim[0] # image size at level0
        if use_padding:
            stop_y = start_y+h
            stop_x = start_x+w
        else:
            stop_y = min(start_y+h, img_h-ref_patch_size[1]+1) # coord at level0
            stop_x = min(start_x+w, img_w-ref_patch_size[0]+1) # coord at level0
        
        logger.debug(f"Bounding Box:{start_x} {start_y} {w} {h}")
        logger.debug(f"Contour Area:{cv2.contourArea(cont)}")

        if bot_right is not None:
            stop_y = min(bot_right[1], stop_y)
            stop_x = min(bot_right[0], stop_x)
        if top_left is not None:
            start_y = max(top_left[1], start_y)
            start_x = max(top_left[0], start_x)

        if bot_right is not None or top_left is not None:
            w, h = stop_x - start_x, stop_y - start_y
            if w <= 0 or h <= 0:
                logger.debug("Contour is not in specified ROI, skip")
                return {}, {}
            else:
                logger.debug(f"Adjusted Bounding Box: {start_x} {start_y} {w} {h}")

        if patch_size > step_size: # overlap tile
            offset = int(patch_size - step_size) // 2
            ref_offset_x = offset*patch_downsample[0]
            ref_offset_y = offset*patch_downsample[1]
            start_x = start_x-ref_offset_x
            if start_x < 0:
                stop_x = stop_x+ref_offset_x-start_x
                start_x = 0
            else:
                stop_x = stop_x+ref_offset_x

            start_y = start_y-ref_offset_y
            if start_y < 0:
                stop_y = stop_y+ref_offset_y-start_y
                start_y = 0
            else:
                stop_y = stop_y+ref_offset_y

        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt':
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'four_pt_hard':
                cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'center':
                cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
            elif contour_fn == 'basic':
                cont_check_fn = isInContourV1(contour=cont)
            elif contour_fn == 'five_pt':
                cont_check_fn = isInContourV3_Easy_5pt(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == '13_pt':
                cont_check_fn = isInContourV3_Easy_13pt(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn

        
        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]

        x_range = np.arange(start_x, stop_x, step=step_size_x)
        y_range = np.arange(start_y, stop_y, step=step_size_y)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
        # ????????????patch?????????
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

        num_workers = mp.cpu_count()
        if num_workers > 4:
            num_workers = 4
        pool = mp.Pool(num_workers)

        iterable = [(coord, contour_holes, ref_patch_size[0], cont_check_fn) for coord in coord_candidates]
        results = pool.starmap(WholeSlideImage.process_coord_candidate, iterable)
        pool.close()
        results = np.array([result for result in results if result is not None])

        args = [(self.path, coord, ref_patch_size, self.sat_thres, 0.05) for coord in results]
        pool = mp.Pool(16)
        results = pool.map(isTissueArea, args) # ???????????????5%?????????ROI??????????????????
        pool.close()
        pool.join()
        results = np.array([result for result in results if result is not None])
        results = np.unique(results, axis=0) # remove duplicated coordinate
        logger.debug('Extracted {} coordinates'.format(len(results)))

        if len(results)>1:
            asset_dict = {'coords' : results}
            return asset_dict
        else:
            return {}


    @staticmethod
    def process_coord_candidate(coord, contour_holes, ref_patch_size, cont_check_fn):
        if WholeSlideImage.isInContours(cont_check_fn, coord, contour_holes, ref_patch_size):
            return coord
        else:
            return None

    def visHeatmap(self, scores, coords, vis_level=-1, 
                   top_left=None, bot_right=None,
                   patch_size=(256, 256), 
                   blank_canvas=False, canvas_color=(220, 20, 50), alpha=0.4, 
                   blur=False, overlap=0.0, 
                   segment=True, use_holes=True,
                   convert_to_percentiles=False, 
                   binarize=False, thresh=0.5,
                   max_size=None,
                   custom_downsample = 1,
                   cmap='coolwarm'):

        """
        Args:
            scores (numpy array of float): Attention scores 
            coords (numpy array of int, n_patches x 2): Corresponding coordinates (relative to lvl 0)
            vis_level (int): WSI pyramid level to visualize
            patch_size (tuple of int): Patch dimensions (relative to lvl 0)
            blank_canvas (bool): Whether to use a blank canvas to draw the heatmap (vs. using the original slide)
            canvas_color (tuple of uint8): Canvas color
            alpha (float [0, 1]): blending coefficient for overlaying heatmap onto original slide
            blur (bool): apply gaussian blurring
            overlap (float [0 1]): percentage of overlap between neighboring patches (only affect radius of blurring)
            segment (bool): whether to use tissue segmentation contour (must have already called self.segmentTissue such that 
                            self.contours_tissue and self.holes_tissue are not None
            use_holes (bool): whether to also clip out detected tissue cavities (only in effect when segment == True)
            convert_to_percentiles (bool): whether to convert attention scores to percentiles
            binarize (bool): only display patches > threshold
            threshold (float): binarization threshold
            max_size (int): Maximum canvas size (clip if goes over)
            custom_downsample (int): additionally downscale the heatmap by specified factor
            cmap (str): name of matplotlib colormap to use
        """

        if vis_level < 0:
            vis_level = self.wsi.get_best_level_for_downsample(32)

        downsample = self.level_downsamples[vis_level]
        scale = [1/downsample[0], 1/downsample[1]] # Scaling from 0 to desired level
                
        if len(scores.shape) == 2:
            scores = scores.flatten()

        if binarize:
            if thresh < 0:
                threshold = 1.0/len(scores)
                
            else:
                threshold =  thresh
        
        else:
            threshold = 0.0

        ##### calculate size of heatmap and filter coordinates/scores outside specified bbox region #####
        if top_left is not None and bot_right is not None:
            scores, coords = screen_coords(scores, coords, top_left, bot_right)
            coords = coords - top_left
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)

        else:
            region_size = self.level_dim[vis_level]
            top_left = (0,0)
            bot_right = self.level_dim[0]
            w, h = region_size

        patch_size  = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
        coords = np.ceil(coords * np.array(scale)).astype(int)
        
        logger.debug('\ncreating heatmap for: ')
        logger.debug(f'top_left: {top_left} bot_right: {bot_right}')
        logger.debug('w: {}, h: {}'.format(w, h))
        logger.debug(f'scaled patch size: {patch_size}')

        ###### normalize filtered scores ######
        if convert_to_percentiles:
            scores = to_percentiles(scores) 

        scores /= 100
        
        ######## calculate the heatmap of raw attention scores (before colormap) 
        # by accumulating scores over overlapped regions ######
        
        # heatmap overlay: tracks attention score over each pixel of heatmap
        # overlay counter: tracks how many times attention score is accumulated over each pixel of heatmap
        overlay = np.full(np.flip(region_size), 0).astype(float)
        counter = np.full(np.flip(region_size), 0).astype(np.uint16)      
        count = 0
        for idx in range(len(coords)):
            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:
                if binarize:
                    score=1.0
                    count+=1
            else:
                score=0.0
            # accumulate attention
            overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += score
            # accumulate counter
            counter[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += 1

        if binarize:
            logger.debug('\nbinarized tiles based on cutoff of {}'.format(threshold))
            logger.debug('identified {}/{} patches as positive'.format(count, len(coords)))
        
        # fetch attended region and average accumulated attention
        zero_mask = counter == 0

        if binarize:
            overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
        else:
            overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
        del counter 
        if blur:
            overlay = cv2.GaussianBlur(overlay,tuple((patch_size * (1-overlap)).astype(int) * 2 +1),0)  

        if segment:
            tissue_mask = self.get_seg_mask(region_size, scale, use_holes=use_holes, offset=tuple(top_left))
            # return Image.fromarray(tissue_mask) # tissue mask
        
        if not blank_canvas:
            # downsample original image and use as canvas
            img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
        else:
            # use blank canvas
            img = np.array(Image.new(size=region_size, mode="RGB", color=(255,255,255))) 

        #return Image.fromarray(img) #raw image

        logger.debug('\ncomputing heatmap image')
        logger.debug('total of {} patches'.format(len(coords)))
        twenty_percent_chunk = max(1, int(len(coords) * 0.2))

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        
        for idx in range(len(coords)):
            if (idx + 1) % twenty_percent_chunk == 0:
                logger.debug('progress: {}/{}'.format(idx, len(coords)))
            
            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:

                # attention block
                raw_block = overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]]
                
                # image block (either blank canvas or orig image)
                img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy()

                # color block (cmap applied to attention block)
                color_block = (cmap(raw_block) * 255)[:,:,:3].astype(np.uint8)

                if segment:
                    # tissue mask block
                    mask_block = tissue_mask[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] 
                    # copy over only tissue masked portion of color block
                    img_block[mask_block] = color_block[mask_block]
                else:
                    # copy over entire color block
                    img_block = color_block

                # rewrite image block
                img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = img_block.copy()
        
        #return Image.fromarray(img) #overlay
        logger.debug('Done')
        del overlay

        if blur:
            img = cv2.GaussianBlur(img,tuple((patch_size * (1-overlap)).astype(int) * 2 +1),0)  

        if alpha < 1.0:
            img = self.block_blending(img, vis_level, top_left, bot_right, alpha=alpha, blank_canvas=blank_canvas, block_size=1024)
        
        img = Image.fromarray(img)
        w, h = img.size

        if custom_downsample > 1:
            img = img.resize((int(w/custom_downsample), int(h/custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size/w if w > h else max_size/h
            img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))
       
        return img

    
    def block_blending(self, img, vis_level, top_left, bot_right, alpha=0.5, blank_canvas=False, block_size=1024):
        logger.debug('\ncomputing blend')
        downsample = self.level_downsamples[vis_level]
        w = img.shape[1]
        h = img.shape[0]
        block_size_x = min(block_size, w)
        block_size_y = min(block_size, h)
        logger.debug('using block size: {} x {}'.format(block_size_x, block_size_y))

        shift = top_left # amount shifted w.r.t. (0,0)
        for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
            for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):
                #logger.debug(x_start, y_start)

                # 1. convert wsi coordinates to image coordinates via shift and scale
                x_start_img = int((x_start - shift[0]) / int(downsample[0]))
                y_start_img = int((y_start - shift[1]) / int(downsample[1]))
                
                # 2. compute end points of blend tile, careful not to go over the edge of the image
                y_end_img = min(h, y_start_img+block_size_y)
                x_end_img = min(w, x_start_img+block_size_x)

                if y_end_img == y_start_img or x_end_img == x_start_img:
                    continue
                #logger.debug('start_coord: {} end_coord: {}'.format((x_start_img, y_start_img), (x_end_img, y_end_img)))
                
                # 3. fetch blend block and size
                blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img] 
                blend_block_size = (x_end_img-x_start_img, y_end_img-y_start_img)
                
                if not blank_canvas:
                    # 4. read actual wsi block as canvas block
                    pt = (x_start, y_start)
                    canvas = np.array(self.wsi.read_region(pt, vis_level, blend_block_size).convert("RGB"))     
                else:
                    # 4. OR create blank canvas block
                    canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255,255,255)))

                # 5. blend color block and canvas block
                img[y_start_img:y_end_img, x_start_img:x_end_img] = cv2.addWeighted(blend_block, alpha, canvas, 1 - alpha, 0, canvas)
        return img

    def get_seg_mask(self, region_size, scale, use_holes=False, offset=(0,0)):
        logger.debug('\ncomputing foreground tissue mask')
        tissue_mask = np.full(np.flip(region_size), 0).astype(np.uint8)
        contours_tissue = self.scaleContourDim(self.contours_tissue, scale)
        offset = tuple((np.array(offset) * np.array(scale) * -1).astype(np.int32))

        contours_holes = self.scaleHolesDim(self.holes_tissue, scale)
        contours_tissue, contours_holes = zip(*sorted(zip(contours_tissue, contours_holes), key=lambda x: cv2.contourArea(x[0]), reverse=True))
        for idx in range(len(contours_tissue)):
            cv2.drawContours(image=tissue_mask, contours=contours_tissue, contourIdx=idx, color=(1), offset=offset, thickness=-1)

            if use_holes:
                cv2.drawContours(image=tissue_mask, contours=contours_holes[idx], contourIdx=-1, color=(0), offset=offset, thickness=-1)
            # contours_holes = self._scaleContourDim(self.holes_tissue, scale, holes=True, area_thresh=area_thresh)
                
        tissue_mask = tissue_mask.astype(bool)
        logger.debug('detected {}/{} of region as tissue'.format(tissue_mask.sum(), tissue_mask.size))
        return tissue_mask


def isTissueArea(args):
    path, coord, ref_patch_size, sat_thres, area_thres = args
    wsi = openslide.open_slide(path) # Open Whole-Slide Image
    img = np.array(wsi.read_region(coord, 0, ref_patch_size))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV space
    img_med = img_hsv[:,:,1]
    img_bin = np.zeros((img_med.shape), dtype=np.uint8)
    img_bin[img_med >= sat_thres] = 255
    h,w = img_bin.shape[:2]
    tissue = np.count_nonzero(img_bin==255)
    all_pixel = h*w
    del img, img_hsv, img_med, img_bin, wsi

    if (tissue/all_pixel) > area_thres: # ?????????5%???????????????ROI???????????????
        logger.debug('tissue area over area_thres')
        return coord
    else:
        logger.debug('tissue area under area_thres')
        return None


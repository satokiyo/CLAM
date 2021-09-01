# TMU
#python3 create_patches_fp.py \
#--source_slides /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210719pdl1TMU/40xRename/PDL1/tmp \
#--source_annotations /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210719pdl1TMU/40xRename/PDL1/tmp \
#--save_dir /media/HDD2/20210331_PDL1/CLAM/20210823_0719pdl1TMU_40xRename_7class_annotation/se_resnext_all_paste \
#--patch_size 1536  \
#--step_size 1024   \
#--patch_resolution_detection 20  \
#--patch_resolution_segmentation 10  \
#--seg  \
#--patch  \
#--stitch   \
#--no_auto_skip  \
#--intensity_thres 176 \
#--area_thres 0.05 \
#--radius 16 \
#--ckpts_detection /media/prostate/20210331_PDL1/CLAM/models/nuclei_detection/ckpts/20x/0716-233238/best_model_1.pth \
#--ckpts_segmentation /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-1_deep_supervision-1_ocr-0/0823-215434/best_model_8.pth

## se_resnext no copy paste
#python3 create_patches_fp.py \
#--source_slides /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210719pdl1TMU/40xRename/PDL1/tmp \
#--source_annotations /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210719pdl1TMU/40xRename/PDL1/tmp \
#--save_dir /media/HDD2/20210331_PDL1/CLAM/20210823_0719pdl1TMU_40xRename_7class_annotation/se_resnext_no_copy_paste \
#--patch_size 1536  \
#--step_size 1024   \
#--patch_resolution_detection 20  \
#--patch_resolution_segmentation 10  \
#--seg  \
#--patch  \
#--stitch   \
#--no_auto_skip  \
#--intensity_thres 176 \
#--area_thres 0.05 \
#--radius 16 \
#--ckpts_detection /media/prostate/20210331_PDL1/CLAM/models/nuclei_detection/ckpts/20x/0716-233238/best_model_1.pth \
#--ckpts_segmentation /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-0_deep_supervision-1_ocr-0/0824-091053/best_model_3.pth


# se_resnext no copy paste 768
python3 create_patches_fp.py \
--source_slides /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210719pdl1TMU/40xRename/PDL1 \
--source_annotations /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210719pdl1TMU/40xRename/PDL1 \
--save_dir /media/HDD2/20210331_PDL1/CLAM/20210823_0719pdl1TMU_40xRename_7class_annotation/se_resnext_no_copy_paste_768_tmp \
--patch_size 1536  \
--step_size 1024   \
--patch_resolution_detection 20  \
--patch_resolution_segmentation 10  \
--seg  \
--patch  \
--stitch   \
--no_auto_skip  \
--intensity_thres 176 \
--area_thres 0.05 \
--radius 16 \
--ckpts_detection /media/prostate/20210331_PDL1/CLAM/models/nuclei_detection/ckpts/20x/0716-233238/best_model_1.pth \
--ckpts_segmentation /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-768_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-0_deep_supervision-1_ocr-0_ssl-0/0830-062514/best_model_5.pth

#--ckpts_segmentation /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-768_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-0_deep_supervision-1_ocr-0/0824-191135/best_model_5.pth



# CHN se-resnext all paste
#--source_slides /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210721pdl1CHN/40xRename/PDL1 \
#--source_annotations /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210721pdl1CHN/40xRename/0823_test \
#--save_dir /media/HDD2/20210331_PDL1/CLAM/20210823_0721pdl1CHN_40xRename_7class_annotation/se_resnext_all_paste \
#--patch_size 1536  \
#--step_size 1024   \
#--patch_resolution_detection 20  \
#--patch_resolution_segmentation 10  \
#--seg  \
#--patch  \
#--stitch   \
#--no_auto_skip  \
#--intensity_thres 176 \
#--area_thres 0.05 \
#--radius 16 \
#--ckpts_detection /media/prostate/20210331_PDL1/CLAM/models/nuclei_detection/ckpts/20x/0716-233238/best_model_1.pth \
#--ckpts_segmentation /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-1_deep_supervision-1_ocr-0/0823-215434/best_model_8.pth

# CHN se-resnext all paste
#--source_slides /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210721pdl1CHN/40xRename/PDL1 \
#--source_annotations /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210721pdl1CHN/40xRename/0823_test \
#--save_dir /media/HDD2/20210331_PDL1/CLAM/20210823_0721pdl1CHN_40xRename_7class_annotation/se_resnext_no_copy_paste \
#--patch_size 1536  \
#--step_size 1024   \
#--patch_resolution_detection 20  \
#--patch_resolution_segmentation 10  \
#--seg  \
#--patch  \
#--stitch   \
#--no_auto_skip  \
#--intensity_thres 176 \
#--area_thres 0.05 \
#--radius 16 \
#--ckpts_detection /media/prostate/20210331_PDL1/CLAM/models/nuclei_detection/ckpts/20x/0716-233238/best_model_1.pth \
#--ckpts_segmentation 


### CHN se-resnext no paste 768
##python3 create_patches_fp.py \
##--source_slides /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210721pdl1CHN/40xRename/PDL1 \
##--source_annotations /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210721pdl1CHN/40xRename/0823_test \
##--save_dir /media/HDD2/20210331_PDL1/CLAM/20210823_0721pdl1CHN_40xRename_7class_annotation/se_resnext_no_copy_paste_768 \
##--patch_size 1536  \
##--step_size 1024   \
##--patch_resolution_detection 20  \
##--patch_resolution_segmentation 10  \
##--seg  \
##--patch  \
##--stitch   \
##--no_auto_skip  \
##--intensity_thres 176 \
##--area_thres 0.05 \
##--radius 16 \
##--ckpts_detection /media/prostate/20210331_PDL1/CLAM/models/nuclei_detection/ckpts/20x/0716-233238/best_model_1.pth \
##--ckpts_segmentation /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-768_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-0_deep_supervision-1_ocr-0_ssl-0/0830-062514/best_model_5.pth

#--ckpts_segmentation /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-768_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-0_deep_supervision-1_ocr-0/0825-124534/best_model_3.pth

#--ckpts_segmentation /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-768_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-0_deep_supervision-1_ocr-0/0824-191135/best_model_5.pth


### change branch seg_overlay only , config ssl = 0
## TMU se-resnext no copy paste 512x512 SSL
#python3 create_patches_fp.py \
#--source_slides /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210719pdl1TMU/40xRename/PDL1 \
#--source_annotations /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210719pdl1TMU/40xRename/PDL1 \
#--save_dir /media/HDD2/20210331_PDL1/CLAM/20210823_0719pdl1TMU_40xRename_7class_annotation/se_resnext_no_copy_paste_512 \
#--patch_size 1536  \
#--step_size 1024   \
#--patch_resolution_detection 20  \
#--patch_resolution_segmentation 10  \
#--seg  \
#--patch  \
#--stitch   \
#--no_auto_skip  \
#--intensity_thres 176 \
#--area_thres 0.05 \
#--radius 16 \
#--ckpts_detection /media/prostate/20210331_PDL1/CLAM/models/nuclei_detection/ckpts/20x/0716-233238/best_model_1.pth \
#--ckpts_segmentation /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-0_deep_supervision-1_ocr-0_ssl-0/0827-130644/best_model_2.pth 


### calculate TPS
### change branch master , config ssl = 0
## TMU se-resnext no copy paste 512x512 
#python3 create_patches_fp.py \
#--source_slides /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210719pdl1TMU/40xRename/PDL1 \
#--source_annotations /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210719pdl1TMU/40xRename/PDL1 \
#--save_dir /media/HDD2/20210331_PDL1/CLAM/20210823_0719pdl1TMU_40xRename_7class_annotation/se_resnext_no_copy_paste_512 \
#--patch_size 1536  \
#--step_size 1024   \
#--patch_resolution_detection 20  \
#--patch_resolution_segmentation 10  \
#--seg  \
#--patch  \
#--stitch   \
#--no_auto_skip  \
#--intensity_thres 176 \
#--area_thres 0.05 \
#--radius 16 \
#--ckpts_detection /media/prostate/20210331_PDL1/CLAM/models/nuclei_detection/ckpts/20x/0716-233238/best_model_1.pth \
#--ckpts_segmentation /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-0_deep_supervision-1_ocr-0_ssl-0/0828-195028/best_model_5.pth

#--ckpts_segmentation /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-0_deep_supervision-1_ocr-0_ssl-1/0826-151549/best_model_6.pth

## CHN se-resnext no copy paste 512x512 SSL
#python3 create_patches_fp.py \
#--source_slides /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210721pdl1CHN/40xRename/PDL1 \
#--source_annotations /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210721pdl1CHN/40xRename/0823_test \
#--save_dir /media/HDD2/20210331_PDL1/CLAM/20210823_0721pdl1CHN_40xRename_7class_annotation/se_resnext_no_copy_paste_SSL \
#--patch_size 1536  \
#--step_size 1024   \
#--patch_resolution_detection 20  \
#--patch_resolution_segmentation 10  \
#--seg  \
#--patch  \
#--stitch   \
#--no_auto_skip  \
#--intensity_thres 176 \
#--area_thres 0.05 \
#--radius 16 \
#--ckpts_detection /media/prostate/20210331_PDL1/CLAM/models/nuclei_detection/ckpts/20x/0716-233238/best_model_1.pth \
#--ckpts_segmentation /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-0_deep_supervision-1_ocr-0_ssl-1/0826-151549/best_model_6.pth





# ckpts
# se_resnext
#--ckpts_segmentation /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-1_deep_supervision-1_ocr-0/0822-202500/best_model_11.pth
# se_resnext only cancer paste
#--ckpts_segmentation /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-1_deep_supervision-1_ocr-0/0823-094911/best_model_3.pth
# efficientnetv2_m
#--ckpts_segmentation /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-efficientnetv2_m_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-1_deep_supervision-1_ocr-0/0821-075128/best_model_9.pth
# efficientnetv2_l
#--ckpts_segmentation /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-efficientnetv2_l_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-1_deep_supervision-1_ocr-0/0822-145404/best_model_6.pth
#--ckpts_segmentation /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-efficientnetv2_l_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-1_deep_supervision-1_ocr-0/0823-062225/best_model_8.pth

##----------------------##

# no_auto_skip を入れないと.h5ファイルがあるだけでスキップしてしまう
#--ckpts_detection /media/prostate/20210331_PDL1/nuclei_detection/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_wot-0.1_wtv-0.7_reg-1.0_nIter-150_normCood-0/0618-235059/best_model_3.pth  \
#--ckpts_detection /media/prostate/20210331_PDL1/nuclei_detection/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_wot-0.1_wtv-0.7_reg-1.0_nIter-150_normCood-0/0713-075119/best_model_1.pth \

# x20
#--ckpts_detection /media/prostate/20210331_PDL1/CLAM/models/nuclei_detection/ckpts/20x/0716-233238/best_model_1.pth \
#--ckpts_detection /media/prostate/20210331_PDL1/CLAM/models/nuclei_detection/ckpts/20x/0716-233238/detection_model.onnx

# x40
#--ckpts_detection /media/prostate/20210331_PDL1/CLAM/models/nuclei_detection/ckpts/40x/0714-195107/best_model_6.pth

#setmentation
#x10
#--ckpts_segmentation /media/prostate/20210331_PDL1/CLAM/models/segmentation/ckpts/10x/0718-080629/best_model_1.pth
#--ckpts_segmentation /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-1_deep_supervision-1_ocr-0/0624-101754/best_model_4.pth

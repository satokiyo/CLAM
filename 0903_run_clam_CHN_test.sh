### change branch master , config ssl = 0
# CHN se-resnext no copy paste 512x512
python3 create_patches_fp.py \
--source_slides /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210721pdl1CHN/40xRename/PDL1 \
--source_annotations /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210721pdl1CHN/40xRename/0823_test \
--save_dir /media/HDD2/20210331_PDL1/CLAM/20210823_0721pdl1CHN_40xRename_7class_annotation/se_resnext_no_copy_paste_512 \
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
--ckpts_segmentation /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-0_deep_supervision-1_ocr-0_ssl-0/0828-195028/best_model_5.pth


#### change branch master , config ssl = 1
## CHN se-resnext no copy paste 512x512 SSL
#python3 create_patches_fp.py \
#--source_slides /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210721pdl1CHN/40xRename/PDL1 \
#--source_annotations /media/HDD2/20210331_PDL1/data/ndpi/2021pdl1/20210721pdl1CHN/40xRename/0823_test \
#--save_dir /media/HDD2/20210331_PDL1/CLAM/20210823_0721pdl1CHN_40xRename_7class_annotation/se_resnext_no_copy_paste_512_SSL \
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


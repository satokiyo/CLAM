python3 create_patches_fp.py \
--source /media/prostate/20210331_PDL1/data/ndpi/20210524forSatou/PDL1/tmp/tmp \
--save_dir /media/HDD2/20210331_PDL1/CLAM/result \
--patch_size 1536  \
--step_size 1024   \
--patch_resolution_detection 20  \
--patch_resolution_segmentation 10  \
--seg  \
--patch  \
--stitch   \
--no_auto_skip  \
--intensity_thres 175 \
--area_thres 0.1 \
--radius 25 \
--ckpts_detection /media/prostate/20210331_PDL1/nuclei_detection/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_wot-0.1_wtv-0.7_reg-1.0_nIter-150_normCood-0/0714-195107/best_model_6.pth \
--ckpts_segmentation /media/prostate/20210331_PDL1/segmentation/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-1_deep_supervision-1_ocr-0/0624-101754/best_model_4.pth

#--source /media/prostate/sample_ndpi/tmp/tmp \
#python3 create_patches_fp.py --source /media/prostate/20210331_PDL1/data/ndpi/20210524forSatou/PDL1/tmp --save_dir /media/prostate/20210331_PDL1/CLAM/result  --patch_size 1024 --step_size 512  --patch_resolution 10 --seg --patch --stitch  --no_auto_skip
#python3 create_patches_fp.py --source /media/prostate/20210331_PDL1/data/ndpi/20210524forSatou/PDL1/tmp --save_dir /media/prostate/20210331_PDL1/CLAM/result  --patch_size 2048 --step_size 512  --patch_level 2 --seg --patch --stitch  --no_auto_skip
#--ckpts_detection /media/prostate/20210331_PDL-1/ckpts/nuclei_detection/encoder-se_resnext50_32x4d_input-512_wot-0.1_wtv-0.7_reg-1.0_nIter-150_normCood-0/0618-235059/best_model_3.pth  \
#--ckpts_segmentation /media/prostate/20210331_PDL-1/ckpts/segmentation/encoder-se_resnext50_32x4d_input-512_downrate-1_scale_pyramid_module-1_attention_branch-0_albumentation-1_copy_paste-1_deep_supervision-1_ocr-0/0624-101754/best_model_4.pth
#--patch_size 2048  \
#--step_size 1280   \
#--save_dir /media/prostate/20210331_PDL1/CLAM/result \

# no_auto_skip を入れないと.h5ファイルがあるだけでスキップしてしまう
#--ckpts_detection /media/prostate/20210331_PDL1/nuclei_detection/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_wot-0.1_wtv-0.7_reg-1.0_nIter-150_normCood-0/0618-235059/best_model_3.pth  \
#--ckpts_detection /media/prostate/20210331_PDL1/nuclei_detection/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_wot-0.1_wtv-0.7_reg-1.0_nIter-150_normCood-0/0713-075119/best_model_1.pth \


# x40
#--ckpts_detection /media/prostate/20210331_PDL1/nuclei_detection/DM-Count/ckpts/encoder-se_resnext50_32x4d_input-512_wot-0.1_wtv-0.7_reg-1.0_nIter-150_normCood-0/0714-195107/best_model_6.pth \
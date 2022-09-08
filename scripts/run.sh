### inversion ###

# Step 1: infer encoder

# python apps/infer_hybrid_encoder.py \
#     --target_img data/ffhq/images512x512/00043/img00043506.png \
#     --g_ckpt pretrained_models/ide3d-ffhq-64-512.pkl \
#     --e_ckpt pretrained_models/encoder-base-hybrid.pkl\
#     --outdir out

# Step 2: run pti
# python inversion/scripts/run_pti.py \
#     --run_name ide3d \
#     --projector_type ide3d \
#     --viz_image \
#     --viz_mesh \
#     --viz_video \
#     --label_path data/ffhq/images512x512/dataset.json \
#     --image_name img00043506


# Step 3: finetune encoder
# python apps/finetune_hybrid_encoder.py \
#     --target_img data/ffhq/images512x512/00043/img00043506.png \
#     --target_code out_inversion/embeddings/ide3d_plus_initial_code/PTI/img00043506/0.pt \
#     --target_label out_inversion/embeddings/ide3d_plus_initial_code/PTI/img00043506/0_label.pt \
#     --g_ckpt out_inversion/checkpoints/model_ide3d_plus_initial_code_img00043506.pt \
#     --e_ckpt pretrained_models/encoder-base-hybrid.pkl \
#     --outdir out \
#     --max-steps 1000

# Step 4: run UI
# python Painter/run_ui.py \
#     --g_ckpt out_inversion/checkpoints/model_ide3d_plus_initial_code_img00043506.pt \
#     --e_ckpt out/img00043506/checkpoints/encoder-base-hybrid_tuned_img00043506.pkl \
#     --target_code out_inversion/embeddings/ide3d_plus_initial_code/PTI/img00043506/0.pt \
#     --target_label out_inversion/embeddings/ide3d_plus_initial_code/PTI/img00043506/0_label.pt \
#     --inversion


### Extract Shapes ### 
# python extract_shapes.py --network pretrained_models/ide3d-ffhq-64-512.pkl --seeds 0-3 --trunc 0.7 --outdir out --cube_size 1 

# python render_mesh.py --fname out/0.npy --outdir out


### Visualize ###
# python visualizer.py --pkls pretrained_models/ide3d-ffhq-64-512.pkl


### Interactive Editing ###
# python Painter/run_ui.py \
#     --g_ckpt pretrained_models/ide3d-ffhq-64-512.pkl \
#     --e_ckpt pretrained_models/encoder-base-hybrid.pkl 


### Animation ###
# python apps/infer_face_animation.py 
#     --drive_root D:/projects/eg3d_official/data/obama_debug/crop \
#     --network pretrained_models/ide3d-ffhq-64-512.pkl 
#     --encoder pretrained_models/encoder-base-hybrid.pkl
#     --grid 4x1 
#     --seeds 52,197,229
#     --outdir out
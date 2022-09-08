## Pretrained models paths
e4e = './pretrained_models/e4e_ffhq_encode.pt'
stylegan2_ada_ffhq = './pretrained_models/ffhq.pkl'
style_clip_pretrained_mappers = ''
ir_se50 = './pretrained_models/model_ir_se50.pth'
dlib = './pretrained_models/align.dat'
ide3d_ffhq = './pretrained_models/ide3d/network-snapshot-014480.pkl'

## Dirs for output files
checkpoints_dir = 'out_inversion/checkpoints'
embedding_base_dir = 'out_inversion/embeddings'
styleclip_output_dir = 'out_inversion/StyleCLIP_results'
experiments_output_dir = 'out_inversion/results'

## Input info
### Input dir, where the images reside
input_data_path = 'examples'
### Inversion identifier, used to keeping track of the inversion results. Both the latent code and the generator
input_data_id = 'ide3d' # 'ide3d_plus_initial_code'


## Keywords
pti_results_keyword = 'PTI'
e4e_results_keyword = 'e4e'
sg2_results_keyword = 'SG2'
sg2_plus_results_keyword = 'SG2_plus'
multi_id_model_type = 'multi_id'

## Edit directions
interfacegan_age = 'editings/interfacegan_directions/age.pt'
interfacegan_smile = 'editings/interfacegan_directions/smile.pt'
interfacegan_rotation = 'editings/interfacegan_directions/rotation.pt'
ffhq_pca = 'editings/ganspace_pca/ffhq_pca.pt'

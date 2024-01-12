
### data preparation
Initialize the COLMAP submodule:
```bash
git submodule update --init --recursive
```
First, set some environment variables:
```bash
SEQUENCE=lego
PATH_TO_VIDEO=lego.mp4
DOWNSAMPLE_RATE=2
SCENE_TYPE=object
```
where
- `SEQUENCE`: your custom name for the video sequence.
- `PATH_TO_VIDEO`: absolute/relative path to your video.
- `DOWNSAMPLE_RATE`: temporal downsampling rate of video sequence (for extracting video frames).
- `SCENE_TYPE`: can be one of ` {outdoor,indoor,object}`.

Put lego.mp4 in project dir.   
Run the following end-to-end script:
```bash
bash projects/neuralangelo/scripts/preprocess.sh ${SEQUENCE} ${PATH_TO_VIDEO} ${DOWNSAMPLE_RATE} ${SCENE_TYPE}
```
### train
python train.py --logdir=logs/lego  --config=/home/ilc/WorkPlace/neuralangelo/projects/neuralangelo/configs/custom/lego.yaml --show_pbar  --single_gpu

### isosurface extraction
python projects/neuralangelo/scripts/extract_mesh.py --config=logs/lego/config.yaml --checkpoint=logs/lego/epoch_00020_iteration_000001000_checkpoint.pt --output_file=logs/lego/mesh_1000.ply --resolution=2048 --block_res=128 --single_gpu


### another example:
DATA_PATH=datasets/1669888279_junyu-10000_body_2022-12-01
bash projects/neuralangelo/scripts/run_colmap.sh ${DATA_PATH}
python projects/neuralangelo/scripts/convert_data_to_json.py --data_dir ${DATA_PATH} --scene_type object
SEQUENCE=1669888279_junyu-10000_body_2022-12-01
python projects/neuralangelo/scripts/generate_config.py --sequence_name ${SEQUENCE} --data_dir ${DATA_PATH} --scene_type object
python train.py --logdir=logs/1669888279_junyu-10000_body_2022-12-01  --config=/home/ilc/WorkPlace/neuralangelo/projects/neuralangelo/configs/custom/1669888279_junyu-10000_body_2022-12-01.yaml --show_pbar  --single_gpu
python projects/neuralangelo/scripts/extract_mesh.py --config=logs/1669888279_junyu-10000_body_2022-12-01/config.yaml --checkpoint=logs/1669888279_junyu-10000_body_2022-12-01/epoch_00060_iteration_000002000_checkpoint.pt --output_file=logs/1669888279_junyu-10000_body_2022-12-01/mesh_epoch_60.ply --resolution=512 --block_res=64 --single_gpu
row_idx=0
while IFS=, read -a col
do
  # Identify the column names
  if [ $row_idx -eq 0 ]; then
    col_idx=0
    for c in ${col[@]}
    do
      if [ $c = "model" ]; then
        model_col=$col_idx
      elif [ $c = "dataset" ]; then
        dataset_col=$col_idx
      elif [ $c = "epochs" ]; then
        epochs_col=$col_idx
      elif [ $c = "num_gpus" ]; then
        num_gpus_col=$col_idx
      elif [ $c = "num_workers" ]; then
        num_workers_col=$col_idx
      elif [ $c = "batch_size" ]; then
        batch_size_col=$col_idx
      elif [ $c = "fp" ]; then
        fp_col=$col_idx
      elif [ $c = "transform_script" ]; then
        transform_script_col=$col_idx
      elif [ $c = "pretrained_weights" ]; then
        pretrained_weights_col=$col_idx
      fi
      col_idx=`expr $col_idx + 1`
    done
  # Row loop
  else
    # Read parameters from the columns
    model=${col[$model_col]}
    dataset=${col[$dataset_col]}
    epochs=${col[$epochs_col]}
    num_gpus=${col[$num_gpus_col]}
    num_workers=${col[$num_workers_col]}
    batch_size=${col[$batch_size_col]}
    fp=${col[$fp_col]}
    transform_script=${col[$transform_script_col]}
    pretrained_weights=${col[$pretrained_weights_col]}

    # Current date
    now=`date '+%Y%m%d%H%M%S'`

    # DETR
    if [ $model = "detr" ]; then
      result_dir="results/detr/$now"
      training_command="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus --use_env models/detr/main.py --coco_path datasets/$dataset --num_workers $num_workers --epochs $epochs --output_dir $result_dir"
      if [ -n "$pretrained_weights" ]; then
        training_command=`expr "$training_command --frozen_weights $pretrained_weights"`
      fi

    # YOLOX
    elif [ $model = "yolox" ]; then
      training_command=""
    fi
    
    # Run the training
    echo $training_command
    $training_command
  fi
  row_idx=`expr $row_idx + 1`
done < $1

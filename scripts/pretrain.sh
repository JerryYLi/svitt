#!/bin/bash
#SBATCH -p g48
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --qos=normal
#SBATCH -c 14
#SBATCH --job-name=sl_pt

export SL_EXP_DIR="output"
export SL_DATA_DIR="data"
export MASTER_PORT=20384

exp_name=$1
corpus=$2  # coco_vg, 4m, ...
exp_dir=${SL_EXP_DIR}
ngpus=$3   # number of GPUs to use, only used if ${mode} == local
mode=$4

if [[ ${corpus} != "coco_vg" ]] && [[ ${corpus} != "coco" ]] && \
  [[ ${corpus} != "webvid_cc3m" ]] && [[ ${corpus} != "cc3m" ]] && \
  [[ ${corpus} != "webvid" ]] && [[ ${corpus} != "webvid_14m" ]]; then
	echo "Does not support corpus ${corpus}"
	exit 1
fi

if [[ ${mode} != "slurm" ]] && [[ ${mode} != "local" ]]; then
	echo "Got mode=${mode}, supported mode: [slurm, local]."
	exit 1
fi

output_dir=${exp_dir}/pt_${corpus}/${exp_name}
config_path=./configs/pretrain_${corpus}.yaml
echo "output dir >> ${output_dir}"
mkdir -p ${output_dir}

############### ======> Your training scripts [START]
if [[ ${mode} == "slurm" ]]; then
	# slurm job, started with
	# sbatch THIS_SCRIPT ... slurm ...
	master_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
	all_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
	echo "All nodes used: ${all_nodes}"
	echo "Master node ${master_node}"
	# prepend MASTER_PORT=XXX when launching
	dist_url="tcp://$master_node:${MASTER_PORT:-40000}"  # default port 40000
	echo "dist_url: ${dist_url}"

	echo "PYTHONPATH: ${PYTHONPATH}"
	which_python=$(which python)
	echo "which python ${which_python}"
	export PYTHONPATH=${PYTHONPATH}:${which_python}
	export PYTHONPATH=${PYTHONPATH}:.
	echo "PYTHONPATH: ${PYTHONPATH}"

	srun \
	--output=${output_dir}/slurm%j.out \
	--error=${output_dir}/slurm%j.err \
	python \
  tasks/pretrain.py \
  ${config_path} \
  output_dir=${output_dir} \
  train_corpus=${corpus} \
  wandb.project=sb_pt_${corpus} \
  wandb.enable=True \
	dist_url=${dist_url} \
	${@:5}
elif [[ ${mode} == "local" ]]; then
  # bash THIS_SCRIPT ... local ...
  rdzv_endpoint="${HOSTNAME}:${MASTER_PORT:-40000}"
  echo "rdzv_endpoint: ${rdzv_endpoint}"

  PYTHONPATH=.:${PYTHONPATH} \
  torchrun --nnodes=1 \
  --nproc_per_node=${ngpus} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${rdzv_endpoint} \
  tasks/pretrain.py \
  ${config_path} \
  output_dir=${output_dir} \
  train_corpus=${corpus} \
  wandb.project=sb_pt_${corpus} \
  wandb.enable=True \
  ${@:5}
else
	echo "mode expects one of [local, slurm], got ${mode}."
fi
############### ======> Your training scripts [END]

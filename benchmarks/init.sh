mkdir -p wandb
echo "[default]
project: $1" > wandb/settings

mkdir -p datasets
cd dataset && python download.py

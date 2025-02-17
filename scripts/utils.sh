

srun --partition medai_llm --time=4-00:00:00 \
    python -u /mnt/petrelfs/wangpingjie/BodySound/ModifiedPretrain/test.py

srun --partition medai_llm --time=4-00:00:00 \
    python -u /mnt/petrelfs/wangpingjie/BodySound/ModifiedPretrain/tool/plot_bar.py


srun --partition medai_llm --time=4-00:00:00 --gres=gpu:1  --quotatype=auto\
    python -u /mnt/petrelfs/wangpingjie/BodySound/ModifiedPretrain/tool/explain.py

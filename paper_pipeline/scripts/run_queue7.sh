#!/usr/bin/env bash
# Hakem revizyonu: çözünürlük-uyumlaştırılmış test + yeniden eğitim
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONIOENCODING=utf8
echo "$(date) START harm_eval" >> results/queue.log
$PY harm_eval.py harm288_pairs > results/harm_eval.log 2>&1
echo "$(date) END harm_eval rc=$?" >> results/queue.log
tag=H_clip_semtmp
if [ ! -f "results/runs/$tag/summary.csv" ]; then
  rm -rf "results/runs/$tag"; echo "$(date) START $tag" >> results/queue.log
  $PY train.py --tag $tag --model tristream --data harm288_pairs --recipe proposed --sem_backbone vit_large_patch14_clip_224.openai \
     --split group --streams sem,tmp --robust > "results/stdout_$tag.txt" 2>&1
  echo "$(date) END $tag rc=$?" >> results/queue.log
fi
echo "$(date) QUEUE7 DONE" >> results/queue.log

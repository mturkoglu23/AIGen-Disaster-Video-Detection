#!/usr/bin/env bash
# Sıralı deney kuyruğu. Tamamlanan koşular (summary.csv mevcut) atlanır, böylece kesilirse yeniden başlatılabilir.
cd "$(dirname "$0")/.."
PY=${PY:-python}
run() { tag=$1; shift
  if [ -f "results/runs/$tag/summary.csv" ]; then echo "skip $tag"; return; fi
  rm -rf "results/runs/$tag"; echo "$(date) START $tag" >> results/queue.log
  $PY train.py --tag "$tag" "$@" > "results/stdout_$tag.txt" 2>&1
  echo "$(date) END $tag rc=$?" >> results/queue.log; }
# --- 1. Kullanıcı modeli: replikasyon ve kısayol/sızıntı ablasyonu
run A_orig_raw_random   --model orig --data raw_hybrid   --split random --recipe orig
run P_tristream         --model tristream --data clean_pairs --split group --recipe proposed --robust
run D_orig_clean_group_fixed --model orig --data clean_hybrid --split group --recipe fixed --robust
run C_orig_clean_group  --model orig --data clean_hybrid --split group --recipe orig --robust
run B_orig_raw_group    --model orig --data raw_hybrid   --split group --recipe orig
# --- 2. Karşılaştırma modelleri (aynı temiz protokol, grup-CV)
run Bl_r2plus1d   --model r2plus1d  --data clean_pairs --split group --recipe generic --robust
run Bl_xception   --model frame_legacy_xception --data clean_pairs --split group --recipe generic --robust
run Bl_effb0      --model frame_efficientnet_b0 --data clean_pairs --split group --recipe generic --robust
run Bl_videomae   --model videomae  --data clean_pairs --split group --recipe generic --bs 4 --robust
# --- 3. Ablasyon (önerilen model)
run Ab_sem        --model tristream --data clean_pairs --split group --recipe proposed --streams sem
run Ab_for        --model tristream --data clean_pairs --split group --recipe proposed --streams for
run Ab_tmp        --model tristream --data clean_pairs --split group --recipe proposed --streams tmp
run Ab_sem_for    --model tristream --data clean_pairs --split group --recipe proposed --streams sem,for
run Ab_sem_tmp    --model tristream --data clean_pairs --split group --recipe proposed --streams sem,tmp
run Ab_for_tmp    --model tristream --data clean_pairs --split group --recipe proposed --streams for,tmp
run Ab_noaux_nodrop --model tristream --data clean_pairs --split group --recipe proposed --aux_w 0 --stream_drop 0
run Ab_nojpeg     --model tristream --data clean_pairs --split group --recipe proposed --no_jpeg_aug --robust
# --- 4. Afet türleri arası genelleme (leave-one-disaster-out)
run L_tristream   --model tristream --data clean_pairs  --split lodo --recipe proposed
run L_orig_fixed  --model orig      --data clean_hybrid --split lodo --recipe fixed
echo "$(date) QUEUE DONE" >> results/queue.log

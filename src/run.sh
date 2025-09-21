export PATH=$HOME/java/bin:$PATH
# python bench_xgb.py --experiment-type h2oxgboost --device gpu --n-runs 105 --threads 16 --out-dir logs

python bench_xgb.py --experiment-type pyxgboost --device cpu --n-runs 105 --threads 16 --out-dir logs
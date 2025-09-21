export PATH=$HOME/java/bin:$PATH
python bench_xgb.py --experiment-type h2oxgboost --device cpu --n-runs 105 --threads 24 --out-dir logs

# python bench_xgb.py --experiment-type pyxgboost --device cpu --n-runs 105 --threads 16 --out-dir logs
if [ "$1" = "train" ]; then
    echo "Start Training...."
    python src/train.py
elif  [ "$1" = "predict" ]; then
    echo "Start Inference...."
    python src/predict.py
elif  [ "$1" = "test" ]; then
    pytest --cov -p no:warnings
elif  [ "$1" = "lint" ]; then
    # python lint.py -m lint
    autopep8 src/ --recursive --in-place --aggressive --aggressive --ignore=E402,E226,E24,W50,W690
fi
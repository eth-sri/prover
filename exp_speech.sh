python test_speech.py --dataset="FSDD" --db=-90 --bound_method="lp" --seed=1000 --model_dir="saved/fsdd.pt"
python test_speech.py --dataset="FSDD" --db=-80 --bound_method="lp" --seed=1000 --model_dir="saved/fsdd.pt"
python test_speech.py --dataset="FSDD" --db=-70 --bound_method="lp" --seed=1000 --model_dir="saved/fsdd.pt"
python test_speech.py --dataset="FSDD" --db=-90 --bound_method="opt" --seed=1000 --model_dir="saved/fsdd.pt"
python test_speech.py --dataset="FSDD" --db=-80 --bound_method="opt" --seed=1000 --model_dir="saved/fsdd.pt"
python test_speech.py --dataset="FSDD" --db=-70 --bound_method="opt" --seed=1000 --model_dir="saved/fsdd.pt"

python test_speech.py --dataset="GSC" --db=-110 --bound_method="lp" --seed=1000 --model_dir="saved/gsc.pt"
python test_speech.py --dataset="GSC" --db=-100 --bound_method="lp" --seed=1000 --model_dir="saved/gsc.pt"
python test_speech.py --dataset="GSC" --db=-90 --bound_method="lp" --seed=1000 --model_dir="saved/gsc.pt"
python test_speech.py --dataset="GSC" --db=-110 --bound_method="opt" --seed=1000 --model_dir="saved/gsc.pt"
python test_speech.py --dataset="GSC" --db=-100 --bound_method="opt" --seed=1000 --model_dir="saved/gsc.pt"
python test_speech.py --dataset="GSC" --db=-90 --bound_method="opt" --seed=1000 --model_dir="saved/gsc.pt"

python test_mnist.py --nframes=4 --nhidden=32 --nlayers=1 --eps=0.01 --bound_method="lp" --seed=1000 --model_dir="saved/mnist_04f_032h_1l.pt"
python test_mnist.py --nframes=4 --nhidden=32 --nlayers=2 --eps=0.01 --bound_method="lp" --seed=1000 --model_dir="saved/mnist_04f_032h_2l.pt"
python test_mnist.py --nframes=4 --nhidden=32 --nlayers=3 --eps=0.01 --bound_method="lp" --seed=1000 --model_dir="saved/mnist_04f_032h_3l.pt"
python test_mnist.py --nframes=4 --nhidden=64 --nlayers=1 --eps=0.01 --bound_method="lp" --seed=1000 --model_dir="saved/mnist_04f_064h_1l.pt"
python test_mnist.py --nframes=4 --nhidden=128 --nlayers=1 --eps=0.01 --bound_method="lp" --seed=1000 --model_dir="saved/mnist_04f_128h_1l.pt"
python test_mnist.py --nframes=7 --nhidden=32 --nlayers=1 --eps=0.01 --bound_method="lp" --seed=1000 --model_dir="saved/mnist_07f_032h_1l.pt"
python test_mnist.py --nframes=4 --nhidden=32 --nlayers=1 --eps=0.01 --bound_method="opt" --seed=1000 --model_dir="saved/mnist_04f_032h_1l.pt"
python test_mnist.py --nframes=4 --nhidden=32 --nlayers=2 --eps=0.01 --bound_method="opt" --seed=1000 --model_dir="saved/mnist_04f_032h_2l.pt"
python test_mnist.py --nframes=4 --nhidden=32 --nlayers=3 --eps=0.01 --bound_method="opt" --seed=1000 --model_dir="saved/mnist_04f_032h_3l.pt"
python test_mnist.py --nframes=4 --nhidden=64 --nlayers=1 --eps=0.01 --bound_method="opt" --seed=1000 --model_dir="saved/mnist_04f_064h_1l.pt"
python test_mnist.py --nframes=4 --nhidden=128 --nlayers=1 --eps=0.01 --bound_method="opt" --seed=1000 --model_dir="saved/mnist_04f_128h_1l.pt"
python test_mnist.py --nframes=7 --nhidden=32 --nlayers=1 --eps=0.01 --bound_method="opt" --seed=1000 --model_dir="saved/mnist_07f_032h_1l.pt"
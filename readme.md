# PROVER - Scalable Polyhedral Verification of Recurrent Neural Networks

This repository contains the source codes and demonstrative scripts for the CAV2021 accepted paper, "Scalable Polyhedral Verification of Recurrent Neural Networks". It is available to download the artefact (an Ubuntu virtual machine ready to run) on [Zenodo](https://zenodo.org/record/4742650#.YNc0KxMzZpQ) as well.


## Preparation

### System Requirement

* Running the experiments on GPU is highly recommended. Estimated running times reported here are based on the setting with a single Tesla V100.
* The framework relies on the [Gurobi optimiser](https://www.gurobi.com/) version 9.1, if you are testing PROVER for academic purpose, you can follow the Gurobi's insturction of academic licence [here](https://www.gurobi.com/academia/academic-program-and-licenses/). Make sure to activate the Gurobi license for version 9.1 before you run the experiments.
* [Conda](https://docs.conda.io/en/latest/) environment is recommended to easily install the dependencies. 
  

### Virtual Environment Setup

`requirements.txt` is the frozen conda list of the environment. You can setup the necessary virtual environment by:
```
conda create --name prover --file requirements.txt --channel gurobi --channel pytorch
conda activate prover
```

After you setup the virtual environment, you need to activate your Gurobi licence.

The paper demonstrates total of four datasets: MNIST, FSDD, GSC, and HAPT. MNIST will be downloaded by PyTorch during the training or the test, but you will need to download other datasets. Run `data_preparation.sh` or follow the below instructions:
```
mkdir data
cd data

git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git

wget http://archive.ics.uci.edu/ml/machine-learning-databases/00341/HAPT%20Data%20Set.zip
mkdir HAPT
unzip HAPT\ Data\ Set.zip -d HAPT

wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir gsc
tar -zxf speech_commands_v0.02.tar.gz --directory gsc

rm *.zip *.tar.gz
cd ..
```


## Main Codes

1. `R2.py`: The core code of PROVER. You can see each abstract transformer class within the code.
2. `relaxation.py`: The code majorly for the LP-based relaxation of the sigmoid X tanh and sigmoid x identity operation.
3. `utils.py`: Miscellaneous methods needed throughout the framework.
4. `*_loader.py`: Dataset-specific loader class.
5. `models_*.py`: Dataset-specific classifier class. There are two different sample architectures: with and without the speech preprocessing modules, `models_mnist.py` and `models_speech.py`, respectively. Note that there always need to be two model classes for the experiment: `torch.nn` based network and `R2` based network with the same architecture.
6. `train_*.py`: Codes that train the classifier model for each dataset. Detailed instruction to run those codes are below.
7. `test_*.py`: Codes to test the robustness of the classifier. Detailed instruction to run those codes are below.


### Supporting operations

We implemented the operations supported by PROVER in `R2.py`. The list below shows the supporting operations, including LSTM. The module names are self-explanative.

* `R2.LSTMCell(input_size, hidden_size, num_layers, prev_layer, prev_cell, method, device)`
  * `input_size`: The dimension of the input to the first LSTM layer.
  * `hidden_size`: The dimension of the hidden state of all LSTM layers.
  * `num_layers`: The number of LSTM layers.
  * `prev_layer`: R2 object placed right before the LSTM layer.
  * `prev_cell`: `None` or R2.LSTMCell handling the previous frame.
  * `method`: The bounding method. Either "lp" or "opt".
  * `device`: The device specification of where the data should be loaded.
* `R2.Linear(in_features, out_features, bias, prev_layer)`
  * `in_features`: The input dimension of the fully connected layer.
  * `out_features`: The output dimension.
  * `bias`: Boolean tag of whether the layer contains bias.
  * `prev_layer`: R2 object placed right before the LSTM layer.
* `R2.Square(prev_layer)`, `R2.Log(prev_layer)`
  * `prev_layer`: R2 object placed right before the LSTM layer.
* `R2.ReLU(inplace, prev_layer)`
  * `inplace`: Boolean tag of whether the ReLU works inplace. 
  * `prev_layer`: R2 object placed right before the LSTM layer.
* `R2.Sigmoidal(func, prev_layer)`
  * `func`: Either "sigmoid" or "tanh".
  * `prev_layer`: R2 object placed right before the LSTM layer.

Also, the core object, `R2.DeepPoly`, is defined as below. 

* `R2.DeepPoly(lb, ub, lexpr, uexpr, device)`
  * `lb`: `torch.Tensor` of the concrete lower bounds of the given DeepPoly abstraction.
  * `ub`: `torch.Tensor` of the concrete upper bounds of the given DeepPoly abstraction.
  * `lexpr`: Linear coefficients to build relative expressions with the previous layer's neurons for the lower bound.
  * `uexpr`: Linear coefficients to build relative expressions with the previous layer's neurons for the upper bound.
  * `device`: Where the values will be stored.
  * `deeppoly_from_perturbation(x, eps, truncate)`: Returns the concretly bounded DeepPoly object on input `x` with the perturbation `eps`. If `truncate` is not `None`, then truncate the min and max bounds to `truncate[0]` and `truncate[1]`, respectively.
  * `deeppoly_from_dB_perturbation(x, eps_db)`: Returns the concretly bounded DeepPolly object on input `x` with the dB perturbation `eps_db`.
  
The current implementation needs the explicit specification of the previous layer/cell's objects. Please refer the `models_*.py` for the detailed implementation of how to build the PROVER models. 


## Reproducing Results

We highly recommend to utilise GPU to reproduce the results, especially for `--bound_method="opt"` option. All the models are stored under `saved/` directory. For the detailed instruction for the self-training of the models, please refer the next section.

You can verify either the existing model or the newly trained model. Please refer the later section for training your own model. Pretrained models provided by the authors are found in `saved/` directory. Currently available models are:

* `fsdd.pt`: the model used for FSDD classification.
* `gsc.pt`: the model used for GSC classification.
* `hapt_(l)L_(h)H.pt`: HAPT classifier with l LSTM layers with h hidden dimension.
* `mnist_(f)F_(h)H_(l)L.pt`: MNIST classifier with l LSTM layers with h hidden dimension. MNIST demonstration is with fixed number of frames f.


### 7.1 Speech classification

We provide two reproducible experiments, shown on Table 1 and Fig. 8. To reproduce the results on Fig. 6, run `exp_speech.sh`. The script will run the experiments of verifying robustness of FSDD and GSC classifier on different perturbation decibels. The results of the experiments will be saved at a csv file `results/exp_speech.csv`. For each row, the first value shows the dataset, either FSDD or GSC, the second is the bounding method, either lp or opt, and the third is the perturbation decibel, the next is the proportion of provable examples, and the last one is the average running time.

You can also do your own single test by directly run the `test_speech.py`. By changing the `--seed` from the value in the script, you will get results from the different subset of the test data.
```
python test_speech.py [-h] [--dataset {FSDD,GSC}] [--db DB] [--bound_method {lp,opt}] [--model_dir MODEL_DIR] [--seed SEED] [--verbose]
```
The result will be appended to the same csv file mentioned above.

You may run the subset of the experiments with `exp_speech_subset.sh` to see the result within the time. This will run the first two lp experiments on FSDD with perturbation decibel -90 and -80. This will add the results at the csv file in 2.5 hours.


### 7.2 Image classification

To get the result of Table 1, run `exp_scalability.sh`. The script will run the experiments of showing the results of various architectures described on the paper. The results of each architecture will be stacked in the several csv files indicating the architecture, e.g., `results/exp_mnist_4f_32h_1l.csv`.

Also, to get the result of Fig. 8, run `exp_mnist.sh`. As the experiment uses only a single architecture, the result will be stored in `results/exp_mnist_4f_32h_2l.csv`. The first value of each row shows the bounding method, the second is the pertrubation epsilon, the next one is the certified precision, and the last is the average running time. 

You can also do your own single test by running `test_mnist.py`. By changing the `--seed` from the value in the script, you will get results from the different subset of the test data.
```
python test_mnist.py [-h] [--nframes {4,7,14,28}] [--nhidden NHIDDEN] [--nlayers NLAYERS] [--eps EPS] [--bound_method {lp,opt}] [--model_dir MODEL_DIR] [--seed SEED] [--verbose]
```
The output format will be the same as before.

`exp_scalability.sh` takes 5.5 hours and `exp_mnist.sh` takes 4.5 hours on GPU, so to see the results in time, run `exp_mnist_subset.sh`. This script will run lp and opt experiment with perturbation epsilon 0.013, shown on fig.8, and will produce the results in an hour.


### 7.3 Motion sensor data classification

We also provide the experiments with the additional dataset named HAPT. To get the result of Fig. 9, run `exp_hapt.sh`. The script will run the experiments of verifying robustness of HAPT classifier on different perturbation budgets. The results of the experiments will be saved at a csv file `results/exp_hapt.csv`. Each row contains bounding method, perturbation epsilon, certified precision, and the average running time, respectively.

You can also run the below `test_hapt.py`, and you can get the results of the motion classifier from the sequential triaxial sensor data.
```
test_hapt.py [-h] [--nhidden NHIDDEN] [--nlayers NLAYERS] [--eps EPS] [--bound_method {lp,opt}] [--model_dir MODEL_DIR] [--seed SEED] [--verbose]
```


### How to interpret the verification progress

During the verification, the standard output will show the progress visually.
You will see the progress with this format:
```
[Testing Input #XXX (Y proven / Z correct)]
```
Here, *XXX* stands for the identifier of the test input, shuffled by the random seed provided before. The proven count *Y* means the number of inputs that are guaranteed to be robust by PROVER. Correct count *Z* is the number of correctly predicted examples until *XXX*-th input. Since our certified proportion is calculated upon the only correct inputs, you may assume the current provability as *Y/Z x 100*%. When *Z* goes 100, the verifier terminates its job and write the result in the designated file.

For each example being tested, the verifier will print *[PROVEN]* or *[FAILED]*.
PROVEN is printed when the input with the given perturbation measure is guaranteed to produce the same result. FAILED is printed, under the same condition, when PROVER claims there might be an adversarial example. The single test is iterated over the number of labels of the dataset, so to completely guarantee the robustness on the example, it should not include any failed label comparison.

You can see more detailed information by providing `--verbose` option to each run. You then will be able to check which false label is not guaranteed to be robust, and the progress of the iterations from PROVER's opt method. 


## Training your own model (optional)

1. Train the MNIST classifier
```
python train_mnist.py {num_frames} {num_hidden} {num_layers} {model_name}
```
The model will be saved in `saved/{model_name}`.

2. Train the FSDD classifier
```
python train_fsdd.py
```
The model will be saved in `saved/fsdd.pt`.

3. Train the GSC classifier
```
python train_gsc.py
```
The model will be saved in `saved/gsc.pt`.

4. Train the HAPT classifier
```
python train_hapt.py
```
The model will be saved in `saved/hapt_4L_256H.pt`.

You can adjust the hyperparameters of each model by modifying above mentioned codes.

After training your own models, you can follow the instructions of the above section to test the model's robustness.


## Use your own benchmarks

The current version of PROVER does not support the native extension for the new benchmarks, but not complicated to do so. In order to apply your own benchmarks other than the artifact provides, you will need:
* your custom `torch.utils.data.DataLoader`, or any comparable iterative data loader for the new benchmark, and
* (optional) new PROVER-format model definition.


### Building new data loader

Following list is the requirements for the new data loader:
1. Each iteration yields `(X, y)`, where `X` is the input sequence of shape `(batch size, sequence length)` and `y` is the integer labels of shape `(batch size)`.
2. Support shuffle switch option for both training and test sessions.

We recommend to make `torch.utils.data.DataSet` and wrap the object via `DataLoader` with various options.


### Verifying models on the new benchmarks

In the artifact, `models_mnist.py` and `models_speech.py` include the essential models. Please refer the following snippets to see how the architectures are implemented:

**models_mnist.py**
```
class MnistModel():
    description:
        This is the base classifier model using the vanilla LSTM module followed by a fully connected layer.

    architecture:
        torch.nn.unfold()
        torch.nn.LSTM()
        torch.nn.Linear()

    initializer parameters:
        in_size: The input length of each frame. This is equivalent to the dimension of i_t of the first LSTM layer.
        hidden_size: The hidden dimension size of LSTM cells. This is equiavlent to the dimension of h_t.
        num_layers: The number of layers of LSTM.
        out_size: The output dimension of the model. This must match with the number of classes of your new benchmark.
    
    methods:
        forward(x): torch.Tensor -> torch.Tensor
            Inherited from torch.nn.Module.

class MnistModelDP():
    description:
        This is the DeepPoly handling version of the model and contains verification method. This class inherits
        MnistModel so can be directly load the state_dict from its result. As the current implementation supports
        frame-wise propagation, the architecture needs to be unrolled.

    architecture:
        for each frame:
            R2.LSTMCell()
        R2.Linear()
    
    initializer parameters:
        in_size: The input length of each frame. This is equivalent to the dimension of i_t of the first LSTM layer.
        hidden_size: The hidden dimension size of LSTM cells. This is equiavlent to the dimension of h_t.
        num_layers: The number of layers of LSTM.
        out_size: The output dimension of the model. This must match with the number of classes of your new benchmark.
    
    methods:
        set_bound_method(bound_method): str -> None
            Setting bounding method byeither "lp" or "opt".
        certify(input, gt, max_iter, verbose): R2.DeepPoly, torch.Tensor, int, bool -> bool
            Certify the model whehter it is robust to the perturbed input in DeepPoly format for the ground truth label gt.
            You can adjust max_iter and verbose to customize the results.
```

**models_speech.py**
```
class SpeechClassifier():
    description:
        This class defines the base speech classifier consists of the speech preprocessing stages and LSTM layers.

    architecture:
        torch.nn.unfold()
        torch.matmul()
        torch.pow()
        torch.matmul()
        torch.log() # // speech preprocessing stages
        torch.nn.Linear()
        torch.nn.ReLU()
        torch.nn.LSTM()
        torch.nn.Linear()
    
    initializer parameters:
        frame_size: The input length of each frame.
        frame_step: The stride size between two adjacent frames.
        n_filt: The number of Mel-frequency filters.
        hidden_dim: The output dimension of the first fully connected layer. This is equivalent to the dimension of i_t
            of the first LSTM layer.
        hiddem_dim2: The hidden dimension size of LSTM cells. This is equiavlent to the dimension of h_t.
        num_layers: The number of layers of LSTM.
        out_dim: The output dimension of the model. This must match with the number of classes of your new benchmark.

    methods:
        forward(x): torch.Tensor -> torch.Tensor
            Inherited from torch.nn.Module.

class SpeechClassifierDP():
    description:
        This is the DeepPoly handling version of the model and contains verification method. This class inherits
        MnistModel so can be directly load the state_dict from its result. As the current implementation supports
        frame-wise propagation, the architecture needs to be unrolled.

    architecture:
        for each frame:
            R2.Linear()
            R2.square()
            R2.Linear()
            R2.Log() # // speech preprocessing stages
            R2.Linear()
            R2.ReLU()
            R2.LSTMCell()
        R2.Linear()
    
    initializer parameters:
        frame_size: The input length of each frame.
        frame_step: The stride size between two adjacent frames.
        n_filt: The number of Mel-frequency filters.
        hidden_dim: The output dimension of the first fully connected layer. This is equivalent to the dimension of i_t
            of the first LSTM layer.
        hiddem_dim2: The hidden dimension size of LSTM cells. This is equiavlent to the dimension of h_t.
        num_layers: The number of layers of LSTM.
        out_dim: The output dimension of the model. This must match with the number of classes of your new benchmark.

    methods:
        set_bound_method(bound_method): str -> None
            Setting bounding method byeither "lp" or "opt".
        certify(input, gt, max_iter, verbose): R2.DeepPoly, torch.Tensor, int, bool -> bool
            Certify the model whehter it is robust to the perturbed input in DeepPoly format for the ground truth label gt.
            You can adjust max_iter and verbose to customize the results.
```

We recommend you to adjust hyperparameters on the existing models and edit the test code.
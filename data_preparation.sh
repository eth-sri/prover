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

# Green AutoML Benchmark

AutoML has risen to one of the most common tools for day-to-day data science pipeline development and several popular prototypes exist. While AutoML systems support data scientists during the tedious process of pipeline generation, it can lead to high computation costs that result from extensive search or pre-training. In light of concerns with regard to the environment and the need for Green IT, we want to holistically analyze the computational cost of pipelines generated through various AutoML systems by combining the cost of system development, execution, and the downstream inference cost. We summarize our findings that show the benefits and disadvantages of implementation designs and their potential for Green AutoML.  


## Setup
```
conda create -n GreenAutoML python=3.7
conda activate GreenAutoML
cd Software/GreenAutoML/
git pull origin main
python -m pip install codecarbon
python -m pip install tabpfn
sudo chmod -R a+r /sys/class/powercap/intel-rapl
sh setup.sh
```

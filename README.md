# Green AutoML

AutoML has risen to one of the most common tools for day-to-day data science pipeline development and several popular prototypes exist. While AutoML systems support data scientists during the tedious process of pipeline generation, it does so under heavy computation costs that result from extensive search or pre-training. In light of concerns with regard to the environment and the desire for Green IT, we want to holistically analyze the computational cost of pipelines generated through various AutoML systems by combining the cost during search and the downstream inference cost. We summarize our findings that show the benefits and disadvantages of implementation designs and their potential for Green AutoML.  

![image](https://user-images.githubusercontent.com/5217389/216223724-05dd746d-4cce-4e64-869e-b791cfe7cee2.png)

Credit Stable Diffusion

## Setup
```
conda create -n AutoMLD python=3.7
conda activate AutoMLD
cd Software/DeclarativeAutoML/
git pull origin main
python -m pip install codecarbon
python -m pip install tabpfn
sudo chmod -R a+r /sys/class/powercap/intel-rapl
sh setup.sh
```

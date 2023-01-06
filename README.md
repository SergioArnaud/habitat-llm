# habitat-llm


## Setup
Conda

setup
```
# create a conda image 
conda create -n habitat-llm  python=3.8.1 cmake=3.14.0
conda activate habitat-llm
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat

# Install habitat
https://github.com/facebookresearch/habitat-lab.git --branch v0.2.3
cd habitat-lab
pip install -e habitat-baselines   
pip install -e habitat-lab
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets --data-path data  

# Other dependencies
conda install -c conda-forge wandb
conda install -c conda-forge ipywidgets

# Language
pip install langchain
pip install openai
pip install regex


```


## Adding new skills
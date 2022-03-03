#### **Instructions for Linux.**

Make sure that the system has Python 3.7 installed. For matplotlib you also 
need something like e.g. `sudo apt-get install python3.7-dev` 

The following instructions have not been tested with conda, but
with `virtualenv` in a system that has multiple versions of
Python available. The following does not assume that the
system-wide version of Python is 3.7.

The first step is to git clone the repository.

#### In the root of the repository
Make sure you have the `virtualenv` package installed and call
`virtualenv venvdqd --python=python3.7`

#### Activate the environment
`source venvdqd/bin/activate`

#### Install the required packages
`pip install -r requirements.txt`

#### Configure the ipython kernel
`ipython kernel install --user --name=venvdqd`
Afterwards make sure to select the `venvdqd` from the 
Kernel->Change Kernel menu of the notebook

#### In case you want to start jupyter notebooks
`python -m notebooks`

#### In case you want to run code
Convert a notebook to `.py` file
For example, in `example_notebooks` run
`jupyter nbconvert --to script 3\)\ Testing\ Example.ipynb`\
and then 
`python 3\)\ Testing\ Example.py`


#### **Instructions for WSL**.


Make sure that the system has Python 3.7 installed. For matplotlib you also 
need something like e.g. `sudo apt-get install python3.7-dev` 

#### To install python3.7 in Ubuntu 20.04 :

`sudo add-apt-repository ppa:deadsnakes/ppa` 

`sudo apt-get update`

`sudo apt-get install python3.7`


We assume you have cloned the git repository in Windows folder somewhere. 

#### To create a virtual environment
Make sure you have the `venv` package installed or type 
`sudo apt install python3.7 python3.7-venv`

Go to the directory in Ubuntu where you want to create the virtual environment.
`python3.7 -m venv venvdqd`

#### Activate the environment
`source venvdqd/bin/activate`

#### Install the required packages
Now within the ubuntu terminal move to the Windows folder containing the project files and run.

`pip install -r requirements.txt`

#### Configure the ipython kernel
`ipython kernel install --user --name=venvdqd`
Afterwards make sure to select the `venvdqd` from the 
Kernel->Change Kernel menu of the notebook

#### In case you want to start jupyter notebooks
`python -m notebooks`

#### In case you want to run code
Convert a notebook to `.py` file
For example, in `example_notebooks` run
`jupyter nbconvert --to script 3\)\ Testing\ Example.ipynb`\
and then 
`python 3\)\ Testing\ Example.py`

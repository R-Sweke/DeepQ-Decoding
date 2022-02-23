Make sure that the system has Python 3.7 installed.
The following instructions have not been tested with conda, but
with `virtualenv` in a system that has multiple versions of
Python available. The following does not assume that the
system-wide version of Python is 3.7.

The first step is to git clone the repository.

####In the root of the repository
Make sure you have the `virtualenv` package installed and call
`virtualenv venvdqd --python=python3.7`

####Activate the environment
`source venvdqd/bin/activate`

####Install the required packages
`pip install -r requirements.txt`

####Configure the ipython kernel
`ipython kernel install --user --name=venvdqd`
Afterwards make sure to select the `venvdqd` from the 
Kernel->Change Kernel menu of the notebook

####In case you want to start jupyter notebooks
`python -m notebooks`

####In case you want to run code
Convert a notebook to `.py` file
For example, in `example_notebooks` run
`jupyter nbconvert --to script 3\)\ Testing\ Example.ipynb`\
and then 
`python 3\)\ Testing\ Example.py`


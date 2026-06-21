module purge
# Set this to a suitable openmpi distribution on your local machine:
module load openmpi/5.0.8

export TMPDIR=~/tmp
mkdir -p $TMPDIR

# Be sure to update the Python path below according to your machine setup:
/path/to/my/python -m venv ~/IRIS/iris_venv
source ~/IRIS/iris_venv/bin/activate

python -m pip install -U pip
# For basic installation:
# python -m pip install --pre -e ~/IRIS
# For installation with optional CuPy features:
python -m pip install --pre -e ~/IRIS[cupy]

rm -r $TMPDIR
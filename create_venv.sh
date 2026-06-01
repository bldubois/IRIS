module purge
module load openmpi/5.0.8

export TMPDIR=~/tmp
mkdir -p $TMPDIR

/path/to/my/python -m venv ~/IRIS/iris_venv
source ~/IRIS/iris_venv/bin/activate

python -m pip install -U pip
python -m pip install --pre -e ~/IRIS[cupy]

rm -r $TMPDIR
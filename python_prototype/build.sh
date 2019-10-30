# Run this script in QMCSoftware/qmcpy
#
# Run the following once:
#    chmod u+x ./build.sh
#
# Run the following every time you want to rebuild Qmcpy:
#
#    conda activate pyqmc  # activate your python virtual environment
#    ./build.sh &> outputs/qmcpy_build.log &


echo "QMCPy build process starts..."

echo "$(date)"

# autopep8
autopep8 . --in-place --recursive --ignore E402,E701,E501,E128,C0413 --exclude ./qmcpy/third_party/magic_point_shop
# pylint
pylint --variable-rgx="[a-z0-9_]{1,30}$" --disable W0622,C0103,C0321 **/*.py
pylint --variable-rgx="[a-z0-9_]{1,30}$" --disable W0622,C0103 **/**/*.py

# Uninstall and Install Qmcpy
pip uninstall --yes qmcpy
python setup.py install
python setup.py clean

# Run unit tests
pythonw -m unittest discover -s test/fasttests
pythonw -m unittest discover -s test/longtests

# Use sphinx to generate HTML documentation from above and Python docstrings
cd sphinx
./autodoc.sh
cd ..

# git commands

git add -f ../docs

# Check time stamps
ls -ltr .

ls -ltr outputs

ls -ltr ../docs/index.html

echo "QMCPy build process is completed."
echo "$(date)"




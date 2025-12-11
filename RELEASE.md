# QMCPy Release Documentation

The following is a guide for those in charge of making new releases of QMCPy.

## Setting Up PyPi Account

In order to publish a release, you will need to have an account on the two websites:

- https://pypi.org
- https://test.pypi.org

Note: These accounts will be different from each other.

### Getting Your API Keys



## Procedure

### Step 1

1. Pull latest changes from the `develop` branch.
2. Run `pip install -e .` to update any packages.
3. Run `make doctests_no_docker` to run tests.

### Step 2

1. In the file called `__init__.py` in the `qmcpy` directory, go to the end of the file and change the `__version__` attribute to the new release number.

Run Step 1 again as a sanity check.

### Step 3 

In the file `pyproject.toml`, there are a number of build systems to use, we will use PDM.

### Step 4

To make sure everything was done correctly, 

### Step 5

After uploading the release to test.pypi.org, create a new environment with Anaconda and download the package using `pip`.

### Step 6

After testing, you can run `pdm publish` to publish to PyPi

### Step 7

Once done, commit and push your changes.

### Step 8

After publishing, go to GitHub, create a pull request to merge `develop` branch to `master`.

Do one final sanity check on all the files before confirming the pull request.

### Step 9

Create a release on the GitHub repository but going to Releases, make a draft of a new release. Give it a tag, a release title, and a description.

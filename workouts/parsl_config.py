import parsl
from parsl.app.app import python_app

# Define the Parsl configuration
parsl.load(parsl.config.Config(
    executors=[
        parsl.executors.ThreadPoolExecutor(max_threads=4)
    ]
))


# Define the Parsl app for generating samples
@python_app
def generate_samples(gbm, n):
    return gbm.gen_samples(n)
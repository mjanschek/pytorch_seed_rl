import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-seed-rl", 
    version="0.0.1",
    author="Michael Janschek",
    author_email="michael.janschek@hhu.de",
    description="A PyTorch Implementation of SEED-RL, originally created by Google Brain for TensorFlow 2.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mjanschek/pytorch_seed_rl",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
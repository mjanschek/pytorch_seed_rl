import setuptools

INSTALL_REQUIRES = [
    'torch>=1.5',
    'gym[atari]>=0.17.2',
    'torchsummary>=1.5.1'
]

TEST_REQUIRES = [
    # testing and coverage
    'pytest', 'coverage', 'pytest-cov',
    # unmandatory dependencies of the package itself
    
    # to be able to run `python setup.py checkdocs`
    'collective.checkdocs', 'pygments',
]

DOCS_REQUIRES = [
    'sphinx>=2.0'
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch_seed_rl", 
    license="Apache 2.0",
    version="0.0.1",
    author="Michael Janschek",
    author_email="michael.janschek@hhu.de",
    description="PyTorch Implementation of SEED-RL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mjanschek/pytorch_seed_rl",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'test': TEST_REQUIRES + INSTALL_REQUIRES,
        'docs': DOCS_REQUIRES + TEST_REQUIRES + INSTALL_REQUIRES,
    },
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="geneeval",
    version="0.1.0rc1",
    author="John Giorgi",
    author_email="johnmgiorgi@gmail.com",
    description=("GeneEval"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BaderLab/GeneEval",
    packages=setuptools.find_packages(),
    keywords=["gene embeddings",],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
    ],
    python_requires=">=3.6.1",
    install_requires=["skorch>=0.8.0"],
    extras_require={
        "dev": ["black", "flake8", "hypothesis", "pytest", "pytest-cov", "coverage", "codecov",]
    },
)

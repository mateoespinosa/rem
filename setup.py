import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="dnn_rem",  # Replace with your own username
    version="0.0.1",
    author="Artificial Intelligence Group - University of Cambridge",
    author_email="zs315@cam.ac.uk",
    description="Deep neural network rule extraction methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mateoespinosa/rem",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6, <3.9',
    install_requires=[
        "Keras-Preprocessing==1.1.2",
        "numpy<1.19.0",
        "pandas==0.25.3",
        "prettytable==1.0.1",
        "PyYAML==5.3.1",
        "rpy2==3.3.6",
        "scikit-learn==0.23.2",
        "scipy==1.5.3",
        "sklearn==0.0",
        "tensorboard==2.3.0",
        "tensorboard-plugin-wit==1.7.0",
        "tensorflow==2.3.1",
        "tensorflow-estimator==2.3.0",
        "tqdm==4.51.0",
    ],
)

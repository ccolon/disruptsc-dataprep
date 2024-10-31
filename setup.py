from setuptools import setup, find_packages

setup(
    name="disruptsc-dataprep",
    version="1.0.1",
    description="Tools to help prepare data for the DisruptSC model.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Celian Colon",
    author_email="celian.colon.2007@polytechnique.org",
    url="https://github.com/ccolon/disruptsc-dataprep",
    packages=find_packages(),
    license="CC BY-NC-ND 4.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "gadm==0.0.5"
    ],
)

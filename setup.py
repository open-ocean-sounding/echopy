import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "echopy",
    version = "0.0.1",
    author = "British Antarctic Survey",
    author_email = "rapidkrill@bas.ac.uk",
    description= "Unsupervised and automated processing package to deliver acoustic-based Krill biomass estimates (RAPIDKRILL)",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/bas-acoustics/echopy",
    packages = setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: UNIX",],
                 )

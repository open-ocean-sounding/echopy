import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "echopy",
    version = "0.0.2",
    author = "British Antarctic Survey",
    author_email = "rapidkrill@bas.ac.uk",
    description= "Fisheries acoustic algorithms in Python",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/bas-acoustics/echopy",
    packages = setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
    ],
)

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "echopy",
    version = "0.2.0",
    author = "EchoPY",
    author_email = "echopy@protonmail.com",
    description= "Fisheries acoustic processing in Python",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/open-ocean-sounding/echopy",
    packages = setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
    ],
)

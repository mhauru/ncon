import setuptools

with open("README.md", "r") as f:
    readme = f.read()

setuptools.setup(
    name="ncon",
    version="1.0.0",
    author="Markus Hauru",
    author_email="markus@mhauru.org",
    description="Tensor network contraction function for Python 3.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/mhauru/ncon",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    keywords=["tensor networks"],
    install_requires=["numpy>=1.11.0"],
    extras_require={"tests": ["pytest", "coverage"]},
    python_requires=">=3.6",
)

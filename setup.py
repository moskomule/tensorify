from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

setup(name="tensorify",
      version="0.0.1",
      author="moskomule",
      author_email="hataya@nlab.jp",
      packages=find_packages(exclude=["test", "docs", "examples"]),
      url="https://github.com/moskomule/tensorify",
      description="extension for PyTorch",
      long_description=readme,
      license="Apache License 2.0",
      )

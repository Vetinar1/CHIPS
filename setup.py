from setuptools import setup

setup(
   name="chips",
   version="1.0",
   description="The Cloudy Heuristic/Iterative Parameterspace Sampler",
   author="Stefan Lueders",
   author_email="chips@vetinari.eu",
   packages=["chips"],
   install_requires=["numpy", "scipy", "pandas", "parse", "matplotlib", "seaborn"],
)
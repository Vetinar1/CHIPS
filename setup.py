from setuptools import setup

setup(
   name="CHIPS",
   version="1.0",
   description="The Cloudy Heuristic/Iterative Parameterspace Sampler",
   author="Stefan Lueders",
   author_email="chips@vetinari.eu",
   packages=["CHIPS"],
   install_requires=["numpy", "scipy", "pandas", "parse", "matplotlib", "seaborn"],
)
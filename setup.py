## python3 setup.py bdist_wheel
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='SymmGauge',
     version = '0.5.0',
     author="Jiawei Ruan",
     author_email="ruanjiaw@gmail.com",
     description="A code for analysing ab-initio many-body perturbation theory calculations",
     long_description=long_description,
     python_requires='>=3.6',
     install_requires = [
                        'numpy>=1.18',
                        'scipy>=1.0',
                        ],
     packages=setuptools.find_packages(),
     license='LICENSE.txt',
     # zip_safe = False,   # need python 3.9
)

from setuptools import setup, find_packages

setup(
    name='pairmap',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'rdkit',
        'lomap',
        'argparse',
        'networkx'
    ],
    author='Furui K',
    author_email='furui@li.c.titech.ac.jp',
    description='A comprehensive tool, PairMap, for the calculation of relative binding free energies in complex compound transformations, involving exhaustive generation of intermediates and construction of intermediate-induced perturbation maps.',
    keywords='pairmap, relative binding free energy, intermediate, perturbation map, free energy perturbation',
)

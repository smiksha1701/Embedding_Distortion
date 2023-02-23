from setuptools import find_packages, setup
setup(
    name='DRD',
    packages=['DRD', 'DRD.calculations', 'DRD.visualizations'],
    version='0.1.0',
    description='Embedding Distortion Vizualization',
    author='Sakevych Mykhailo',
    license='MIT',
    install_requires=["scikit-learn", "numpy", "networkx", "matplotlib", "ipywidgets", "plotly", "scipy"]
)
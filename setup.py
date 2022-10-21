from setuptools import setup

setup(
    name='ConfinedBrownianAnalysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'ipympl==0.9.2',
        'jupyterlab==3.4.8',
        'matplotlib==3.6.1',
        'numpy==1.23.4',
        'scipy==1.9.2',
        'seaborn==0.11',
        'tqdm==4.64.1',
        'dill==0.3.5.1'
    ],
    packages=[
        'ConfinedBrownianAnalysis',
    ],
)

from setuptools import setup, find_packages

setup(
    name='IHSetHybridModels',
    version='0.0.2',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'xarray',
        'numba',
        'scipy',
        'matplotlib',
        'IHSetUtils @ git+https://github.com/IHCantabria/IHSetUtils.git',
        'fast_optimization @ git+https://github.com/defreitasL/fast_optimization.git'
    ],
    author='Lucas de Freitas Pereira',
    author_email='lucas.defreitas@unican.es',
    description='IH-SET Hybrid models',
    url='https://github.com/IHCantabria/IHSetHybridModels',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
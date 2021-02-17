# SET UP FILE:
# in order to run in notebooks as an import XXXXXXXXXXXXX
# 1. Git clone the repository to a local computer
# 2. go to the outermost XXXXXXXXXXX folder
# 3. use "pip install . "
# 4. import packages into a jupyter notebook using "from XXXX import xxxxxx"

import setuptools

setuptools.setup(
    name='phasIR',
    version='0.1'
    url='',
    # url='https://github.com/pozzo-research-group/phasIR.git',
    license='MIT',
    author='Shrilakshmi Bonageri, Jaime Rodriguez,' + \
        'Sage Scheiwiller, Maria Politi',
    description='A package for high-throughput measurement of ' + \
        'melting temperature using IR bolometry',
    description_content_type='text/markdown; charset=UTF-8; variant=GFM',
    short_description='Melting temperature determination using IR bolometry',
    short_description_content_type='text/markdown',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown; charset=UTF-8; variant=GFM',
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=['numpy',
                      'pandas',
                      'scikit-image',
                      'scikit-learn',
                      'scipy',
                      'matplotlib',
                      'h5py',
                      'Keras',
                      'tensorflow',
                      'tensorflow-estimator',
                      'Keras-Applications',
                      'Keras-Preprocessing'],
    zip_safe=False,
)

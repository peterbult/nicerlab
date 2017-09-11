from setuptools import setup, find_packages

setup(  
    name='nicerlab',
    version='0.1',
    description='A nicer timing framework',

    author='Peter Bult',
    author_email='',
    url='https://github.com/peterbult/nicerlab',
    license='MIT',

    packages=find_packages(),
    scripts=['bin/ni-lightcurve', 'bin/gti-select'],
    zip_safe=False
)


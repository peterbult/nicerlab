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
    install_requires=[
        'numpy>=1.13'
        'astropy>=2.0'
    ],
    zip_safe=False
)


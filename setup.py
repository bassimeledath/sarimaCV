from setuptools import setup, find_packages

setup(
    name='sarimaCV',
    version='0.1',
    packages=find_packages(),
    description='SARIMA Cross-Validation for Time Series Analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Bassim Eledath',
    author_email='bassimfaizal@gmail.com',
    url='https://github.com/bassimeledath/sarimaCV',
    install_requires=[
        'numpy',
        'pandas',
        'statsmodels',
        'concurrent.futures'
    ],
)

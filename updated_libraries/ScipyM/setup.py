from setuptools import setup

install_requires = [x.strip('\n') for x in open("requirements.txt", 'r').readlines()]

setup(
    name='newScience',
    version='0.1',
    description='newScience trial package',
    author='HKU',
    install_requires=install_requires,
    packages=['newScience'],
)
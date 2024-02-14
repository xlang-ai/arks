from setuptools import setup

install_requires = [x.strip('\n') for x in open("requirements.txt", 'r').readlines()]

setup(
    name='jumptensor',
    version='0.1',
    description='jumptensor trial package',
    author='HKU',
    install_requires=install_requires,
    packages=['jumptensor'],
)
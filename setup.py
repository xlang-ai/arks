from setuptools import setup

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name='arks',
    packages=['arks'],
    version='0.0.0',
    license='Apache License 2.0',
    description='Retrieval-augmented code generation tool',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Hongjin SU',
    author_email='hjsu@cs.hku.hk',
    keywords=['retrieval', 'codegen']
)
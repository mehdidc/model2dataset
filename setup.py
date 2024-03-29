#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

def load_requirements(f):
    return [l.strip() for l in open(f).readlines() ]

requirements = load_requirements("requirements.txt")

test_requirements = requirements + ["pytest", "pytest-runner"]

setup(
    author="Mehdi Cherti",
    author_email='mehdicherti@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Compute embeddings or generated samples on datasets using pre-trained models",
    entry_points={
        'console_scripts': [
            'model2dataset=model2dataset.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='model2dataset',
    name='model2dataset',
    packages=find_packages(include=['model2dataset', 'model2dataset.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/mehdidc/model2dataset',
    version='0.1.0',
    zip_safe=False,
)

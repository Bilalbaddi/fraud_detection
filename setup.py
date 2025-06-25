from setuptools import find_packages,setup

from typing import List


def get_requirements_list()-> List:
    requirements = []
    try:
        with open('requirements.txt','r') as file:
            requirements = file.readlines()
            requirements = [req.replace('\n','')  for req in requirements]
            if '-e .' in requirements:
                requirements.remove('-e .')
    except FileNotFoundError:
        print('file not found')
    return requirements


setup(
    name='fraud_detection',
    version='0.0.1',
    author='Bilal',
    author_email= 'bilalbaddi7@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements_list()
)



    
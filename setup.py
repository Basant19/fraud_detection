from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function returns the list of requirements without comments,
    empty lines, or editable install lines like '-e .'
    '''
    requirements = []
    with open(file_path) as file_obj:
        for line in file_obj:
            line = line.strip()
            # Ignore empty lines
            if not line:
                continue
            # Remove comments
            if '#' in line:
                line = line.split('#')[0].strip()
            # Skip editable mode
            if line == HYPEN_E_DOT:
                continue
            requirements.append(line)
    return requirements

setup(
    name='fraud_detection',
    version='1.0.1',
    author='Basant',
    author_email='btirkey1208@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)

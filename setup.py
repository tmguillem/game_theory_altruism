from setuptools import setup, find_packages
"""Setup module for project."""

setup(
    name='game_theory_altruism',
    version='0.1',
    description='Code for the project: `Is altruism evolutionary stable`. For the class Introduction to Game Theory'
                'spring semester 2020.',

    author='Guillem Torrente i Marti',
    author_email='guillemtorrente@hotmail.com',

    packages=find_packages(exclude=[]),
    python_requires='>=3.5',
    install_requires=[
        'numpy',
        'tqdm',
        'matplotlib',
    ],
)
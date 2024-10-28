
# setup.py
from setuptools import setup, find_packages

setup(
    name='corosims',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Add your dependencies here
    ],
    entry_points={
        'console_scripts': [
            # Define any command-line scripts here
        ],
    },
    author='Jorge Llop-Sayson',
    author_email='jorge.llop.sayson@jpl.nasa.gov',
    description='A brief description of your library',
    python_requires='>=3.6',
)

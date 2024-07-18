
# setup.py
from setuptools import setup, find_packages

setup(
    name='cgisim_sims',
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
    author='Your Name',
    author_email='your.email@example.com',
    description='A brief description of your library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/the_library',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

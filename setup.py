from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='deepbondnew',
    version='0.0.3',
    description='Deep neural approach to Boundary and Disfluency Detection',
    long_description=readme,
    author='Marcos Treviso',
    author_email='marcostreviso@usp.br',
    url='https://github.com/mtreviso/deepbond',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    data_files=['LICENSE'],
    zip_safe=False,
    keywords='evaluator',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    py_modules=['deepbondnew']
)

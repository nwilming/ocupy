from setuptools import setup, find_packages

version = '0.1'
name='ocupy'

setup(name=name,
    version=version,
    description="Oculography Analysis Toolbox",
    long_description=
    """Ocupy provides functions for eye-tracking data analysis:

    * FixMat objects for reading of and filtering by fixation- and meta-data
    * Corresponding objects for stimulus data, aligned to FixMat objects
    * Measures for prediction quality for eye-tracking data: AUC, NSS, KL, EMD.
    * Lower and upper bound calculation for prediction quality of
      attention models.
    * RPC Client/Server for parallel task execution on a grid
    * Evaluation (with cross-validation) of attention models 
    """,
    license='GPL v.2, see LICENSE',
    author='WhiteMatter Labs GmbH',
    author_email='nwilming@uos.de',
    url='http://github.com/nwilming/ocupy/',
    packages=find_packages(exclude=['ez_setup']),
    classifiers=[
      'Development Status :: 5 - Production/Stable',
      'Environment :: Console',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: GNU General Public License (GPL)',
      'Operating System :: OS Independent',
      'Topic :: Scientific/Engineering :: Information Analysis',
      'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    zip_safe=True,
    include_package_data=True,
    setup_requires=[
      'numpy',
    ],
    install_requires=[
      'numpy',
      'scipy',
      'PIL',
    ],
    test_suite='ocupy.tests',
    extras_require={
        'doc':['sphinx','matplotlib']
    },
    )

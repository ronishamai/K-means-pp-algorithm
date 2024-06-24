from setuptools import setup, find_packages, Extension

setup(
    name='spkmeans',
    version='0.1.0',
    author='Ayala Koslowsky, Roni Shamai',
    author_email='ronishamai@mail.tau.ac.il',
    description='the normalized spectral clustering algorithm',
    install_requires=['invoke'],
    packages=find_packages(),
    license='GPL-2',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    ext_modules=[
        Extension(
            'spkmeans',
            ['spkmeansmodule.c','spkmeans.c']
        )

    ]

)

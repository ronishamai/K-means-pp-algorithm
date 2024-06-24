from setuptools import setup, find_packages, Extension

setup(
    name='mykmeanssp',
    version='0.1.0',
    author='Ayala Koslowsky, Roni Shamai',
    author_email='ronishamai@mail.tau.ac.il',
    description='K-means++ algorithm, which is used to choose initial centroids for the K-means algorithm',
    install_requires=['invoke'],
    package=find_packages(),
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
            'mykmeanssp',
            ['kmeans.c']
        )

    ]

)

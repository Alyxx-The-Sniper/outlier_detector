from setuptools import setup, find_packages

setup(
    name='outlier_detector',
    version='0.1.0',
    author='kaikuh pogi',
    description='Outlier detection library with multiple models and visualizations.',
    url='https://github.com/your-username/outlier_detector',  # Replace with your GitHub URL
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'tensorflow',
        'keras',
        'umap-learn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)

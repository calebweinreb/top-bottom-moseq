from setuptools import setup, Extension

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='top-bottom-moseq',
    version='0.0.0',
    author='Caleb Weinreb',
    author_email='calebsw@gmail.com',
    include_package_data=True,
    packages=['top_bottom_moseq'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent'
    ],
    ext_modules=[Extension(
        'orthographic', 
        ['top_bottom_moseq/orthographic.c'], 
        extra_compile_args=['-fPIC','-shared']
    )],
    python_requires='>=3.7, <3.8',
    install_requires=[
        'av',
        'click',
        'h5py',
        'ipykernel',
        'imageio',
        'imageio-ffmpeg',
        'joblib',
        'kornia',
        'matplotlib',
        'numpy',
        'open3d==0.9.0',
        'pyyaml',
        'scikit-learn',
        'scipy',
        'segmentation_models_pytorch',
        'torch<1.13',
        'torchinfo',
        'torchplot',
        'torchvision',
        'tqdm',
    ], 
    url='https://github.com/calebweinreb/top-bottom-moseq'
)

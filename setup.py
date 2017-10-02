try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='noxer',
      version='0.1',
      description='Streamlined supervised learning and deployment of ML models.',
      long_description=(open('README.txt').read()),
      url='https://github.com/iaroslav-ai/noxer/',
      license='MIT',
      author='The noxer contributors',
      packages=['noxer'],
      install_requires=["numpy", "scipy", "scikit-learn", "keras", "h5py"]
      )

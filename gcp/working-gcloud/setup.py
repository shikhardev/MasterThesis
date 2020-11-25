from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      include_package_data=True,
      description='Hpbandster config evals',
      author='Shikhar Dev',
      author_email='shikhardev@gmail.com',
      license='GIPSA',
      install_requires=[
          'hpbandster',
          'numpy',
          'torch',
          'torchvision',
          'tensorflow',
          'keras',
          'matplotlib'
      ],
      zip_safe=False)

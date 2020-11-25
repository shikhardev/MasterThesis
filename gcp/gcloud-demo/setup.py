from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      include_package_data=True,
      description='example to run keras on gcloud ml-engine',
      author='ZHAO Zilong',
      author_email='zilong.zhao@gipsa-lab.fr',
      license='GIPSA',
      install_requires=[
          'keras==2.2.0',
          'h5py'
      ],
      zip_safe=False)

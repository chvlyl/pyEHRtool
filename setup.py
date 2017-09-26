from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='pyEHRtool',
      version='0.1',
      description='A python package for processing EHR data',
      url='http://github.com/chvlyl/pyEHRtool',
      author='Eric Chen',
      author_email='chvlyl@gmail.com',
      license='MIT',
      packages=['pyEHRtool'],
      install_requires=[
          'pandas',
          'numpy',
      ],
      include_package_data=True,
      
      test_suite='nose.collector',
      tests_require=['nose'],
      
      zip_safe=False)

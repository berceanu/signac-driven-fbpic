from setuptools import setup

setup(
   name='PostProc',
   version='0.1.0',
   author='Andrei Berceanu',
   author_email='andrei.berceanu@eli-np.ro',
   packages=['postproc'],
#    scripts=['bin/script1','bin/script2'],
#    url='http://pypi.python.org/pypi/PackageName/',
   license='LICENSE.md',
   description='Package containing modules for post-processing fbpic simulation output',
   long_description=open('README.md').read(),
   install_requires=[
       "signac >= 1.1.0",
       "signac-flow >= 0.7.1",
       "signac-dashboard >= 0.2.3",
       "fbpic >= 0.12.0",
       "opmd_viewer >= 0.8.2"
   ],
)

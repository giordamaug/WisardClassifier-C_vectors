from setuptools import setup
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(name                = 'Wisard Library',
      version             = '1.0',
      author              = 'Maurizio Giordano',
      author_email        = 'maurizio.giordano@cnr.it',
      maintainer          = 'Maurizio Giordano',
      maintainer_email    = 'maurizio.giordano@cnr.it',
      description         = 'Wisard Library',
      long_description    = """...
          """,
      platforms           = ['Mac OSX', 'POSIX',],
      classifiers         = [
                             '...',
                             ],
      license             = 'GNU Lesser General Public License v2 or later (LGPLv2+)',
      packages            = ['wisard', 'WisardClassifier'],
      ext_modules=[Extension("wisard_wrapper",
                             ["wisard_wrapper.pyx",
                              "Discriminator.cpp", "Ram.cpp"],
                             language="c++",
                             extra_compile_args=["-std=c++11"])],
      cmdclass = {'build_ext': build_ext})

from distutils.core import setup, Extension

main_libraries =  ['torch']
library_dirs = ['torch']
library_dirs.append('./torch/csrc')

module = Extension('torch._C',
                   sources = ['torch/csrc/stub.c'],
                   libraries = main_libraries,
                   library_dirs = library_dirs,
                   language = 'c')

setup(name = 'pxtorch', version = '1.0', description = 'This is a pxtorch module', ext_modules = [module])

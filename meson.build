project(
  'neural-network',
  'cpp',
  default_options: [
    'cpp_std=c++20',
    'optimization=3',
    #'b_sanitize=address,undefined',
    'warning_level=2'
  ]
)

eigen = dependency('eigen3')
openblas = dependency('openblas')

add_project_arguments([ '-march=native', '-mtune=native', '-DEIGEN_USE_BLAS' ], language: 'cpp')

sources = []

subdir('src')

executable('neural-network', sources, dependencies : [ eigen, openblas ])

project(
  'neural-network',
  'cpp',
  default_options: [
    'cpp_std=c++17',
    # 'optimization=3',
    'b_sanitize=address,undefined',
    'warning_level=2'
  ]
)

add_project_arguments([ '-march=native', '-mtune=native' ], language: 'cpp')

sources = []

subdir('src')

exe = executable('neural-network', sources)

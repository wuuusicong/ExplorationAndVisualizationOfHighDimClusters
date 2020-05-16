from distutils.core import setup

setup(
  name = 'cluster-plot',
  packages = ['cluster-plot'],
  version = '0.1',
  license='MIT',
  description = 'Implementation of ClusterPlot paper',
  author = 'Or Malkai',
  author_email = 'ormalkai@gmail.com',
  url = 'https://github.com/user/reponame',   # TODO
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # TODO
  keywords = ['ClusterPlot', 'ClusterPlots', 'cluster', 'plot', 'ClustersPlot'],   # Keywords that define your package best
  install_requires=[            # TODO
          'validators',
          'beautifulsoup4',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Data Scientists, Researches, Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)
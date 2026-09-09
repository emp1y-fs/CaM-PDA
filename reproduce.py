"""Run the paper benchmark download and evaluation workflow."""
import runpy

if __name__ == '__main__':
    runpy.run_module('reproduction.run', run_name='__main__')

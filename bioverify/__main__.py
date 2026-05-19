# __main__.py - Main entry point for BioVerify CLI.
# Author: Sebastian Chupac (xchupa03)
# Date: 19.05.2026
"""
Main entry point for BioVerify CLI.

Allows running: python -m bioverify <command>
"""
from .cli.index import main

if __name__ == '__main__':
    main()

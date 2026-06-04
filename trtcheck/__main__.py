"""Enable ``python -m trtcheck`` to run the CLI.

The console-script entry point (``trtcheck = trtcheck.cli:main``) already works
once the package is installed, but the documented ``python -m trtcheck`` form
needs this module. Both dispatch to the same Click command.
"""

from trtcheck.cli import main

if __name__ == "__main__":
    main()

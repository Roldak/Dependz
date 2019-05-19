import libdependzlang as ldl

import os
import sys


def run(src_file):
    ctx = ldl.AnalysisContext()
    u = ctx.get_from_file(src_file)
    assert not u.diagnostics
    u.root.dump()


if __name__ == "__main__":
    run(os.path.join(sys.argv[1], "test.dep"))


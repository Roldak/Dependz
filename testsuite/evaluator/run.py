import libdependzlang as ldl

import os
import sys


def defs_to_eval(unit):
    return unit.root.findall(
        lambda n: n.is_a(ldl.Definition) and "test" in n.f_ident.text
    )


def run(src_file):
    ctx = ldl.AnalysisContext()
    u = ctx.get_from_file(src_file)

    assert not u.diagnostics

    for d in defs_to_eval(u):
        r = d.p_value.p_to_string
        print("{} = {}".format(d.f_ident.text, r))



if __name__ == "__main__":
    run(os.path.join(sys.argv[1], "test.dep"))

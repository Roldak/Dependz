import libdependzlang as ldl

import os
import sys

import traceback


def targets(unit):
    return unit.root.findall(
        lambda n: n.is_a(ldl.Introduction) and "test" in n.f_ident.text
    )

def find_Type_intro(unit):
    return unit.root.find(
        lambda n:
        n.is_a(ldl.Identifier) and
        n.text == "Type" and
        n.p_is_introducing
    )


def run(src_file):
    ctx = ldl.AnalysisContext()
    u = ctx.get_from_file(src_file)
    type_intro = find_Type_intro(u)

    assert not u.diagnostics

    for d in targets(u):
        intro_name = d.f_ident.text

        try:
            r = d.f_term.p_constructors
        except ldl.PropertyError:
            r = None
            print("A crash occurred while discovering constructors of {}.".format(
                intro_name
            ))
            traceback.print_exc()

        if r is not None:
            print("{}: Success".format(intro_name))
            print("  {}".format([
                x.text for x in r if x != type_intro
            ]))
        else:
            print("{} : Failed".format(intro_name))


if __name__ == "__main__":
    run(os.path.join(sys.argv[1], "test.dep"))

import libdependzlang as ldl

import os
import sys

import traceback


def defs_to_domain_check(unit):
    return unit.root.findall(
        lambda n: n.is_a(ldl.Definition) and "test" in n.f_ident.text
    )


def run(src_file):
    ctx = ldl.AnalysisContext()
    u = ctx.get_from_file(src_file)

    def set_debug(v):
        u.root.p_set_logic_equation_debug_mode(1 if v else 0)

    assert not u.diagnostics

    for d in defs_to_domain_check(u):
        def_name = d.f_ident.text

        try:
            r = d.p_check_domains(3)
        except ldl.PropertyError:
            r = False

        if r:
            print("{}: Success".format(def_name))
            for t in [d.f_term] + d.f_term.findall(ldl.Term):
                dom = t.p_domain_val
                print("  {}: {}".format(t, dom.p_to_string if dom else "None"))
        else:
            set_debug(True)
            print("{} : Failed".format(def_name))

            try:
                d.p_check_domains(3)
            except ldl.PropertyError:
                print("A crash occurred while domain checking {}.".format(
                    def_name
                ))
                traceback.print_exc()

            set_debug(False)


if __name__ == "__main__":
    run(os.path.join(sys.argv[1], "test.dep"))

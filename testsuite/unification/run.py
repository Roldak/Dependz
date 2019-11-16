import libdependzlang as ldl

import os
import sys

import traceback


def queries(unit):
    firsts = unit.root.findall(
        lambda n:
        n.is_a(ldl.Introduction)
        and "first" in n.f_ident.text
    )
    seconds_name = [
        f.f_ident.text.replace("first", "second")
        for f in firsts
    ]
    return [
        ldl.UnifyQuery(
            first=f.f_term,
            second=unit.root.find(
                lambda d:
                d.is_a(ldl.Introduction) and
                d.f_ident.text == s_name
            ).f_term
        )
        for f, s_name in zip(firsts, seconds_name)
    ]


def run(src_file):
    ctx = ldl.AnalysisContext()
    u = ctx.get_from_file(src_file)

    def set_debug(v):
        u.root.p_set_logic_equation_debug_mode(1 if v else 0)

    assert not u.diagnostics

    all_queries = queries(u)
    all_free_syms = set()
    for q in all_queries:
        for term in [q.first, q.second]:
            all_free_syms.update(term.p_free_symbols())

    try:
        substs = u.root.p_unify_all(list(all_queries), list(all_free_syms))
    except ldl.PropertyError:
        substs = None

    if substs:
        print("Unification success: {}".format({
            s.from_symbol: s.to_term.p_to_string
            for s in substs
        }))
    else:
        print("Unification failure")
        set_debug(True)
        try:
            u.root.p_unify_all(list(all_queries), list(all_free_syms))
        except ldl.PropertyError:
            print(traceback.print_exc())


if __name__ == "__main__":
    run(os.path.join(sys.argv[1], "test.dep"))

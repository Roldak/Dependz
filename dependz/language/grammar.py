from __future__ import absolute_import, division, print_function

from langkit.parsers import Grammar, List, Or, Pick, Null
from language.lexer import dependz_lexer as L
from language.ast import (
    Program, Introduction, Definition, Abstraction, Apply, SourceId, Arrow,
    Term
)


dependz_grammar = Grammar('main_rule')
D = dependz_grammar

dependz_grammar.add_rules(
    main_rule=List(D.toplevel, empty_valid=True, list_cls=Program),
    toplevel=Or(D.intro, D.definition),
    intro=Introduction(D.ident, ':', D.defterm, L.Newlines),
    definition=Definition(D.ident, '=', D.term, L.Newlines),

    ident=SourceId(L.Ident),

    term=Or(Apply(D.term, D.term1), D.term1),
    term1=Or(D.ident, D.term2),
    term2=Or(Abstraction('\\', D.ident, '.', D.term), D.term3),
    term3=Pick('(', D.term, ')'),

    defterm=Or(
        Or(Arrow(Null(Term), D.defterm1, '->', D.defterm),
           Arrow(D.term, ':', D.defterm1, '->', D.defterm)),
        D.defterm1
    ),
    defterm1=Or(Pick("(", D.defterm, ")"), D.term),
)

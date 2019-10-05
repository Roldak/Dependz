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
    definition=Definition(D.ident, '=', D.defterm, L.Newlines),

    ident=SourceId(L.Ident),

    defterm=Or(
        Or(Arrow(Null(Term), D.term, '->', D.defterm),
           Arrow(D.term, ':', D.term, '->', D.defterm)),
        D.term
    ),
    term=Or(Apply(D.term, D.term1), D.term1),
    term1=Or(D.ident, D.term2),
    term2=Or(Abstraction('\\', D.ident, '.', D.term), D.term3),
    term3=Pick('(', D.defterm, ')')
)

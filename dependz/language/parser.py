from __future__ import absolute_import, division, print_function

from langkit.dsl import ASTNode, abstract, Field, T, Bool
from langkit.envs import EnvSpec, add_env, add_to_env_kv
from langkit.expressions import Self, langkit_property
from langkit.parsers import Grammar, List, Or, Pick
from language.lexer import dependz_lexer as L


def newlines():
    return _(List(D.nl, empty_valid=True))


@abstract
class DependzNode(ASTNode):
    """
    Root node class for Dependz AST nodes.
    """
    pass


@abstract
class DefTerm(DependzNode):
    pass


@abstract
class Term(DefTerm):
    pass


class Identifier(Term):
    token_node = True

    @langkit_property(public=True, return_type=Bool)
    def is_defining():
        return Self.parent.is_a(T.Introduction)

    @langkit_property(public=True, return_type=T.Introduction.entity)
    def intro():
        return Self.node_env.get_first(Self.symbol).cast(T.Introduction)

    @langkit_property(public=True, return_type=DefTerm.entity)
    def kind():
        return Self.intro.term


class Apply(Term):
    fun = Field(type=Term)
    args = Field(type=Term.list)


class Arrow(DefTerm):
    lhs = Field(type=Term)
    rhs = Field(type=DefTerm)


class Introduction(DependzNode):
    """
    Identifer : Term
    """
    ident = Field(type=Identifier)
    term = Field(type=DefTerm)

    env_spec = EnvSpec(
        add_to_env_kv(Self.ident.symbol, Self)
    )


class Program(Introduction.list):
    env_spec = EnvSpec(
        add_env()
    )


dependz_grammar = Grammar('main_rule')
D = dependz_grammar

dependz_grammar.add_rules(
    main_rule=List(D.intro, empty_valid=True, list_cls=Program),
    intro=Introduction(D.ident, ':', D.defterm, L.Newlines),

    ident=Identifier(L.Identifier),

    term=Or(D.apply, D.term1),
    term1=Or(D.ident, D.parens),
    apply=Apply(D.term1, List(D.term1, empty_valid=False)),
    parens=Pick('(', D.term, ')'),

    defterm=Or(D.arrow, D.defterm1),
    defterm1=D.term,
    arrow=Arrow(D.defterm1, '->', D.defterm)
)

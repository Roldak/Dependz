from __future__ import absolute_import, division, print_function

from langkit.dsl import ASTNode, abstract, Field, T
from langkit.envs import EnvSpec, add_env, add_to_env_kv
from langkit.expressions import Self, Entity, langkit_property
from langkit.parsers import Grammar, List, Opt, Or, Pick, _
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
class Term(DependzNode):
    pass


class Identifier(Term):
    token_node = True

    @langkit_property(public=True)
    def is_defining():
        return Self.parent.is_a(T.Introduction)

    @langkit_property(public=True)
    def intro():
        return Self.node_env.get_first(Self.symbol).cast(T.Introduction)

    @langkit_property(public=True)
    def kind():
        return Self.intro.term


class Arrow(Term):
    lhs = Field(type=Term)
    rhs = Field(type=Term)


class Introduction(DependzNode):
    """
    Identifer : Term
    """
    ident = Field(type=Identifier)
    term = Field(type=Term)

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
    intro=Introduction(D.ident, ':', D.term, L.Newlines),
    ident=Identifier(L.Identifier),
    term=Or(D.arrow, D.term1),
    term1=D.ident,
    arrow=Arrow(D.term1, '->', D.term)
)

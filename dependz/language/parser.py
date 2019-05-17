from __future__ import absolute_import, division, print_function

from langkit.dsl import (
    ASTNode, abstract, Field, T, Bool, LexicalEnv, synthetic, Struct,
    UserField, NullField, Symbol
)
from langkit.envs import EnvSpec, add_env, add_to_env_kv
from langkit.expressions import (
    Self, langkit_property, Property, AbstractProperty, Not, No, If,
    ArrayLiteral, String
)
from langkit.parsers import Grammar, List, Or, Pick
from language.lexer import dependz_lexer as L


@abstract
class DependzNode(ASTNode):
    """
    Root node class for Dependz AST nodes.
    """
    @langkit_property(public=True, memoized=True)
    def make_apply(t1=T.Term, t2=T.Term):
        return Apply.new(lhs=t1, rhs=t2)

    @langkit_property(public=True, memoized=True)
    def make_ident(name=T.Symbol):
        return T.FreshId.new(name=name)


@abstract
class DefTerm(DependzNode):
    to_string = AbstractProperty(public=True, type=T.String)

    @langkit_property(public=True, return_type=T.Identifier.entity.array)
    def unbound_ids():
        return Self.collect_ids.filter(lambda i: i.intro.is_null)

    @langkit_property(return_type=T.Identifier.entity.array,
                      memoized=True)
    def collect_ids():
        return Self.match(
            lambda id=Identifier: id.as_bare_entity.singleton,
            lambda _: Self.children.mapcat(
                lambda t: t.cast(DefTerm).then(
                    lambda dt: dt.collect_ids
                )
            )
        )


@abstract
class Term(DefTerm):
    pass


@abstract
class Identifier(Term):
    sym = AbstractProperty(type=Symbol)

    to_string = Property(Self.sym.image)

    @langkit_property(public=True, return_type=Bool)
    def is_defining():
        return Self.parent.is_a(Introduction)

    @langkit_property(public=True, return_type=T.Introduction.entity)
    def intro():
        return Self.node_env.get_first(Self.sym).cast(Introduction)

    @langkit_property(public=True, return_type=DefTerm.entity)
    def kind():
        return Self.intro.term


@synthetic
class FreshId(Identifier):
    name = UserField(type=T.Symbol)
    sym = Property(Self.name)


class SourceId(Identifier):
    token_node = True
    sym = Property(Self.symbol)


@synthetic
class Apply(Term):
    lhs = Field(type=Term)
    rhs = Field(type=Term)

    to_string = Property(
        Self.lhs.to_string.concat(String(' ')).concat(Self.rhs.to_string)
    )


class Arrow(DefTerm):
    lhs = Field(type=Term)
    rhs = Field(type=DefTerm)

    to_string = Property(
        Self.lhs.to_string.concat(String(' -> ')).concat(Self.rhs.to_string)
    )


class Introduction(DependzNode):
    """
    Identifer : Term
    """
    ident = Field(type=SourceId)
    term = Field(type=DefTerm)

    env_spec = EnvSpec(
        add_to_env_kv(Self.ident.sym, Self)
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

    ident=SourceId(L.Ident),

    term=Or(Apply(D.term, D.term1), D.term1),
    term1=Or(D.ident, D.parens),
    parens=Pick('(', D.term, ')'),

    defterm=Or(D.arrow, D.defterm1),
    defterm1=D.term,
    arrow=Arrow(D.defterm1, '->', D.defterm)
)

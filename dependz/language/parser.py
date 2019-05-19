from __future__ import absolute_import, division, print_function

from langkit.dsl import (
    ASTNode, abstract, Field, T, Bool, LexicalEnv, synthetic, Struct,
    UserField, NullField, Symbol
)
from langkit.envs import EnvSpec, add_env, add_to_env_kv, handle_children
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
        return FreshId.new(name=name)

    @langkit_property(public=True, memoized=True)
    def make_abstraction(id=T.Identifier, rhs=T.Term):
        return Abstraction.new(ident=id, term=rhs)

    @langkit_property(external=True, return_type=T.Symbol,
                      uses_entity_info=False, uses_envs=False)
    def fresh_symbol(prefix=T.Symbol):
        pass


@abstract
class DefTerm(DependzNode):
    to_string = AbstractProperty(public=True, type=T.String)


@abstract
class Term(DefTerm):
    pass


@abstract
class Identifier(Term):
    sym = AbstractProperty(type=Symbol)

    to_string = Property(Self.sym.image)

    @langkit_property(public=True, return_type=T.Introduction.entity)
    def intro():
        return Self.node_env.get_first(Self.sym).cast(Introduction)


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

    to_string = Property(String("(").concat(
        Self.lhs.to_string.concat(String(' ')).concat(Self.rhs.to_string)
    ).concat(String(")")))


@synthetic
class Abstraction(Term):
    ident = Field(type=Identifier)
    term = Field(type=Term)

    to_string = Property(String("(\\").concat(
        Self.ident.to_string.concat(String('. ')).concat(Self.term.to_string)
    ).concat(String(")")))


class Arrow(DefTerm):
    lhs = Field(type=Term)
    rhs = Field(type=DefTerm)

    to_string = Property(
        Self.lhs.to_string.concat(String(' -> ')).concat(Self.rhs.to_string)
    )


class Introduction(DependzNode):
    """
    Identifer : DefTerm
    """
    ident = Field(type=SourceId)
    term = Field(type=DefTerm)

    @langkit_property(public=True, return_type=T.Definition.entity)
    def definition():
        return Self.children_env.get_first('__definition').cast(T.Definition)

    env_spec = EnvSpec(
        add_to_env_kv(Self.ident.sym, Self),
        add_env()
    )


class Definition(DependzNode):
    """
    Identifier = Term
    """
    ident = Field(type=SourceId)
    term = Field(type=Term)

    env_spec = EnvSpec(
        handle_children(),
        add_to_env_kv(
            '__definition', Self,
            dest_env=Self.ident.intro.children_env
        )
    )


class Program(DependzNode.list):
    env_spec = EnvSpec(
        add_env()
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

    defterm=Or(D.arrow, D.defterm1),
    defterm1=D.term,
    arrow=Arrow(D.defterm1, '->', D.defterm)
)

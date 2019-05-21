from __future__ import absolute_import, division, print_function

from langkit.dsl import (
    ASTNode, abstract, Field, T, Bool, LexicalEnv, synthetic, Struct,
    UserField, NullField, Symbol
)
from langkit.envs import EnvSpec, add_env, add_to_env_kv, handle_children
from langkit.expressions import (
    Self, Entity, langkit_property, Property, AbstractProperty, Not, No, If,
    ArrayLiteral, String, Var, AbstractKind, Let, CharacterLiteral as Character
)


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

    @langkit_property(public=True, memoized=True)
    def make_arrow(t1=T.DefTerm, t2=T.DefTerm):
        return Arrow.new(lhs=t1, rhs=t2)

    @langkit_property(external=True, return_type=T.Symbol,
                      uses_entity_info=False, uses_envs=False)
    def fresh_symbol(prefix=T.Symbol):
        pass

    @langkit_property(external=True, return_type=T.LogicVar,
                      uses_entity_info=False, uses_envs=False)
    def create_logic_var():
        pass


@abstract
class DefTerm(DependzNode):
    to_string = AbstractProperty(public=True, type=T.String)


@abstract
class Term(DefTerm):
    @langkit_property(public=True, return_type=T.Term, memoized=False)
    def eval():
        return Self.match(
            lambda id=Identifier: id.intro._.definition.then(
                lambda d: d.term.node.eval,
                default_val=id
            ),
            lambda ap=Apply: ap.lhs.eval.cast(Abstraction).then(
                lambda ab: ab.term.substitute(ab.ident.sym, ap.rhs).eval,
                default_val=ap
            ),
            lambda other: other
        )

    @langkit_property(public=True, return_type=T.Term, memoized=False)
    def substitute(sym=T.Symbol, val=T.Term):
        return Self.match(
            lambda id=Identifier: If(
                id.sym == sym,
                val,
                id
            ),
            lambda ap=Apply: ap.parent.make_apply(
                ap.lhs.substitute(sym, val),
                ap.rhs.substitute(sym, val)
            ),
            lambda ab=Abstraction: If(
                ab.ident.sym == sym,
                ab,
                If(
                    val.is_free(ab.ident.sym),
                    ab.fresh_symbol(ab.ident.sym).then(
                        lambda symp: ab.parent.make_abstraction(
                            ab.make_ident(symp),
                            ab.term
                            .rename(ab.ident.sym, symp)
                            .substitute(sym, val)
                        )
                    ),
                    ab.parent.make_abstraction(
                        ab.ident,
                        ab.term.substitute(sym, val)
                    )
                )

            )
        )

    @langkit_property(public=True, return_type=T.Term, memoized=False)
    def normalize():
        return Self.eval.match(
            lambda id=Identifier: id,
            lambda ap=Apply: ap.parent.make_apply(
                ap.lhs.normalize,
                ap.rhs.normalize
            ),
            lambda ab=Abstraction: ab.parent.make_abstraction(
                ab.ident,
                ab.term.normalize
            )
        )

    @langkit_property(return_type=T.Term, public=True)
    def rename(old=T.Symbol, by=T.Symbol):
        return Self.match(
            lambda id=Identifier: If(
                id.sym == old,
                id.parent.make_ident(by),
                id
            ),
            lambda ap=Apply: ap.parent.make_apply(
                ap.lhs.rename(old, by),
                ap.rhs.rename(old, by)
            ),
            lambda ab=Abstraction: If(
                old == ab.ident.sym,
                ab,
                ab.parent.make_abstraction(
                    ab.ident,
                    ab.term.rename(old, by)
                )
            )
        )

    @langkit_property(return_type=T.Bool, public=True)
    def is_free(sym=T.Symbol):
        return Self.match(
            lambda id=Identifier: id.sym == sym,
            lambda ap=Apply: ap.lhs.is_free(sym) | ap.rhs.is_free(sym),
            lambda ab=Abstraction: (ab.ident.sym != sym) | ab.term.is_free(sym)
        )


@abstract
class Identifier(Term):
    sym = AbstractProperty(type=Symbol)

    to_string = Property(Self.sym.image)

    @langkit_property(public=True, return_type=T.Bool)
    def is_introducing():
        return Self.parent.is_a(Introduction)

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


@synthetic
class Arrow(DefTerm):
    lhs = Field(type=DefTerm)
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

    @langkit_property(public=True, return_type=T.String)
    def eval_and_print():
        return Self.term.normalize.to_string

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

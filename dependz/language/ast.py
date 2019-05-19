from __future__ import absolute_import, division, print_function

from langkit.dsl import (
    ASTNode, abstract, Field, T, Bool, LexicalEnv, synthetic, Struct,
    UserField, NullField, Symbol
)
from langkit.envs import EnvSpec, add_env, add_to_env_kv, handle_children
from langkit.expressions import (
    Self, Entity, langkit_property, Property, AbstractProperty, Not, No, If,
    ArrayLiteral, String, Var, AbstractKind, Let
)


class Substitution(Struct):
    sym = UserField(type=T.Symbol)
    actual = UserField(type=T.Term.entity)


class Result(Struct):
    value = UserField(type=T.Term.entity)
    context = UserField(type=T.Substitution.array)


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
    @langkit_property(public=False, return_type=T.Result,
                      kind=AbstractKind.abstract)
    def eval(ctx=T.Substitution.array):
        pass

    @langkit_property(return_type=T.Term, public=True,
                      kind=AbstractKind.abstract)
    def substitute(old=T.Symbol, by=T.Symbol):
        pass


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

    @langkit_property()
    def eval(ctx=T.Substitution.array):
        return ctx.find(lambda s: s.sym == Self.sym).then(
            lambda s: Result.new(
                value=s.actual,
                context=ctx
            ),
            default_val=Self.intro.definition.then(
                lambda d: d.eval.then(lambda r: Result.new(
                    value=r.value,
                    context=r.context.concat(ctx)
                )),
                default_val=Result.new(
                    value=Entity,
                    context=ctx
                )
            )
        )

    @langkit_property()
    def substitute(old=T.Symbol, by=T.Symbol):
        return Self.parent.make_ident(
            If(Self.sym == old, by, Self.sym)
        )


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

    @langkit_property()
    def eval(ctx=T.Substitution.array):
        evaled_lhs = Var(Entity.lhs.eval(ctx))
        evaled_rhs = Var(Entity.rhs.eval(evaled_lhs.context))

        return evaled_lhs.value.cast(T.Abstraction).then(
            lambda abs: abs.apply(evaled_rhs.context, evaled_rhs.value),
            default_val=Result.new(
                value=Self.parent.make_apply(
                    evaled_lhs.value.node,
                    evaled_rhs.value.node
                ).as_entity,
                context=evaled_rhs.context
            )
        )

    @langkit_property()
    def substitute(old=T.Symbol, by=T.Symbol):
        return Self.parent.make_apply(
            Self.lhs.substitute(old, by),
            Self.rhs.substitute(old, by)
        )


@synthetic
class Abstraction(Term):
    ident = Field(type=Identifier)
    term = Field(type=Term)

    to_string = Property(String("(\\").concat(
        Self.ident.to_string.concat(String('. ')).concat(Self.term.to_string)
    ).concat(String(")")))

    @langkit_property()
    def eval(ctx=T.Substitution.array):
        return Result.new(
            value=Entity,
            context=ctx
        )

    @langkit_property()
    def apply(ctx=T.Substitution.array, val=T.Term.entity):
        fresh_id = Var(Self.fresh_symbol(Self.ident.sym))

        subst = Var(Substitution.new(
            sym=fresh_id,
            actual=val
        ).singleton)

        return (
            Entity.term
            .substitute(Self.ident.sym, fresh_id)
            .as_entity
            .eval(subst.concat(ctx))
        )

    @langkit_property()
    def substitute(old=T.Symbol, by=T.Symbol):
        return Self.parent.make_abstraction(
            Self.ident.substitute(old, by).cast(Identifier),
            Self.term.substitute(old, by)
        )


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

    @langkit_property(public=True, return_type=T.Term.entity)
    def value():
        return Entity.eval.value

    @langkit_property(return_type=Result, memoized=True)
    def eval():
        return Entity.term.eval(No(Substitution.array))

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
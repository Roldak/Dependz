from __future__ import absolute_import, division, print_function

from langkit.dsl import (
    ASTNode, abstract, Field, T, Bool, LexicalEnv, synthetic, Struct,
    UserField, NullField, Symbol, LogicVar
)
from langkit.envs import EnvSpec, add_env, add_to_env_kv, handle_children
from langkit.expressions import (
    Self, Entity, langkit_property, Property, AbstractProperty, Not, No, If,
    ArrayLiteral, String, Var, AbstractKind, Let, Bind, LogicTrue, Or, And
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

    @langkit_property(return_type=T.DefTerm.entity)
    def normalized_domain():
        return Entity.match(
            lambda t=Term: t.node.normalize,
            lambda a=Arrow: a.parent.make_arrow(
                a.lhs.normalized_domain.node,
                a.rhs.normalized_domain.node
            )
        ).as_entity

    @langkit_property(public=True, return_type=T.Bool)
    def equivalent(other=T.DefTerm.entity):
        return Entity.match(
            lambda id=Identifier: other.cast(Identifier).then(
                lambda o: o.sym == id.sym
            ),
            lambda ap=Apply: other.cast(Apply).then(
                lambda o: And(
                    ap.lhs.equivalent(o.lhs),
                    ap.rhs.equivalent(o.rhs)
                )
            ),
            lambda ab=Abstraction: other.cast(Abstraction).then(
                lambda o: Self.fresh_symbol("eq").then(
                    lambda sym:
                    ab.term.rename(ab.ident.sym, sym).as_entity.equivalent(
                        o.term.rename(o.ident.sym, sym).as_entity
                    )
                )
            ),
            lambda ar=Arrow: other.cast(Arrow).then(
                lambda o: And(
                    ar.lhs.equivalent(o.lhs),
                    ar.rhs.equivalent(o.rhs)
                )
            )
        )


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

    @langkit_property(return_type=T.LogicVar, memoized=True)
    def domain_var():
        return Self.create_logic_var

    @langkit_property(return_type=T.Equation)
    def bind_occurrences(sym=T.Symbol, dest=T.LogicVar):
        return Self.match(
            lambda id=Identifier: If(
                id.sym == sym,
                Bind(id.domain_var, dest,
                     conv_prop=DefTerm.normalized_domain,
                     eq_prop=DefTerm.equivalent),
                LogicTrue()
            ),
            lambda ap=Apply: And(
                ap.lhs.bind_occurrences(sym, dest),
                ap.rhs.bind_occurrences(sym, dest)
            ),
            lambda ab=Abstraction: If(
                ab.ident.sym == sym,
                LogicTrue(),
                ab.term.bind_occurrences(sym, dest)
            )
        )

    @langkit_property(return_type=T.Equation)
    def domain_equation():
        v = Var(Self.domain_var)
        return Self.match(
            lambda id=Identifier: id.intro.then(
                lambda intro: Bind(v, intro.term.normalized_domain,
                                   conv_prop=DefTerm.normalized_domain,
                                   eq_prop=DefTerm.equivalent),
                default_val=LogicTrue()
            ),
            lambda ap=Apply: And(
                ap.lhs.domain_equation,
                ap.rhs.domain_equation,
                Bind(ap.lhs.domain_var, ap.rhs.domain_var,
                     conv_prop=Arrow.param, eq_prop=DefTerm.equivalent),
                Bind(ap.lhs.domain_var, ap.domain_var,
                     conv_prop=Arrow.result, eq_prop=DefTerm.equivalent)
            ),
            lambda ab=Abstraction: And(
                ab.term.bind_occurrences(ab.ident.sym, ab.ident.domain_var),
                Bind(ab.domain_var, ab.ident.domain_var,
                     conv_prop=Arrow.param, eq_prop=DefTerm.equivalent),
                Bind(ab.domain_var, ab.term.domain_var,
                     conv_prop=Arrow.result, eq_prop=DefTerm.equivalent),
                ab.term.domain_equation
            )
        )

    @langkit_property(return_type=T.DefTerm, public=True)
    def domain():
        return Self.domain_var.get_value.node.cast_or_raise(DefTerm)


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

    @langkit_property(return_type=DefTerm.entity)
    def param():
        return Entity.lhs.normalized_domain

    @langkit_property(return_type=DefTerm.entity)
    def result():
        return Entity.rhs.normalized_domain


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

    @langkit_property(public=True, return_type=T.Bool)
    def check_domains():
        domain_eq = And(
            Bind(Self.term.domain_var,
                 Self.ident.intro.term.normalized_domain),
            Self.term.domain_equation
        )
        return domain_eq.solve

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

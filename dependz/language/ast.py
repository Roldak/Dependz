from __future__ import absolute_import, division, print_function

from langkit.dsl import (
    ASTNode, abstract, Field, T, Bool, LexicalEnv, synthetic, Struct,
    UserField, NullField, Symbol, LogicVar, Annotations
)
from langkit.envs import EnvSpec, add_env, add_to_env_kv, handle_children
from langkit.expressions import (
    Self, Entity, langkit_property, Property, AbstractProperty, Not, No, If,
    ArrayLiteral, String, Var, AbstractKind, Let, Bind, LogicTrue, LogicFalse,
    Or, And, PropertyError, ignore, Try, Cond
)


class Renaming(Struct):
    from_symbol = UserField(type=T.Symbol)
    to_symbol = UserField(type=T.Symbol)


class Substitution(Struct):
    from_symbol = UserField(type=T.Symbol)
    to_term = UserField(type=T.DefTerm)


class Template(Struct):
    origin = UserField(type=T.Term)
    instance = UserField(type=T.DefTerm)


class Binding(Struct):
    target = UserField(type=T.Term)
    domain_val = UserField(type=T.DefTerm)


class UnifyEquation(Struct):
    eq = UserField(type=T.Equation)
    renamings = UserField(type=Renaming.array)


class DomainEquation(Struct):
    eq = UserField(type=T.Equation)
    templates = UserField(type=T.Identifier.array)


@abstract
class DependzNode(ASTNode):
    """
    Root node class for Dependz AST nodes.
    """
    @langkit_property(public=True, memoized=True)
    def make_ident(name=T.Symbol):
        return FreshId.new(name=name)

    @langkit_property(public=True, memoized=True)
    def make_apply(t1=T.Term, t2=T.Term):
        return SyntheticApply.new(lhs=t1, rhs=t2)

    @langkit_property(public=True, memoized=True)
    def make_abstraction(id=T.Identifier, rhs=T.Term):
        return SyntheticAbstraction.new(ident=id, term=rhs)

    @langkit_property(public=True, memoized=True)
    def make_arrow(t1=T.DefTerm, t2=T.DefTerm,
                   ident=(T.Identifier, No(T.Identifier))):
        return SyntheticArrow.new(lhs=t1, rhs=t2, binder=ident)

    @langkit_property(external=True, return_type=T.Symbol,
                      uses_entity_info=False, uses_envs=False)
    def fresh_symbol(prefix=T.Symbol):
        pass

    @langkit_property(external=True, return_type=T.LogicVar,
                      uses_entity_info=False, uses_envs=False)
    def create_logic_var():
        pass

    @langkit_property(public=True, external=True, return_type=T.Bool,
                      uses_entity_info=False, uses_envs=False)
    def set_logic_equation_debug_mode(mode=T.Int):
        pass


@synthetic
class LogicVarArray(DependzNode):
    @langkit_property(return_type=T.LogicVar, memoized=True)
    def elem(idx=T.Int):
        return Self.create_logic_var


@abstract
class DefTerm(DependzNode):
    annotations = Annotations(custom_trace_image=True)

    to_string = AbstractProperty(public=True, type=T.String)

    @langkit_property(return_type=T.DefTerm.entity)
    def normalized_domain():
        return Entity.match(
            lambda t=Term: t.node.normalize,
            lambda a=Arrow: a.parent.make_arrow(
                a.lhs.normalized_domain.node,
                a.rhs.normalized_domain.node,
                a.binder.node
            )
        ).as_entity

    @langkit_property(return_type=T.DefTerm, public=True)
    def renamed_domain(old=T.Symbol, by=T.Symbol):
        return Self.match(
            lambda t=Term: t.rename(old, by),
            lambda ar=Arrow: Self.parent.make_arrow(
                ar.lhs.renamed_domain(old, by),
                ar.rhs.renamed_domain(old, by),
                ar.binder._.rename(old, by).cast(Identifier)
            )
        )

    @langkit_property(return_type=T.DefTerm, public=True)
    def substituted_domain(sym=T.Symbol, val=T.DefTerm):
        return Self.match(
            lambda t=Term: t.substitute(sym, val.cast_or_raise(Term)),
            lambda ar=Arrow: Self.parent.make_arrow(
                ar.lhs.substituted_domain(sym, val),
                ar.rhs.substituted_domain(sym, val),
                ar.binder.then(lambda b: If(
                    b.sym == sym,
                    No(Identifier),
                    b
                ))
            )
        )

    @langkit_property(public=True, return_type=T.Bool)
    def equivalent_entities(other=T.DefTerm.entity):
        return Entity.node.equivalent(other.node)

    @langkit_property(return_type=T.Bool)
    def equivalent(other=T.DefTerm):
        return Self.match(
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
                    lambda sym: ab.term.rename(ab.ident.sym, sym).equivalent(
                        o.term.rename(o.ident.sym, sym)
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

    @langkit_property(return_type=UnifyEquation, uses_entity_info=False)
    def unify_equation(other=T.DefTerm,
                       symbols=T.Symbol.array,
                       vars=LogicVarArray):

        def index_of(symbol, then, els):
            return Let(lambda sym=symbol: symbols.filtermap(
                lambda i, e: i,
                lambda e: e == sym
            ).then(
                lambda x: Let(lambda idx=x.at(0): then(idx)),
                default_val=els
            ))

        def to_logic(bool):
            return If(bool, LogicTrue(), LogicFalse())

        def combine(lhs, other_lhs, rhs, other_rhs):
            return Let(
                lambda
                e1=lhs.unify_equation(other_lhs, symbols, vars),
                e2=rhs.unify_equation(other_rhs, symbols, vars):

                UnifyEquation.new(
                    eq=And(e1.eq, e2.eq),
                    renamings=e1.renamings.concat(e2.renamings)
                )
            )

        def unify_case(expected_type, then):
            return other.cast(expected_type).then(
                then,
                default_val=UnifyEquation.new(
                    eq=other.cast(Identifier).then(
                        lambda oid: index_of(
                            oid.sym,
                            lambda idx: Bind(
                                vars.elem(idx), Self.as_bare_entity,
                                eq_prop=DefTerm.equivalent_entities
                            ),
                            LogicFalse()
                        ),
                        default_val=LogicFalse()
                    ),
                    renamings=No(Renaming.array)
                )
            )

        return Self.match(
            lambda id=Identifier: UnifyEquation.new(
                eq=index_of(
                    id.sym,
                    lambda idx: other.cast(Identifier).then(
                        lambda oid: index_of(
                            oid.sym,
                            lambda idx2: Bind(
                                vars.elem(idx), vars.elem(idx2),
                                eq_prop=DefTerm.equivalent_entities
                            ),
                            Bind(vars.elem(idx), other.as_bare_entity,
                                 eq_prop=DefTerm.equivalent_entities)
                        ),
                        default_val=Bind(vars.elem(idx), other.as_bare_entity,
                                         eq_prop=DefTerm.equivalent_entities)
                    ),
                    other.cast(Identifier).then(
                        lambda oid: index_of(
                            oid.sym,
                            lambda idx: Bind(
                                vars.elem(idx), Self.as_bare_entity,
                                eq_prop=DefTerm.equivalent_entities
                            ),
                            to_logic(id.equivalent(other))
                        ),
                        default_val=LogicFalse()
                    )
                ),
                renamings=No(Renaming.array)
            ),

            lambda ab=Abstraction: unify_case(
                Abstraction,
                lambda o: Let(
                    lambda sym=Self.fresh_symbol("eq"): Let(
                        lambda
                        rab=ab.term.rename(ab.ident.sym, sym),
                        rob=o.term.rename(o.ident.sym, sym):

                        Let(
                            lambda r=rab.unify_equation(rob, symbols, vars):
                            UnifyEquation.new(
                                eq=r.eq,
                                renamings=r.renamings.concat(Renaming.new(
                                    from_symbol=sym,
                                    to_symbol=ab.ident.sym
                                ).singleton)
                            )
                        )
                    )
                )
            ),

            lambda ap=Apply: unify_case(
                Apply,
                lambda oap: combine(
                    ap.lhs, oap.lhs,
                    ap.rhs, oap.rhs
                )
            ),

            lambda ar=Arrow: unify_case(
                Arrow,
                lambda oar: combine(
                    ar.lhs, oar.lhs,
                    ar.rhs, oar.rhs
                )
            )
        )

    @langkit_property(return_type=LogicVarArray, memoized=True)
    def make_logic_var_array():
        return LogicVarArray.new()

    @langkit_property(return_type=T.DefTerm)
    def rename_all(renamings=Renaming.array, idx=(T.Int, 0)):
        return renamings.at(idx).then(
            lambda r:
            Self.renamed_domain(r.from_symbol, r.to_symbol).rename_all(
                renamings, idx + 1
            ),
            default_val=Self
        )

    @langkit_property(return_type=T.DefTerm)
    def substitute_all(substs=Substitution.array, idx=(T.Int, 0)):
        return substs.at(idx).then(
            lambda r:
            Self.substituted_domain(r.from_symbol, r.to_term).substitute_all(
                substs, idx + 1
            ),
            default_val=Self
        )

    @langkit_property(public=True, return_type=Substitution.array,
                      activate_tracing=False)
    def unify(other=T.DefTerm, symbols=T.Symbol.array):
        vars = Var(Self.make_logic_var_array)
        unify_eq = Var(Self.unify_equation(
            other,
            symbols,
            vars
        ))
        return If(
            unify_eq.eq.solve,
            symbols.map(
                lambda i, s: Substitution.new(
                    from_symbol=s,
                    to_term=vars.elem(i).get_value._.cast(DefTerm).rename_all(
                        unify_eq.renamings
                    )
                )
            ).filter(lambda s: Not(s.to_term.is_null)),
            PropertyError(Substitution.array, "Unification failed")
        )

    @langkit_property(public=True, return_type=T.Symbol.array)
    def free_symbols():
        def combine(lhs, rhs):
            return Let(
                lambda l=lhs.free_symbols:

                l.concat(rhs.free_symbols.filter(
                    lambda s: Not(l.contains(s))
                ))
            )

        return Self.match(
            lambda id=Identifier: id.intro.then(
                lambda _: No(T.Symbol.array),
                default_val=id.sym.singleton
            ),
            lambda ab=Abstraction: ab.term.free_symbols.filter(
                lambda s: s != ab.ident.sym
            ),
            lambda ap=Apply: combine(ap.lhs, ap.rhs),
            lambda ar=Arrow: combine(ar.lhs, ar.rhs)
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

    @langkit_property(public=True, return_type=T.Term)
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
    def bind_occurrences(sym=T.Symbol, orig=T.LogicVar):
        return Self.match(
            lambda id=Identifier: If(
                id.sym == sym,
                Bind(orig, id.domain_var,
                     conv_prop=DefTerm.normalized_domain,
                     eq_prop=DefTerm.equivalent_entities),
                LogicTrue()
            ),
            lambda ap=Apply: And(
                ap.lhs.bind_occurrences(sym, orig),
                ap.rhs.bind_occurrences(sym, orig)
            ),
            lambda ab=Abstraction: If(
                ab.ident.sym == sym,
                LogicTrue(),
                ab.term.bind_occurrences(sym, orig)
            )
        )

    @langkit_property(return_type=T.DomainEquation,
                      uses_entity_info=False,
                      activate_tracing=False)
    def domain_equation(bindings=Binding.array):
        relevant_binding = Var(bindings.find(
            lambda b: b.target == Self
        ))

        result = Var(Self.match(
            lambda id=Identifier: id.intro.then(
                lambda intro: Cond(
                    Not(relevant_binding.is_null),
                    DomainEquation.new(
                        eq=LogicTrue(),
                        templates=No(Identifier.array)
                    ),

                    intro.generic_formals.length > 0,
                    DomainEquation.new(
                        eq=LogicTrue(),
                        templates=id.singleton
                    ),

                    DomainEquation.new(
                        eq=Bind(
                            Self.domain_var, intro.term.normalized_domain,
                            conv_prop=DefTerm.normalized_domain,
                            eq_prop=DefTerm.equivalent_entities
                        ),
                        templates=No(Identifier.array)
                    )
                ),
                default_val=DomainEquation.new(
                    eq=LogicTrue(),
                    templates=No(Identifier.array)
                )
            ),
            lambda ap=Apply: Let(
                lambda
                lhs_eq=ap.lhs.domain_equation(bindings),
                rhs_eq=ap.rhs.domain_equation(bindings):

                DomainEquation.new(
                    eq=And(
                        lhs_eq.eq,
                        rhs_eq.eq,
                        Bind(ap.lhs.domain_var, ap.rhs.domain_var,
                             conv_prop=Arrow.param,
                             eq_prop=DefTerm.equivalent_entities),
                        Bind(ap.lhs.domain_var, ap.domain_var,
                             conv_prop=Arrow.result,
                             eq_prop=DefTerm.equivalent_entities)
                    ),
                    templates=lhs_eq.templates.concat(rhs_eq.templates)
                )
            ),
            lambda ab=Abstraction: Let(
                lambda term_eq=ab.term.domain_equation(bindings):

                DomainEquation.new(
                    eq=And(
                        ab.term.bind_occurrences(
                            ab.ident.sym, ab.ident.domain_var
                        ),
                        Bind(ab.domain_var, ab.ident.domain_var,
                             conv_prop=Arrow.param,
                             eq_prop=DefTerm.equivalent_entities),
                        Bind(ab.domain_var, ab.term.domain_var,
                             conv_prop=Arrow.result,
                             eq_prop=DefTerm.equivalent_entities),
                        term_eq.eq
                    ),
                    templates=term_eq.templates
                )
            )
        ))

        return relevant_binding.then(
            lambda b: DomainEquation.new(
                eq=Bind(
                    Self.domain_var, b.domain_val.as_bare_entity,
                    eq_prop=DefTerm.equivalent_entities
                ) & result.eq,
                templates=result.templates
            ),
            default_val=result
        )

    @langkit_property(public=True, return_type=Binding.array,
                      activate_tracing=False)
    def instantiate_templates(templates=Template.array,
                              formals=T.Symbol.array):
        def make_binding(term, domain):
            return Binding.new(
                target=term,
                domain_val=domain
            )

        templated_result = Var(Self.match(
            lambda id=Identifier: make_binding(
                id,
                templates.find(lambda t: t.origin == id).then(
                    lambda t: t.instance,
                    default_val=id.domain_val
                )
            ).singleton,

            lambda ap=Apply: Let(
                lambda
                lhs_res=ap.lhs.instantiate_templates(templates, formals),
                rhs_res=ap.rhs.instantiate_templates(templates, formals):

                Let(
                    lambda
                    substs=lhs_res.at(0).domain_val.cast(Arrow).lhs.unify(
                        rhs_res.at(0).domain_val, formals
                    ):

                    Let(
                        lambda
                        new_bindings=lhs_res.concat(rhs_res).map(
                            lambda b: make_binding(
                                b.target,
                                b.domain_val.substitute_all(substs)
                            )
                        ):

                        make_binding(
                            ap,
                            new_bindings.at(0).domain_val.cast_or_raise(Arrow)
                            .rhs
                        ).singleton.concat(new_bindings)
                    )
                )
            ),

            lambda ab=Abstraction: Let(
                lambda
                term_res=ab.term.instantiate_templates(templates, formals):

                make_binding(
                    ab,
                    Self.parent.make_arrow(
                        ab.ident.domain_val,
                        term_res.at(0).domain_val
                    )
                ).singleton.concat(term_res)
            )
        ))

        return Self.domain_val.then(
            lambda expected_dom: Let(
                lambda substs=templated_result.at(0).domain_val.unify(
                    expected_dom, formals
                ): templated_result.map(
                    lambda b: make_binding(
                        b.target,
                        b.domain_val.substitute_all(substs).then(
                            lambda r: Let(lambda str=r.to_string: r)
                        )
                    )
                )
            ),
            default_val=templated_result
        )

    @langkit_property(return_type=T.DefTerm, public=True)
    def domain_val():
        return Self.domain_var.get_value._.node.cast_or_raise(DefTerm)


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


class Apply(Term):
    lhs = Field(type=Term)
    rhs = Field(type=Term)

    to_string = Property(String("(").concat(
        Self.lhs.to_string.concat(String(' ')).concat(Self.rhs.to_string)
    ).concat(String(")")))


@synthetic
class SyntheticApply(Apply):
    pass


class Abstraction(Term):
    ident = Field(type=Identifier)
    term = Field(type=Term)

    to_string = Property(String("(\\").concat(
        Self.ident.to_string.concat(String('. ')).concat(Self.term.to_string)
    ).concat(String(")")))


@synthetic
class SyntheticAbstraction(Abstraction):
    pass


class Arrow(DefTerm):
    binder = Field(type=Identifier)
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


@synthetic
class SyntheticArrow(Arrow):
    pass


class Introduction(DependzNode):
    """
    Identifer : DefTerm
    """
    ident = Field(type=SourceId)
    term = Field(type=DefTerm)

    @langkit_property(public=True, return_type=T.Definition.entity)
    def definition():
        return Self.children_env.get_first('__definition').cast(T.Definition)

    @langkit_property(public=True, return_type=T.Symbol.array,
                      memoized=True)
    def generic_formals():
        return Self.term.free_symbols

    @langkit_property(public=True, return_type=Template)
    def as_template(origin_term=T.Term):
        renamings = Var(Self.generic_formals.map(lambda s: Renaming.new(
            from_symbol=s,
            to_symbol=Self.fresh_symbol(s)
        )))
        return Template.new(
            origin=origin_term,
            instance=Self.term.rename_all(renamings).as_bare_entity
            .normalized_domain.node,
        )

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
    def check_domains(tries=(T.Int, -1)):
        return Self.check_domains_internal(No(Binding.array), tries)

    @langkit_property(public=False, return_type=T.Bool)
    def check_domains_internal(bindings=Binding.array, tries=T.Int):
        term_eq = Var(Self.term.domain_equation(bindings))
        domain_eq = And(
            Bind(Self.term.domain_var,
                 Self.ident.intro.term.normalized_domain),
            term_eq.eq
        )
        self_formals = Self.ident.intro.generic_formals
        return term_eq.templates.then(
            lambda templates: Try(
                domain_eq.solve,
                tries != 0
            ).then(lambda _: Let(
                lambda
                instances=templates.map(lambda t: t.intro.as_template(t)):

                Self.term.instantiate_templates(
                    instances,
                    instances.mapcat(lambda i: i.instance.free_symbols)
                ).then(lambda result: Self.check_domains_internal(
                    result.filter(
                        lambda b: b.domain_val.free_symbols.all(
                            lambda sym: self_formals.contains(sym)
                        )
                    ),
                    tries - 1
                ))
            )),
            default_val=domain_eq.solve
        )

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

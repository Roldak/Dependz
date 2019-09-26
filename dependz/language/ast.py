from __future__ import absolute_import, division, print_function

from langkit.dsl import (
    ASTNode, abstract, Field, T, Bool, LexicalEnv, synthetic, Struct,
    UserField, NullField, Symbol, LogicVar, Annotations
)
from langkit.envs import EnvSpec, add_env, add_to_env_kv, handle_children
from langkit.expressions import (
    Self, Entity, langkit_property, Property, AbstractProperty, Not, No, If,
    ArrayLiteral, String, Var, AbstractKind, Let, Bind, LogicTrue, LogicFalse,
    Or, And, PropertyError, ignore, Try, Cond, Predicate, DynamicVariable,
    bind, CharacterLiteral
)


GLOBAL_ACTIVATE_TRACING = False


class Renaming(Struct):
    from_symbol = UserField(type=T.Symbol)
    to_symbol = UserField(type=T.Symbol)


class Substitution(Struct):
    from_symbol = UserField(type=T.Symbol)
    to_term = UserField(type=T.DefTerm)


class Template(Struct):
    origin = UserField(type=T.Term)
    instance = UserField(type=T.DefTerm)
    new_symbols = UserField(type=T.Symbol.array)


class Binding(Struct):
    target = UserField(type=T.Term)
    domain_val = UserField(type=T.DefTerm)


class UnifyEquation(Struct):
    eq = UserField(type=T.Equation)
    renamings = UserField(type=Renaming.array)


class DomainEquation(Struct):
    eq = UserField(type=T.Equation)
    templates = UserField(type=T.Identifier.array)


class UnifyQuery(Struct):
    first = UserField(type=T.DefTerm)
    second = UserField(type=T.DefTerm)


class TypingsDescription(Struct):
    bindings = UserField(type=T.Binding.array)
    equations = UserField(type=T.UnifyQuery.array)


class UnificationContext(Struct):
    symbols = UserField(type=T.Symbol.array)
    vars = UserField(type=T.LogicVarArray)


class HigherOrderUnificationContext(Struct):
    arg = UserField(type=T.Term)
    res = UserField(type=T.Term)


unification_context = DynamicVariable(
    "unification_context", type=UnificationContext
)

ho_unification_context = DynamicVariable(
    "ho_unification_context", type=HigherOrderUnificationContext
)


@abstract
class DependzNode(ASTNode):
    """
    Root node class for Dependz AST nodes.
    """
    @langkit_property(public=True)
    def make_ident(name=T.Symbol):
        return Self.unit.root.make_ident_from_self(name)

    @langkit_property(public=True)
    def make_apply(t1=T.Term, t2=T.Term):
        return Self.unit.root.make_apply_from_self(t1, t2)

    @langkit_property(public=True)
    def make_abstraction(id=T.Identifier, rhs=T.Term):
        return Self.unit.root.make_abstraction_from_self(id, rhs)

    @langkit_property(public=True)
    def make_arrow(t1=T.DefTerm, t2=T.DefTerm,
                   t3=(T.Term, No(T.Term))):
        return Self.unit.root.make_arrow_from_self(t1, t2, t3)

    @langkit_property(return_type=T.LogicVarArray, memoized=True)
    def make_logic_var_array():
        return LogicVarArray.new()

    @langkit_property(memoized=True, return_type=T.FreshId)
    def make_ident_from_self(name=T.Symbol):
        return FreshId.new(name=name)

    @langkit_property(memoized=True, return_type=T.SyntheticApply)
    def make_apply_from_self(t1=T.Term, t2=T.Term):
        return SyntheticApply.new(lhs=t1, rhs=t2)

    @langkit_property(memoized=True, return_type=T.SyntheticAbstraction)
    def make_abstraction_from_self(id=T.Identifier, rhs=T.Term):
        return SyntheticAbstraction.new(ident=id, term=rhs)

    @langkit_property(memoized=True, return_type=T.SyntheticArrow)
    def make_arrow_from_self(t1=T.DefTerm, t2=T.DefTerm,
                             t3=(T.Term, No(T.Term))):
        return SyntheticArrow.new(lhs=t1, rhs=t2, binder=t3)

    @langkit_property(return_type=T.Symbol)
    def unique_fresh_symbol(prefix=T.Symbol):
        return Self.concat_symbol_and_integer(
            prefix,
            Self.unit.root.next_global_integer
        )

    @langkit_property(external=True, return_type=T.Int,
                      uses_entity_info=False, uses_envs=False)
    def next_global_integer():
        pass

    @langkit_property(external=True, return_type=T.Symbol,
                      uses_entity_info=False, uses_envs=False)
    def concat_symbol_and_integer(s=T.Symbol, i=T.Int):
        pass

    @langkit_property(external=True, return_type=T.LogicVar,
                      uses_entity_info=False, uses_envs=False)
    def create_logic_var():
        pass

    @langkit_property(public=True, external=True, return_type=T.Bool,
                      uses_entity_info=False, uses_envs=False)
    def set_logic_equation_debug_mode(mode=T.Int):
        pass

    @langkit_property(return_type=T.DependzNode, activate_tracing=True)
    def here():
        return Self

    @langkit_property(public=True, external=True, return_type=T.Bool,
                      uses_entity_info=False, uses_envs=False)
    def dump_mmz_map():
        pass

    @langkit_property(return_type=Substitution.array,
                      activate_tracing=GLOBAL_ACTIVATE_TRACING)
    def unify_all(queries=UnifyQuery.array, symbols=T.Symbol.array,
                  allow_incomplete=(T.Bool, False)):
        vars = Var(Self.make_logic_var_array)

        def construct_equations():
            return queries.map(
                lambda q: unification_context.bind(
                    UnificationContext.new(
                        symbols=symbols,
                        vars=vars
                    ),
                    q.first.unify_equation(q.second)
                )
            )

        query_results = Var(construct_equations())
        unify_eq = Var(Or(
            query_results.logic_all(lambda r: r.eq),
            LogicTrue()
        ))
        renamings = Var(query_results.mapcat(
            lambda r: r.renamings
        ))

        res = Var(Try(
            unify_eq.solve._or(PropertyError(Bool, "Cannot happen")),
            False
        ))

        substs = Var(
            symbols.map(
                lambda s: Substitution.new(
                    from_symbol=s,
                    to_term=vars.elem(s).get_value._.cast(DefTerm).rename_all(
                        renamings
                    )
                )
            ).filter(lambda s: Not(s.to_term.is_null))
        )
        new_queries = Var(queries.map(
            lambda q: UnifyQuery.new(
                first=q.first.substitute_all(substs).dnorm,
                second=q.second.substitute_all(substs).dnorm
            )
        ))
        left_symbols = Var(symbols.filter(
            lambda sym: Not(substs.exists(
                lambda subst: subst.from_symbol == sym
            ))
        ))
        incomplete = And(
            left_symbols.length > 0,
            queries.any(lambda i, old_q: Let(
                lambda new_q=new_queries.at(i): Or(
                    Not(old_q.first.equivalent(new_q.first)),
                    Not(old_q.second.equivalent(new_q.second))
                )
            ))
        )
        return Cond(
            incomplete,
            substs.concat(
                Self.unify_all(new_queries, left_symbols, allow_incomplete)
            ),

            res | allow_incomplete,
            Try(
                construct_equations().logic_all(lambda r: r.eq).solve,
                True
            ).then(
                lambda _: substs,
                default_val=PropertyError(
                    Substitution.array,
                    "Unification failed (terms are not unifiable)"
                )
            ),

            PropertyError(
                Substitution.array,
                "Unification failed (not all metavariables could be bound)"
            )
        )


@synthetic
class LogicVarArray(DependzNode):
    @langkit_property(return_type=T.LogicVar, memoized=True)
    def elem(s=T.Symbol):
        return Self.create_logic_var


@abstract
class DefTerm(DependzNode):
    annotations = Annotations(custom_trace_image=True)

    to_string = AbstractProperty(public=True, type=T.String)

    @langkit_property(return_type=T.DefTerm.entity)
    def normalized_domain():
        return Entity.node.dnorm.as_entity

    @langkit_property(return_type=T.DefTerm, memoized=True)
    def dnorm():
        return Self.match(
            lambda t=Term: t.normalize,
            lambda a=Arrow: a.make_arrow(
                a.lhs.dnorm,
                a.rhs.dnorm,
                a.binder._.normalize
            )
        )

    @langkit_property(return_type=T.DefTerm, public=True, memoized=True)
    def renamed_domain(old=T.Symbol, by=T.Symbol):
        return Self.match(
            lambda t=Term: t.rename(old, by),
            lambda ar=Arrow: Self.make_arrow(
                ar.lhs.renamed_domain(old, by),
                ar.rhs.renamed_domain(old, by),
                ar.binder._.rename(old, by)
            )
        )

    @langkit_property(return_type=T.DefTerm, public=True, memoized=True)
    def substituted_domain(sym=T.Symbol, val=T.DefTerm):
        return Self.match(
            lambda t=Term: t.substitute(sym, val.cast_or_raise(Term)),
            lambda ar=Arrow: Self.make_arrow(
                ar.lhs.substituted_domain(sym, val),
                ar.rhs.substituted_domain(sym, val),
                ar.binder._.substitute(sym, val.cast_or_raise(Term))
            )
        )

    @langkit_property(public=True, return_type=T.Bool)
    def equivalent_entities(other=T.DefTerm.entity):
        return Entity.node.equivalent(other.node)

    @langkit_property(return_type=T.Bool, memoized=True)
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
                lambda o: ab.term.free_fresh_symbol("eq", o.term).then(
                    lambda sym:
                    ab.term.rename(ab.ident.sym, sym).equivalent(
                        o.term.rename(o.ident.sym, sym)
                    )
                )
            ),
            lambda ar=Arrow: other.cast(Arrow).then(
                lambda o: And(
                    ar.lhs.equivalent(o.lhs),
                    ar.rhs.equivalent(o.rhs),
                    ar.binder.then(
                        lambda b: o.binder.then(
                            lambda ob: b.equivalent(ob),
                            default_val=Not(ar.has_constraining_binder)
                        ),
                        default_val=Not(o.has_constraining_binder)
                    )
                )
            )
        )

    @langkit_property(return_type=T.DefTerm,
                      dynamic_vars=[unification_context])
    def solve_time_substitution():
        symbols = Var(unification_context.symbols)
        vars = Var(unification_context.vars)

        substs = Var(symbols.map(
            lambda s: Substitution.new(
                from_symbol=s,
                to_term=vars.elem(s).get_value._.cast_or_raise(DefTerm).node
            )
        ).filter(
            lambda s: Not(s.to_term.is_null)
        ))

        return Self.substitute_all(substs).dnorm

    @langkit_property(return_type=T.Bool,
                      dynamic_vars=[unification_context],
                      activate_tracing=GLOBAL_ACTIVATE_TRACING)
    def unifies_with(other=T.DefTerm):
        current_self = Var(Self.solve_time_substitution.dnorm)
        current_other = Var(other.solve_time_substitution.dnorm)
        return Try(
            Let(
                lambda substs=current_self.unify(
                    current_other,
                    unification_context.symbols,
                    allow_incomplete=True
                ): True
            ),
            False
        )

    @langkit_property(return_type=UnifyEquation, uses_entity_info=False,
                      dynamic_vars=[unification_context])
    def first_order_flexible_flexible_equation(other=T.DefTerm):
        vars = Var(unification_context.vars)
        self_var = Var(vars.elem(Self.cast(Identifier).sym))
        other_var = Var(vars.elem(other.cast(Identifier).sym))

        return UnifyEquation.new(
            eq=Bind(self_var, other_var, eq_prop=DefTerm.equivalent_entities),
            renamings=No(Renaming.array)
        )

    @langkit_property(return_type=UnifyEquation, uses_entity_info=False,
                      dynamic_vars=[unification_context])
    def first_order_flexible_semirigid_equation(other=T.DefTerm):
        self_var = Var(
            unification_context.vars.elem(Self.cast(Identifier).sym)
        )

        return UnifyEquation.new(
            eq=And(
                Predicate(DefTerm.unifies_with, self_var, other),
                LogicTrue()  # Bind(vars_in_flexible_eq, Self, conv_prop=...)
            ),
            renamings=No(Renaming.array)
        )

    @langkit_property(return_type=UnifyEquation, uses_entity_info=False,
                      dynamic_vars=[unification_context])
    def first_order_flexible_rigid_equation(other=T.DefTerm):
        self_var = Var(
            unification_context.vars.elem(Self.cast(Identifier).sym)
        )

        return UnifyEquation.new(
            eq=Bind(
                self_var,
                other.as_bare_entity,
                eq_prop=DefTerm.equivalent_entities
            ),
            renamings=No(Renaming.array)
        )

    @langkit_property(return_type=UnifyEquation, uses_entity_info=False,
                      dynamic_vars=[unification_context])
    def first_order_rigid_rigid_equation(other=T.DefTerm):

        def to_logic(bool):
            return If(bool, LogicTrue(), LogicFalse())

        def combine(x, y, *others):
            assert (len(others) % 2 == 0)

            if len(others) == 0:
                return x.unify_equation(y)
            else:
                return Let(
                    lambda
                    e1=x.unify_equation(y),
                    e2=combine(*others):

                    UnifyEquation.new(
                        eq=And(e1.eq, e2.eq),
                        renamings=e1.renamings.concat(e2.renamings)
                    )
                )

        def unify_case(expected_type, then):
            return other.cast(expected_type).then(
                then,
                default_val=UnifyEquation.new(
                    eq=LogicFalse(),
                    renamings=No(Renaming.array)
                )
            )

        return Self.match(
            lambda id=Identifier: unify_case(
                Identifier,
                lambda oid: UnifyEquation.new(
                    eq=to_logic(id.sym == oid.sym),
                    renamings=No(Renaming.array)
                )
            ),

            lambda ab=Abstraction: unify_case(
                Abstraction,
                lambda o: Let(
                    lambda sym=ab.term.free_fresh_symbol("eq", o.term): Let(
                        lambda
                        rab=ab.term.rename(ab.ident.sym, sym),
                        rob=o.term.rename(o.ident.sym, sym):

                        Let(
                            lambda r=rab.unify_equation(rob):
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
                lambda oar: Cond(
                    Or(
                        And(ar.binder.is_null, oar.binder.is_null),
                        Or(
                            ar.binder.is_null
                            & Not(oar.has_constraining_binder),
                            oar.binder.is_null
                            & Not(ar.has_constraining_binder)
                        )
                    ),
                    combine(
                        ar.lhs, oar.lhs,
                        ar.rhs, oar.rhs
                    ),

                    Or(ar.binder.is_null, oar.binder.is_null),
                    UnifyEquation.new(
                        eq=LogicTrue(),
                        renamings=No(Renaming.array)
                    ),

                    combine(
                        ar.binder, oar.binder,
                        ar.lhs, oar.lhs,
                        ar.rhs, oar.rhs
                    )
                )
            )
        )

    @langkit_property(return_type=UnifyEquation, uses_entity_info=False,
                      dynamic_vars=[unification_context])
    def first_order_unify_equation(other=T.DefTerm):
        symbols = Var(unification_context.symbols)

        self_is_metavar = Var(Self.cast(Identifier).then(
            lambda id: symbols.contains(id.sym)
        ))
        other_is_metavar = Var(other.cast(Identifier).then(
            lambda id: symbols.contains(id.sym)
        ))
        self_has_metavar = Var(symbols.any(lambda s: Self.is_free(s)))
        other_has_metavar = Var(symbols.any(lambda s: other.is_free(s)))

        return Cond(
            self_is_metavar & other_is_metavar,
            Self.first_order_flexible_flexible_equation(other),

            self_is_metavar & other_has_metavar,
            Self.first_order_flexible_semirigid_equation(other),

            other_is_metavar & self_has_metavar,
            other.first_order_flexible_semirigid_equation(Self),

            self_is_metavar,
            Self.first_order_flexible_rigid_equation(other),

            other_is_metavar,
            other.first_order_flexible_rigid_equation(Self),

            Self.first_order_rigid_rigid_equation(other)
        )

    @langkit_property(return_type=T.Bool,
                      dynamic_vars=[unification_context,
                                    ho_unification_context],
                      activate_tracing=GLOBAL_ACTIVATE_TRACING)
    def higher_order_check_current_solution():
        return Entity.make_apply(
            Self.cast_or_raise(Term),
            ho_unification_context.arg
        ).unifies_with(
            ho_unification_context.res
        )

    @langkit_property(return_type=T.DefTerm.entity,
                      dynamic_vars=[unification_context,
                                    ho_unification_context])
    def higher_order_construct_imitation():
        res = Var(ho_unification_context.res)
        body = Var(res.solve_time_substitution.cast_or_raise(Term))
        fresh_sym = Var(body.free_fresh_symbol("ho"))

        return Entity.make_abstraction(
            Self.make_ident(fresh_sym),
            body
        ).as_bare_entity

    @langkit_property(return_type=T.DefTerm.entity,
                      dynamic_vars=[unification_context,
                                    ho_unification_context])
    def higher_order_construct_projection():
        arg = Var(ho_unification_context.arg)
        res = Var(ho_unification_context.res)
        fresh_sym = Var(res.free_fresh_symbol("ho"))

        return Entity.make_abstraction(
            Self.make_ident(fresh_sym),
            res.solve_time_substitution.cast_or_raise(Term).anti_substitute(
                arg.solve_time_substitution.cast_or_raise(Term),
                fresh_sym
            )
        ).as_bare_entity

    @langkit_property(return_type=T.UnifyEquation,
                      activate_tracing=GLOBAL_ACTIVATE_TRACING,
                      dynamic_vars=[unification_context])
    def higher_order_single_arg_equation(arg=T.Term, res=T.Term,
                                         ho_sym=T.Symbol):
        metavar = Var(unification_context.vars.elem(ho_sym))

        ho_ctx = Var(HigherOrderUnificationContext.new(
            arg=arg,
            res=res
        ))

        imitate = Var(ho_unification_context.bind(
            ho_ctx,
            Bind(
                metavar,
                Self.as_bare_entity,
                eq_prop=DefTerm.equivalent_entities,
                conv_prop=DefTerm.higher_order_construct_imitation
            )
        ))

        project = Var(ho_unification_context.bind(
            ho_ctx,
            Bind(
                metavar,
                Self.as_bare_entity,
                eq_prop=DefTerm.equivalent_entities,
                conv_prop=DefTerm.higher_order_construct_projection
            )
        ))

        ignore = Var(ho_unification_context.bind(
            ho_ctx,
            Predicate(
                DefTerm.higher_order_check_current_solution,
                metavar
            )
        ))

        return UnifyEquation.new(
            eq=Or(
                imitate,
                project,
                ignore
            ),
            renamings=No(Renaming.array)
        )

    @langkit_property(return_type=T.UnifyEquation,
                      dynamic_vars=[unification_context])
    def higher_order_unify_equation(other=T.DefTerm, ho_term=T.Identifier):
        is_single_arg_equation = Var(And(
            Self.cast(Apply).lhs == ho_term,
            other.is_a(Term)
        ))
        return Cond(
            is_single_arg_equation,
            Self.higher_order_single_arg_equation(
                Self.cast(Apply).rhs,
                other.cast_or_raise(Term),
                ho_term.sym
            ),

            UnifyEquation.new(
                eq=LogicTrue(),
                renamings=No(Renaming.array)
            )
        )

    @langkit_property(return_type=UnifyEquation, uses_entity_info=False,
                      dynamic_vars=[unification_context])
    def unify_equation(other=T.DefTerm):
        symbols = Var(unification_context.symbols)

        def outermost_metavar_application_of(term):
            return term.cast(Apply)._.left_most_term.cast(Identifier).then(
                lambda id: If(
                    # Check the id is indeed a metavariable, and that it's the
                    # first time we discover this higher order application.
                    # (if it's not, it means term.parent must not be an Apply).
                    And(symbols.contains(id.sym),
                        Not(term.parent.cast(Apply)._.lhs == term)),
                    id,
                    No(Identifier)
                )
            )

        self_hoa = Var(outermost_metavar_application_of(Self))
        other_hoa = Var(outermost_metavar_application_of(other))

        unify_eqs = Var(
            Self.first_order_unify_equation(other).singleton
            .concat(self_hoa.then(
                lambda id:
                Self.higher_order_unify_equation(other, id).singleton
            )).concat(other_hoa.then(
                lambda id:
                other.higher_order_unify_equation(Self, id).singleton
            ))
        )

        return UnifyEquation.new(
            eq=unify_eqs.logic_any(lambda eq: eq.eq),
            renamings=unify_eqs.mapcat(lambda eq: eq.renamings)
        )

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
            lambda r: Self.substituted_domain(
                r.from_symbol,
                r.to_term
            ).substitute_all(
                substs, idx + 1
            ),
            default_val=Self
        )

    @langkit_property(public=True, return_type=Substitution.array)
    def unify(other=T.DefTerm, symbols=T.Symbol.array,
              allow_incomplete=(T.Bool, False)):
        return Self.unify_all(UnifyQuery.new(
            first=Self,
            second=other
        ).singleton, symbols, allow_incomplete)

    @langkit_property(public=True, return_type=T.Symbol.array, memoized=True)
    def free_symbols(deep=(T.Bool, False)):
        return Self.free_symbols_impl(deep, 0)

    @langkit_property(public=True, return_type=T.Symbol.array)
    def free_symbols_impl(deep=T.Bool, cur_depth=T.Int):
        def combine(l, r):
            return l.concat(r.filter(
                lambda s: Not(l.contains(s))
            ))

        def rec(node, inc_depth, then):
            return Let(
                lambda l=node.free_symbols_impl(
                    deep, (cur_depth + 1) if inc_depth else cur_depth
                ): then(l)
            )

        return Self.match(
            lambda id=Identifier: id.intro.then(
                lambda _: No(T.Symbol.array),
                default_val=id.sym.singleton
            ),

            lambda ab=Abstraction:
            ab.term.free_symbols_impl(deep, cur_depth).filter(
                lambda s: s != ab.ident.sym
            ),

            lambda ap=Apply: rec(
                ap.lhs, False,
                lambda l: rec(
                    ap.rhs, False,
                    lambda r: combine(l, r)
                )
            ),

            lambda ar=Arrow: If(
                And(Not(deep), cur_depth > 0),
                No(T.Symbol.array),
                rec(
                    ar.lhs, True,
                    lambda l: rec(
                        ar.rhs, False,
                        lambda r: ar.binder.then(
                            lambda b: rec(
                                b, False,
                                lambda x: combine(x, combine(l, r))
                            ),
                            default_val=combine(l, r)
                        )
                    )
                )
            )
        )

    @langkit_property(return_type=T.Bool, public=True)
    def is_free(sym=T.Symbol):
        return Self.match(
            lambda id=Identifier:
            id.sym == sym,

            lambda ap=Apply:
            ap.lhs.is_free(sym) | ap.rhs.is_free(sym),

            lambda ab=Abstraction:
            (ab.ident.sym != sym) & ab.term.is_free(sym),

            lambda ar=Arrow:
            ar.lhs.is_free(sym)
            | ar.rhs.is_free(sym)
            | ar.binder._.is_free(sym)
        )

    @langkit_property(return_type=T.Symbol)
    def free_fresh_symbol(prefix=T.Symbol, other=(T.DefTerm, No(T.DefTerm)),
                          i=(T.Int, 0)):
        sym = Var(Self.concat_symbol_and_integer(prefix, i))
        return If(
            And(Self.is_free(sym), Or(other.is_null, other.is_free(sym))),
            Self.free_fresh_symbol(prefix, other, i + 1),
            sym
        )


@abstract
class Term(DefTerm):
    @langkit_property(return_type=T.Term)
    def eval_case(matches=T.Term, then_case=T.Term, else_case=T.Term):
        constr = Var(matches.cast_or_raise(Identifier))
        return Cond(
            Self.cast(Identifier)._.sym == constr.sym,
            then_case.eval,

            Self.cast(Apply)._.left_most_term.cast(Identifier)
            ._.sym == constr.sym,
            Self.cast(Apply).replace_left_most_term_with(then_case).eval,

            else_case._.extract_case_and_eval(Self)
        )

    @langkit_property(return_type=T.Term)
    def extract_case_and_eval(arg=T.Term):
        case_expr = Var(Self.cast_or_raise(Apply))
        case_id = Var(case_expr.left_most_term.cast(Identifier))
        case_lhs = Var(case_expr.lhs.cast_or_raise(Apply))
        has_else = Var(case_lhs.lhs != case_id)
        return If(
            case_id._.sym == "case",
            If(
                has_else,
                arg.eval_case(
                    case_lhs.lhs.cast_or_raise(Apply).rhs,
                    case_lhs.rhs,
                    case_expr.rhs
                ),
                arg.eval_case(
                    case_lhs.rhs,
                    case_expr.rhs,
                    No(T.Term)
                )
            ),
            PropertyError(T.Term, "expected `case`")
        )

    @langkit_property(return_type=T.Term, memoized=True)
    def eval_match():
        elim_call = Var(Self.cast_or_raise(Apply))
        elim_evaled = Var(elim_call.lhs.eval.cast(Apply))
        elim_id = Var(elim_evaled._.left_most_term)
        return If(
            elim_id.cast(Identifier)._.sym == "match",
            elim_evaled.rhs.extract_case_and_eval(
                elim_call.rhs.eval
            )._or(Self),
            Self
        )

    @langkit_property(public=True, return_type=T.Term, memoized=True)
    def eval():
        return Self.match(
            lambda id=Identifier: id.intro._.definition.then(
                lambda d: d.term.node.eval,
                default_val=id
            ),
            lambda ap=Apply: ap.lhs.eval.cast(Abstraction).then(
                lambda ab: ab.term.substitute(ab.ident.sym, ap.rhs).eval,
                default_val=ap.eval_match
            ),
            lambda ab=Abstraction: ab.term.cast(Apply).then(
                lambda ap: If(
                    And(
                        ap.rhs.cast(Identifier)._.sym == ab.ident.sym,
                        Not(ap.lhs.is_free(ab.ident.sym))
                    ),
                    ap.lhs.eval,
                    ab
                ),
            )._or(ab)
        )

    @langkit_property(public=True, return_type=T.Term, memoized=True)
    def substitute(sym=T.Symbol, val=T.Term):
        return Self.match(
            lambda id=Identifier: If(
                id.sym == sym,
                val,
                id
            ),
            lambda ap=Apply: ap.make_apply(
                ap.lhs.substitute(sym, val),
                ap.rhs.substitute(sym, val)
            ),
            lambda ab=Abstraction: If(
                ab.ident.sym == sym,
                ab,
                If(
                    val.is_free(ab.ident.sym),
                    ab.term.free_fresh_symbol(ab.ident.sym).then(
                        lambda symp: ab.make_abstraction(
                            ab.make_ident(symp),
                            ab.term
                            .rename(ab.ident.sym, symp)
                            .substitute(sym, val)
                        )
                    ),
                    ab.make_abstraction(
                        ab.ident,
                        ab.term.substitute(sym, val)
                    )
                )
            )
        )

    @langkit_property(return_type=T.Term, memoized=True)
    def anti_substitute(val=T.Term, sym=T.Symbol):
        return If(
            Self.equivalent(val),
            Self.make_ident(sym),
            Self.match(
                lambda id=Identifier: id,
                lambda ap=Apply: ap.make_apply(
                    ap.lhs.anti_substitute(val, sym),
                    ap.rhs.anti_substitute(val, sym)
                ),
                lambda ab=Abstraction: ab.make_abstraction(
                    ab.ident.anti_substitute(val, sym)
                    .cast_or_raise(Identifier),
                    ab.term.anti_substitute(val, sym)
                )
            )
        )

    @langkit_property(return_type=T.Bool, memoized=True)
    def contains_term(t=T.Term, include_self=(T.Bool, False)):
        return If(
            include_self & Self.equivalent(t),
            True,
            Self.match(
                lambda id=Identifier: False,
                lambda ap=Apply: Or(
                    ap.lhs.contains_term(t, True),
                    ap.rhs.contains_term(t, True)
                ),
                lambda ab=Abstraction: Or(
                    ab.ident.contains_term(t, True),
                    ab.term.contains_term(t, True)
                )
            )
        )

    @langkit_property(public=True, return_type=T.Term, memoized=True)
    def normalize():
        evaled = Var(Self.eval)

        return If(
            # prevent infinite evaluation
            evaled.contains_term(Self),
            Self,
            evaled.match(
                lambda id=Identifier: id,
                lambda ap=Apply: ap.make_apply(
                    ap.lhs.normalize,
                    ap.rhs.normalize
                ),
                lambda ab=Abstraction: ab.make_abstraction(
                    ab.ident,
                    ab.term.normalize
                )
            )
        )

    @langkit_property(return_type=T.Term, public=True, memoized=True)
    def rename(old=T.Symbol, by=T.Symbol):
        return Self.match(
            lambda id=Identifier: If(
                id.sym == old,
                id.make_ident(by),
                id
            ),
            lambda ap=Apply: ap.make_apply(
                ap.lhs.rename(old, by),
                ap.rhs.rename(old, by)
            ),
            lambda ab=Abstraction: If(
                old == ab.ident.sym,
                ab,
                ab.make_abstraction(
                    ab.ident,
                    ab.term.rename(old, by)
                )
            )
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
                      activate_tracing=GLOBAL_ACTIVATE_TRACING)
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

    @langkit_property(public=False, return_type=TypingsDescription,
                      activate_tracing=GLOBAL_ACTIVATE_TRACING)
    def instantiate_templates(result_domain=T.DefTerm,
                              templates=Template.array,
                              reps=T.Substitution.array):
        def make_binding(domain):
            return Binding.new(
                target=Self,
                domain_val=domain
            )

        def rec_apply(ap, f):
            return Let(
                lambda
                lhs_res=ap.lhs.instantiate_templates(
                    Self.make_arrow(
                        No(T.DefTerm),
                        result_domain
                    ),
                    templates,
                    reps
                ): Let(
                    lambda
                    rhs_res=ap.rhs.instantiate_templates(
                        lhs_res.bindings.at(0).domain_val.cast(Arrow).lhs,
                        templates, reps
                    ):

                    f(lhs_res, rhs_res)
                )
            )

        templated_result = Var(Self.match(
            lambda id=Identifier: TypingsDescription.new(
                bindings=make_binding(
                    templates.find(lambda t: t.origin == id).then(
                        lambda t: t.instance,
                        default_val=id.domain_val._or(result_domain)
                    )
                ).singleton,
                equations=No(UnifyQuery.array)
            ),

            lambda ap=Apply: rec_apply(
                ap,
                lambda lhs_res, rhs_res: Let(
                    lambda qs=UnifyQuery.new(
                        first=lhs_res.bindings.at(0)
                        .domain_val.cast(Arrow).lhs,
                        second=rhs_res.bindings.at(0).domain_val
                    ).singleton.concat(
                        lhs_res.bindings.at(0)
                        .domain_val.cast(Arrow).binder.then(
                            lambda b: UnifyQuery.new(
                                first=b,
                                second=rhs_res.bindings.at(0)
                                .target.substitute_all(reps).dnorm
                            ).singleton
                        )
                    ):

                    TypingsDescription.new(
                        bindings=make_binding(
                            lhs_res.bindings.at(0)
                            .domain_val.cast_or_raise(Arrow).rhs
                        ).singleton.concat(
                            lhs_res.bindings
                        ).concat(
                            rhs_res.bindings
                        ),

                        equations=lhs_res.equations.concat(
                            rhs_res.equations
                        ).concat(
                            qs
                        )
                    )
                )
            ),

            lambda ab=Abstraction: Let(
                lambda
                term_res=ab.term.instantiate_templates(
                    result_domain._.cast(Arrow).rhs,
                    templates,
                    reps.filter(
                        lambda s: s.from_symbol != ab.ident.sym
                    ).concat(result_domain._.cast(Arrow).binder.then(
                        lambda b: Substitution.new(
                            from_symbol=ab.ident.sym,
                            to_term=b
                        ).singleton
                    ))
                ):

                TypingsDescription.new(
                    bindings=make_binding(
                        Self.make_arrow(
                            ab.ident.domain_val._or(
                                result_domain._.cast(Arrow).lhs
                            ),
                            term_res.bindings.at(0).domain_val,
                            result_domain._.cast(Arrow).binder
                        )
                    ).singleton.concat(term_res.bindings),
                    equations=term_res.equations
                )
            )
        ))

        return Self.domain_val.then(
            lambda expected_dom: Let(
                lambda q=UnifyQuery.new(
                    first=templated_result.bindings.at(0).domain_val,
                    second=expected_dom
                ): TypingsDescription.new(
                    bindings=templated_result.bindings,
                    equations=q.singleton.concat(templated_result.equations)
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

    @langkit_property(public=True, return_type=T.Introduction.entity,
                      memoized=True)
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

    @langkit_property(return_type=Term)
    def left_most_term():
        return Self.lhs.cast(Apply).then(
            lambda ap: ap.left_most_term,
            default_val=Self.lhs
        )

    @langkit_property(return_type=Term)
    def replace_left_most_term_with(other=Term):
        return Self.make_apply(
            Self.lhs.cast(Apply).then(
                lambda ap: ap.replace_left_most_term_with(other),
                default_val=other
            ),
            Self.rhs
        )


@synthetic
class SyntheticApply(Apply):
    pass


class Abstraction(Term):
    ident = Field(type=Identifier)
    term = Field(type=Term)

    @langkit_property()
    def to_string():
        actual_self = Var(If(
            Self.is_synthesized,
            Self.first_available_standard_symbol.then(
                lambda s: Self.make_abstraction(
                    Self.make_ident(s),
                    Self.term.rename(Self.ident.sym, s)
                ),
                default_val=Self
            ),
            Self,
        ))
        return String("(\\").concat(
            actual_self.ident.to_string.concat(String('. ')).concat(
                actual_self.term.to_string
            )
        ).concat(String(")"))

    @langkit_property(return_type=Bool)
    def is_synthesized():
        return Self.ident.to_string.contains(CharacterLiteral("$"))

    @langkit_property(return_type=Symbol)
    def first_available_standard_symbol():
        return ArrayLiteral(["x", "y", "z", "e"], element_type=T.Symbol).find(
            lambda s: Not(Self.term.is_free(s))
        )


@synthetic
class SyntheticAbstraction(Abstraction):
    pass


class Arrow(DefTerm):
    binder = Field(type=Term)
    lhs = Field(type=DefTerm)
    rhs = Field(type=DefTerm)

    @langkit_property()
    def to_string():
        return String('(').concat(Self.binder.then(
            lambda b: b.to_string.concat(String(':'))
            .concat(Self.lhs._.to_string),
            default_val=Self.lhs._.to_string
        ).concat(String(' -> ')).concat(Self.rhs.to_string)
         .concat(String(')')))

    @langkit_property(return_type=DefTerm.entity)
    def param():
        return Entity.lhs.normalized_domain

    @langkit_property(return_type=DefTerm.entity)
    def result():
        return Entity.rhs.normalized_domain

    @langkit_property(return_type=T.Bool)
    def has_constraining_binder():
        return Self.binder.then(
            lambda b: b.cast(Identifier).then(
                lambda id: If(
                    id.intro.is_null,
                    Self.lhs.is_free(id.sym) | Self.rhs.is_free(id.sym),
                    True
                ),
                default_val=True
            )
        )


@synthetic
class SyntheticArrow(Arrow):
    pass


class Introduction(DependzNode):
    """
    Identifer : DefTerm
    """
    ident = Field(type=SourceId)
    term = Field(type=DefTerm)

    @langkit_property(public=True, return_type=T.Definition.entity,
                      memoized=True)
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
            to_symbol=Self.unique_fresh_symbol(s)
        )))
        return Template.new(
            origin=origin_term,
            instance=Self.term.rename_all(renamings).dnorm,
            new_symbols=renamings.map(lambda r: r.to_symbol)
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
        return Self.check_domains_internal(
            Self.ident.intro.term.normalized_domain,
            No(Binding.array), tries
        )

    @langkit_property(public=False, return_type=T.Bool)
    def check_domains_internal(expected_domain=T.DefTerm.entity,
                               bindings=Binding.array, tries=T.Int):
        term_eq = Var(Self.term.domain_equation(bindings))
        domain_eq = Var(And(
            Bind(Self.term.domain_var, expected_domain),
            term_eq.eq
        ))
        return term_eq.templates.then(
            lambda templates: (tries != 0) & Try(
                domain_eq.solve,
                True
            ).then(lambda _: Let(
                lambda instances=templates.map(
                    lambda t: t.intro.as_template(t)
                ):

                Self.term.instantiate_templates(
                    expected_domain.node,
                    instances,
                    No(Substitution.array)
                ).then(lambda result: Self.check_domains_internal(
                    expected_domain,
                    Let(
                        lambda substs=Self.unify_all(
                            result.equations,
                            instances.mapcat(lambda i: i.instance.free_symbols)
                        ): result.bindings.map(
                            lambda b: Binding.new(
                                target=b.target,
                                domain_val=
                                b.domain_val.substitute_all(substs).dnorm
                            )
                        ).filter(
                            lambda b: b.domain_val.free_symbols.all(
                                lambda sym: Not(instances.any(
                                    lambda i: i.new_symbols.contains(sym)
                                ))
                            )
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

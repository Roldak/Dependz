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
    to_term = UserField(type=T.Term)


class Template(Struct):
    origin = UserField(type=T.Term)
    instance = UserField(type=T.Term)
    new_symbols = UserField(type=T.Symbol.array)


class Binding(Struct):
    target = UserField(type=T.Term)
    domain_val = UserField(type=T.Term)


class UnifyEquation(Struct):
    eq = UserField(type=T.Equation)
    renamings = UserField(type=Renaming.array)


class DomainEquation(Struct):
    eq = UserField(type=T.Equation)
    templates = UserField(type=T.Identifier.array)


class UnifyQuery(Struct):
    first = UserField(type=T.Term)
    second = UserField(type=T.Term)


class TypingsDescription(Struct):
    bindings = UserField(type=T.Binding.array)
    equations = UserField(type=T.UnifyQuery.array)
    new_symbols = UserField(type=T.Symbol.array)


class UnificationContext(Struct):
    symbols = UserField(type=T.Symbol.array)
    vars = UserField(type=T.LogicVarArray)


class HigherOrderUnificationContext(Struct):
    arg = UserField(type=T.Term)
    res = UserField(type=T.Term)


class ConstrainedTerm(Struct):
    term = UserField(type=T.Term)
    constraints = UserField(type=Substitution.array)


class Constructor(Struct):
    template = UserField(type=Template)
    substs = UserField(type=Substitution.array)


class SynthesisContext(Struct):
    intros = UserField(type=T.Symbol.array)


class SynthesizationHole(Struct):
    sym = UserField(type=T.Symbol)
    domain_val = UserField(type=T.Term)
    ctx = UserField(type=SynthesisContext)


class SynthesizationAttempt(Struct):
    term = UserField(type=T.Term)
    holes = UserField(type=SynthesizationHole.array)


unification_context = DynamicVariable(
    "unification_context", type=UnificationContext
)

ho_unification_context = DynamicVariable(
    "ho_unification_context", type=HigherOrderUnificationContext
)

synthesis_context = DynamicVariable(
    "synthesis_context", type=SynthesisContext
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
    def make_arrow(t1=T.Term, t2=T.Term,
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
    def make_arrow_from_self(t1=T.Term, t2=T.Term,
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

    @langkit_property(public=True, return_type=Substitution.array,
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
                    to_term=vars.elem(s).get_value._.cast(Term).rename_all(
                        renamings
                    )
                )
            ).filter(lambda s: Not(s.to_term.is_null))
        )
        new_queries = Var(queries.map(
            lambda q: UnifyQuery.new(
                first=q.first.substitute_all(substs).normalize,
                second=q.second.substitute_all(substs).normalize
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
class Term(DependzNode):
    annotations = Annotations(custom_trace_image=True)

    to_string = AbstractProperty(public=True, type=T.String)

    @langkit_property(return_type=T.Term,
                      dynamic_vars=[unification_context])
    def solve_time_substitution():
        symbols = Var(unification_context.symbols)
        vars = Var(unification_context.vars)

        substs = Var(symbols.map(
            lambda s: Substitution.new(
                from_symbol=s,
                to_term=vars.elem(s).get_value._.cast_or_raise(Term).node
            )
        ).filter(
            lambda s: Not(s.to_term.is_null)
        ))

        return Self.substitute_all(substs).normalize

    @langkit_property(return_type=T.Bool,
                      dynamic_vars=[unification_context],
                      activate_tracing=GLOBAL_ACTIVATE_TRACING)
    def unifies_with(other=T.Term):
        current_self = Var(Self.solve_time_substitution.normalize)
        current_other = Var(other.solve_time_substitution.normalize)
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
    def first_order_flexible_flexible_equation(other=T.Term):
        vars = Var(unification_context.vars)
        self_var = Var(vars.elem(Self.cast(Identifier).sym))
        other_var = Var(vars.elem(other.cast(Identifier).sym))

        return UnifyEquation.new(
            eq=Bind(self_var, other_var, eq_prop=Term.equivalent_entities),
            renamings=No(Renaming.array)
        )

    @langkit_property(return_type=UnifyEquation, uses_entity_info=False,
                      dynamic_vars=[unification_context])
    def first_order_flexible_semirigid_equation(other=T.Term):
        self_var = Var(
            unification_context.vars.elem(Self.cast(Identifier).sym)
        )

        return UnifyEquation.new(
            eq=And(
                Predicate(Term.unifies_with, self_var, other),
                LogicTrue()  # Bind(vars_in_flexible_eq, Self, conv_prop=...)
            ),
            renamings=No(Renaming.array)
        )

    @langkit_property(return_type=UnifyEquation, uses_entity_info=False,
                      dynamic_vars=[unification_context])
    def first_order_flexible_rigid_equation(other=T.Term):
        self_var = Var(
            unification_context.vars.elem(Self.cast(Identifier).sym)
        )

        return UnifyEquation.new(
            eq=Bind(
                self_var,
                other.as_bare_entity,
                eq_prop=Term.equivalent_entities
            ),
            renamings=No(Renaming.array)
        )

    @langkit_property(return_type=UnifyEquation, uses_entity_info=False,
                      dynamic_vars=[unification_context])
    def first_order_rigid_rigid_equation(other=T.Term):

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
    def first_order_unify_equation(other=T.Term):
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
            Self,
            ho_unification_context.arg
        ).unifies_with(
            ho_unification_context.res
        )

    @langkit_property(return_type=T.Term.entity,
                      dynamic_vars=[unification_context,
                                    ho_unification_context])
    def higher_order_construct_imitation():
        res = Var(ho_unification_context.res)
        body = Var(res.solve_time_substitution)
        fresh_sym = Var(body.free_fresh_symbol("ho"))

        return Entity.make_abstraction(
            Self.make_ident(fresh_sym),
            body
        ).as_bare_entity

    @langkit_property(return_type=T.Term.entity,
                      dynamic_vars=[unification_context,
                                    ho_unification_context])
    def higher_order_construct_projection():
        arg = Var(ho_unification_context.arg)
        res = Var(ho_unification_context.res)
        fresh_sym = Var(res.free_fresh_symbol("ho"))

        return Entity.make_abstraction(
            Self.make_ident(fresh_sym),
            res.solve_time_substitution.anti_substitute(
                arg.solve_time_substitution,
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
                eq_prop=Term.equivalent_entities,
                conv_prop=Term.higher_order_construct_imitation
            )
        ))

        project = Var(ho_unification_context.bind(
            ho_ctx,
            Bind(
                metavar,
                Self.as_bare_entity,
                eq_prop=Term.equivalent_entities,
                conv_prop=Term.higher_order_construct_projection
            )
        ))

        ignore = Var(ho_unification_context.bind(
            ho_ctx,
            Predicate(
                Term.higher_order_check_current_solution,
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
    def higher_order_unify_equation(other=T.Term, ho_term=T.Identifier):
        is_single_arg_equation = Var(And(
            Self.cast(Apply).lhs == ho_term,
            other.is_a(Term)
        ))
        return Cond(
            is_single_arg_equation,
            Self.higher_order_single_arg_equation(
                Self.cast(Apply).rhs,
                other,
                ho_term.sym
            ),

            UnifyEquation.new(
                eq=LogicTrue(),
                renamings=No(Renaming.array)
            )
        )

    @langkit_property(return_type=UnifyEquation, uses_entity_info=False,
                      dynamic_vars=[unification_context])
    def unify_equation(other=T.Term):
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

    @langkit_property(public=True, return_type=Substitution.array)
    def unify(other=T.Term, symbols=T.Symbol.array,
              allow_incomplete=(T.Bool, False)):
        return Self.unify_all(UnifyQuery.new(
            first=Self,
            second=other
        ).singleton, symbols, allow_incomplete)

    @langkit_property(return_type=T.Term, memoized=True)
    def final_result_domain():
        return Self.normalize.match(
            lambda ar=Arrow: ar.rhs.final_result_domain,
            lambda x: x
        )

    @langkit_property(return_type=T.Int, memoized=True)
    def param_count():
        return Self.normalize.match(
            lambda ar=Arrow: ar.rhs.param_count + 1,
            lambda _: 0
        )

    @langkit_property(return_type=T.Term.array)
    def call_args():
        return Self.match(
            lambda ap=Apply: ap.lhs.call_args.concat(ap.rhs.singleton),
            lambda _: No(Term.array)
        )

    @langkit_property(return_type=T.Bool)
    def is_call_to(sym=T.Symbol, left_args=T.Int):
        return Self.match(
            lambda i=Identifier: (i.sym == sym) & (left_args == 0),
            lambda ap=Apply: ap.lhs.is_call_to(sym, left_args - 1),
            lambda _: False
        )

    @langkit_property(return_type=T.Term.array)
    def find_calls_to(sym=T.Symbol, args=T.Int):
        return Self.match(
            lambda ap=Apply: ap.lhs.find_calls_to(sym, args).concat(
                ap.rhs.find_calls_to(sym, args)
            ),
            lambda ar=Arrow: ar.lhs.find_calls_to(sym, args).concat(
                ar.rhs.find_calls_to(sym, args)
            ).concat(ar.binder._.find_calls_to(sym, args)),
            lambda ab=Abstraction: ab.term.find_calls_to(sym, args),
            lambda _: No(Term.array)
        ).concat(Self.is_call_to(sym, args).then(
            lambda _: Self.singleton
        ))

    @langkit_property(return_type=T.Constructor.array,
                      activate_tracing=GLOBAL_ACTIVATE_TRACING)
    def constructors_impl(generics=T.Symbol.array):
        constrs = Var(Self.unit.root.cast(Program).all_constructors.map(
            lambda c: c.as_template(c.ident)
        ))

        return constrs.mapcat(
            lambda c: Let(
                lambda inst=c.instance.final_result_domain:

                Try(
                    Let(
                        lambda substs=inst.unify(
                            Self,
                            c.new_symbols.concat(generics),
                            allow_incomplete=True
                        ):

                        Constructor.new(
                            template=Template.new(
                                origin=c.origin,
                                instance=c.instance.substitute_all(substs),
                                new_symbols=c.new_symbols.filter(
                                    lambda sym: Not(substs.any(
                                        lambda subst: subst.from_symbol == sym
                                    ))
                                )
                            ),
                            substs=substs
                        ).singleton
                    ),
                    No(Constructor.array)
                )
            )
        )

    @langkit_property(return_type=T.Constructor.array.array)
    def grouped_constructors_impl(constrs=T.Constructor.array, i=T.Int):
        filtered = Var(constrs.filter(
            lambda c: c.template.instance.param_count == i
        ))
        not_filtered = Var(constrs.filter(
            lambda c: c.template.instance.param_count != i
        ))
        return filtered.singleton.concat(
            not_filtered.then(
                lambda rest: Self.grouped_constructors_impl(rest, i + 1)
            )
        )

    @langkit_property(return_type=T.Constructor.array.array)
    def grouped_constructors(generics=T.Symbol.array):
        return Self.grouped_constructors_impl(
            Self.constructors_impl(generics), 0
        )

    @langkit_property(public=True, return_type=T.Term.entity.array)
    def constructors():
        normed = Var(Self.normalize)
        generics = Var(normed.free_symbols)

        ignore(Var(Cond(
            normed.is_a(Arrow),
            PropertyError(T.Bool, "Cannot list constructors of arrow types"),

            normed.is_a(Abstraction),
            PropertyError(T.Bool, "Abstractions are not valid domains"),

            True
        )))

        return normed.constructors_impl(generics).map(
            lambda c: c.template.origin.as_bare_entity
        )

    @langkit_property(return_type=SynthesizationAttempt,
                      dynamic_vars=[synthesis_context])
    def synthesize_abstraction():
        ar = Var(Self.cast_or_raise(Arrow))

        sym = Var(Self.unique_fresh_symbol(ar.binder._.cast(Identifier).then(
            lambda id: id.sym,
            default_val="x"
        )))
        id = Var(Self.make_ident(sym))

        body_attempt = Var(ar.binder.then(
            lambda b: b.cast(Identifier).then(
                lambda i: i.intro.then(
                    lambda _: ar.rhs,
                    default_val=ar.rhs.substitute(i.sym, id)
                ),
                default_val=PropertyError(
                    Term, "Non-identifier binders are not handled yet."
                )
            ),
            default_val=ar.rhs
        ).then(
            lambda rhs: synthesis_context.bind(
                SynthesisContext.new(
                    intros=synthesis_context.intros.concat(sym.singleton)
                ),
                rhs.synthesize_impl
            )
        ))

        return SynthesizationAttempt.new(
            term=Self.make_abstraction(id, body_attempt.term),
            holes=body_attempt.holes
        )

    @langkit_property(return_type=SynthesizationAttempt,
                      dynamic_vars=[synthesis_context])
    def synthesize_apply_arrow(built=T.Term, ar=T.Arrow):
        binder = Var(ar.binder)

        is_introduced = Var(binder.then(
            lambda b: b.free_symbols.all(
                lambda s: synthesis_context.intros.contains(s)
            )
        ))

        arg = Var(If(
            is_introduced,
            SynthesizationAttempt.new(
                term=binder,
                holes=No(SynthesizationHole.array)
            ),
            ar.lhs.synthesize_impl
        ))

        new_built = Var(Self.make_apply(built, arg.term))

        rhs_type = Var(If(
            binder.is_null | is_introduced,
            ar.rhs,
            ar.rhs.substitute(
                binder.cast(Identifier).then(
                    lambda b: b.sym,
                    default_val=PropertyError(
                        Symbol, "Non-identifier binders are not handled yet."
                    )
                ),
                arg.term
            )
        ))

        rec = Var(Self.synthesize_apply(new_built, rhs_type))

        return SynthesizationAttempt.new(
            term=rec.term,
            holes=rec.holes.concat(arg.holes),
        )

    @langkit_property(return_type=SynthesizationAttempt,
                      dynamic_vars=[synthesis_context],
                      activate_tracing=GLOBAL_ACTIVATE_TRACING)
    def synthesize_apply(built=T.Term, callee_type=T.Term):
        return callee_type.match(
            lambda ar=Arrow:
            Self.synthesize_apply_arrow(built, ar),

            lambda _: SynthesizationAttempt.new(
                term=built,
                holes=No(SynthesizationHole.array)
            )
        )

    @langkit_property(return_type=Constructor.array)
    def synthesizable_constructors(generics=T.Symbol.array):
        return Self.grouped_constructors(generics).mapcat(
            lambda constrs: constrs
        )

    @langkit_property(return_type=SynthesizationAttempt,
                      dynamic_vars=[synthesis_context])
    def synthesize_impl():
        return Self.normalize.match(
            lambda ar=Arrow: ar.synthesize_abstraction,
            lambda ab=Abstraction: PropertyError(
                SynthesizationAttempt, "Abstractions are not valid domains"
            ),
            lambda other: Let(
                lambda hole=Self.make_ident(Self.unique_fresh_symbol("hole")):

                SynthesizationAttempt.new(
                    term=hole,
                    holes=SynthesizationHole.new(
                        sym=hole.sym,
                        domain_val=other,
                        ctx=synthesis_context
                    ).singleton
                )
            )
        )

    @langkit_property(return_type=SynthesizationAttempt.array,
                      activate_tracing=GLOBAL_ACTIVATE_TRACING)
    def synthesize_attempt(attempt=SynthesizationAttempt,
                           origin=T.Introduction):
        hole_syms = Var(attempt.holes.map(lambda h: h.sym))
        first_hole = Var(attempt.holes.at(0))

        dom = Var(first_hole.domain_val)

        constrs = Var(
            first_hole.domain_val.synthesizable_constructors(hole_syms)
        )
        return synthesis_context.bind(first_hole.ctx, constrs.map(
            lambda c: dom.synthesize_apply(
                c.template.origin,
                c.template.instance
            ).then(lambda atp: Let(
                lambda substs=Substitution.new(
                    from_symbol=first_hole.sym,
                    to_term=atp.term
                ).singleton.concat(c.substs): SynthesizationAttempt.new(
                    term=attempt.term.substitute_all(substs, unsafe=True),
                    holes=attempt.holes.concat(atp.holes).filtermap(
                        lambda h: SynthesizationHole.new(
                            sym=h.sym,
                            domain_val=h.domain_val.substitute_all(substs),
                            ctx=h.ctx
                        ),
                        lambda h: Not(substs.any(
                            lambda s: s.from_symbol == h.sym
                        ))
                    )
                )
            ))
        ))

    @langkit_property(return_type=T.Term,
                      activate_tracing=GLOBAL_ACTIVATE_TRACING)
    def synthesize_breadth_first_search(attempts=SynthesizationAttempt.array,
                                        origin=T.Introduction, depth=T.Int):
        result = Var(attempts.find(lambda atp: atp.holes.length == 0))
        return Cond(
            Not(result.is_null),
            result.term,

            depth == 0,
            attempts.at(0).term,

            Self.synthesize_breadth_first_search(
                attempts.mapcat(
                    lambda atp: Self.synthesize_attempt(atp, origin)
                ),
                origin,
                depth - 1
            )
        )

    @langkit_property(public=True, return_type=T.Term)
    def synthesize(origin=(T.Introduction, No(T.Introduction))):
        return synthesis_context.bind(
            SynthesisContext.new(
                intros=No(Symbol.array)
            ),
            Self.synthesize_breadth_first_search(
                Self.synthesize_impl.singleton,
                origin,
                10
            )
        )

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
    def free_fresh_symbol(prefix=T.Symbol, other=(T.Term, No(T.Term)),
                          i=(T.Int, 0)):
        sym = Var(Self.concat_symbol_and_integer(prefix, i))
        return If(
            And(Self.is_free(sym), Or(other.is_null, other.is_free(sym))),
            Self.free_fresh_symbol(prefix, other, i + 1),
            sym
        )

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

            # todo: handle arrows here?
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
        elim_id = Var(elim_evaled._.lhs)
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
            )._or(ab),
            lambda ar=Arrow: ar
        )

    @langkit_property(public=True, return_type=T.Bool)
    def equivalent_entities(other=T.Term.entity):
        return Entity.node.equivalent(other.node)

    @langkit_property(return_type=T.Bool, memoized=True)
    def equivalent(other=T.Term):
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

    @langkit_property(return_type=T.Term)
    def substitute_all(substs=Substitution.array, idx=(T.Int, 0),
                       unsafe=(T.Bool, False)):
        return substs.at(idx).then(
            lambda r: Self.substitute(
                r.from_symbol,
                r.to_term,
                unsafe
            ).substitute_all(
                substs, idx + 1, unsafe
            ),
            default_val=Self
        )

    @langkit_property(public=True, return_type=T.Term, memoized=True)
    def substitute(sym=T.Symbol, val=T.Term, unsafe=(T.Bool, False)):
        return Self.match(
            lambda id=Identifier: If(
                id.sym == sym,
                val,
                id
            ),
            lambda ap=Apply: ap.make_apply(
                ap.lhs.substitute(sym, val, unsafe),
                ap.rhs.substitute(sym, val, unsafe)
            ),
            lambda ab=Abstraction: If(
                ab.ident.sym == sym,
                ab,
                If(
                    val.is_free(ab.ident.sym) & Not(unsafe),
                    ab.term.free_fresh_symbol(ab.ident.sym).then(
                        lambda symp: ab.make_abstraction(
                            ab.make_ident(symp),
                            ab.term
                            .rename(ab.ident.sym, symp)
                            .substitute(sym, val, False)
                        )
                    ),
                    ab.make_abstraction(
                        ab.ident,
                        ab.term.substitute(sym, val, unsafe)
                    )
                )
            ),
            lambda ar=Arrow: Self.make_arrow(
                ar.lhs.substitute(sym, val, unsafe),
                ar.rhs.substitute(sym, val, unsafe),
                ar.binder._.substitute(sym, val, unsafe)
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
                ),
                lambda ar=Arrow: ar.make_arrow(
                    ar.lhs.anti_substitute(val, sym),
                    ar.rhs.anti_substitute(val, sym),
                    ar.binder._.anti_substitute(val, sym)
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
                ),
                lambda ar=Arrow: Or(
                    ar.lhs.contains_term(t, True),
                    ar.rhs.contains_term(t, True),
                    ar.binder._.contains_term(t, True)
                )
            )
        )

    @langkit_property(return_type=T.Term.entity)
    def normalized_entities():
        return Entity.node.normalize.as_entity

    @langkit_property(public=True, return_type=T.Term, memoized=True)
    def normalize():
        evaled = Var(Self.eval)
        to_norm = Var(If(
            # prevent infinite evaluation
            evaled.contains_term(Self),
            Self,
            evaled
        ))

        return to_norm.match(
            lambda id=Identifier: id,
            lambda ap=Apply: ap.make_apply(
                ap.lhs.normalize,
                ap.rhs.normalize
            ),
            lambda ab=Abstraction: ab.make_abstraction(
                ab.ident,
                ab.term.normalize
            ),
            lambda ar=Arrow: ar.make_arrow(
                ar.lhs.normalize,
                ar.rhs.normalize,
                ar.binder._.normalize
            )
        )

    @langkit_property(return_type=T.Term)
    def rename_all(renamings=Renaming.array, idx=(T.Int, 0)):
        return renamings.at(idx).then(
            lambda r:
            Self.rename(r.from_symbol, r.to_symbol).rename_all(
                renamings, idx + 1
            ),
            default_val=Self
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
            ),
            lambda ar=Arrow: ar.make_arrow(
                ar.lhs.rename(old, by),
                ar.rhs.rename(old, by),
                ar.binder._.rename(old, by)
            )
        )

    @langkit_property(return_type=T.LogicVar, memoized=True)
    def domain_var():
        return Self.create_logic_var

    @langkit_property(return_type=T.Identifier.array)
    def find_occurrences(sym=T.Symbol):
        return Self.match(
            lambda id=Identifier: If(
                id.sym == sym,
                id.singleton,
                No(Identifier.array)
            ),

            lambda ap=Apply:
            ap.lhs.find_occurrences(sym)
            .concat(ap.rhs.find_occurrences(sym)),

            lambda ab=Abstraction: If(
                ab.ident.sym == sym,
                No(Identifier.array),
                ab.term.find_occurrences(sym)
            ),

            lambda ar=Arrow:
            ar.lhs.find_occurrences(sym)
            .concat(ar.rhs.find_occurrences(sym))
            .concat(ar.binder._.find_occurrences(sym))
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
                            Self.domain_var, intro.term.normalized_entities,
                            conv_prop=Term.normalized_entities,
                            eq_prop=Term.equivalent_entities
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
                             eq_prop=Term.equivalent_entities),
                        Bind(ap.lhs.domain_var, ap.domain_var,
                             conv_prop=Arrow.result,
                             eq_prop=Term.equivalent_entities)
                    ),
                    templates=lhs_eq.templates.concat(rhs_eq.templates)
                )
            ),
            lambda ab=Abstraction: Let(
                lambda term_eq=ab.term.domain_equation(bindings):

                DomainEquation.new(
                    eq=And(
                        ab.term.find_occurrences(ab.ident.sym).logic_all(
                            lambda id: Bind(
                                ab.ident.domain_var, id.domain_var,
                                conv_prop=Term.normalized_entities,
                                eq_prop=Term.equivalent_entities
                            ),
                        ),
                        Bind(ab.domain_var, ab.ident.domain_var,
                             conv_prop=Arrow.param,
                             eq_prop=Term.equivalent_entities),
                        Bind(ab.domain_var, ab.term.domain_var,
                             conv_prop=Arrow.result,
                             eq_prop=Term.equivalent_entities),
                        term_eq.eq
                    ),
                    templates=term_eq.templates
                )
            ),

            lambda ar=Arrow: Let(
                lambda
                lhs_eq=ar.lhs.domain_equation(bindings),
                rhs_eq=ar.rhs.domain_equation(bindings),
                binder_eq=ar.binder.then(
                    lambda b: b.domain_equation(bindings),
                    default_val=DomainEquation.new(
                        eq=LogicTrue(),
                        templates=No(Identifier.array)
                    )
                ):

                DomainEquation.new(
                    eq=And(
                        lhs_eq.eq,
                        rhs_eq.eq,
                        binder_eq.eq,

                        ar.binder.then(
                            lambda b: Bind(
                                b.domain_var, ar.lhs.normalize.as_bare_entity,
                                eq_prop=Term.equivalent_entities
                            ),
                            default_val=LogicTrue()
                        ),

                        Or(
                            Bind(ar.domain_var, ar.lhs.domain_var,
                                 eq_prop=Term.equivalent_entities),
                            Bind(ar.domain_var, ar.rhs.domain_var,
                                 eq_prop=Term.equivalent_entities)
                        ),
                        Predicate(Term.is_highest_ranked_term, ar.domain_var,
                                  ar.lhs.domain_var, ar.rhs.domain_var)
                    ),
                    templates=lhs_eq.templates.concat(
                        rhs_eq.templates
                    ).concat(
                        binder_eq.templates
                    )
                )
            )
        ))

        return relevant_binding.then(
            lambda b: DomainEquation.new(
                eq=Bind(
                    Self.domain_var, b.domain_val.as_bare_entity,
                    eq_prop=Term.equivalent_entities
                ) & result.eq,
                templates=result.templates
            ),
            default_val=result
        )

    @langkit_property(public=False, return_type=TypingsDescription,
                      activate_tracing=GLOBAL_ACTIVATE_TRACING)
    def instantiate_templates(result_domain=T.Term,
                              templates=Template.array,
                              reps=T.Substitution.array):
        must_synthesize_arrow = Var(
            Self.cast(Abstraction).then(
                lambda ab: result_domain.cast(Arrow).then(
                    lambda ar: False,
                    default_val=True
                ),
                default_val=False
            )
        )
        actual_result_domain = Var(If(
            must_synthesize_arrow,
            Let(
                lambda
                lhs_sym=Self.unique_fresh_symbol("lhs"),
                rhs_sym=Self.unique_fresh_symbol("rhs"),
                binder_sym=Self.unique_fresh_symbol("binder"):

                Self.make_arrow(
                    Self.make_ident(lhs_sym),
                    Self.make_ident(rhs_sym),
                    Self.make_ident(binder_sym)
                )
            ),
            result_domain
        ))
        instantiation = Var(Self.instantiate_templates_impl(
            actual_result_domain,
            templates,
            reps
        ))
        return If(
            must_synthesize_arrow,
            Let(
                lambda ar=actual_result_domain.cast_or_raise(Arrow):

                TypingsDescription.new(
                    bindings=instantiation.bindings,
                    equations=instantiation.equations.concat(
                        UnifyQuery.new(
                            first=ar,
                            second=result_domain
                        ).singleton
                    ),
                    new_symbols=instantiation.new_symbols.concat(
                        ArrayLiteral([
                            ar.lhs.cast(Identifier).sym,
                            ar.rhs.cast(Identifier).sym,
                            ar.binder.cast(Identifier).sym
                        ])
                    )
                )
            ),
            instantiation
        )

    @langkit_property(public=False, return_type=TypingsDescription)
    def instantiate_templates_impl(result_domain=T.Term,
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
                    result_domain.then(
                        lambda r_dom: Self.make_arrow(No(T.Term), r_dom)
                    ),
                    templates,
                    reps
                ): Let(
                    lambda
                    rhs_res=ap.rhs.instantiate_templates(
                        lhs_res.bindings.at(0).domain_val.cast(Arrow)._.lhs,
                        templates, reps
                    ):

                    f(lhs_res, rhs_res)
                )
            )

        templated_result = Var(Self.match(
            lambda id=Identifier: TypingsDescription.new(
                bindings=templates.find(lambda t: t.origin == id).then(
                    lambda t: t.instance,
                    default_val=id.domain_val._or(result_domain)
                ).then(
                    lambda dom: make_binding(dom)
                ).singleton,

                equations=No(UnifyQuery.array),

                new_symbols=No(Symbol.array)
            ),

            lambda ap=Apply: rec_apply(
                ap,
                lambda lhs_res, rhs_res: Let(
                    lambda qs=rhs_res.bindings.at(0).domain_val.then(
                        lambda rhs_dom: UnifyQuery.new(
                            first=lhs_res.bindings.at(0)
                            .domain_val.cast(Arrow).lhs,
                            second=rhs_dom
                        ).singleton
                    ).concat(
                        lhs_res.bindings.at(0)
                        .domain_val.cast(Arrow)._.binder.then(
                            lambda b: UnifyQuery.new(
                                first=b,
                                second=rhs_res.bindings.at(0)
                                .target.substitute_all(reps).normalize
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
                        ),

                        new_symbols=lhs_res.new_symbols.concat(
                            rhs_res.new_symbols
                        )
                    )
                )
            ),

            lambda ab=Abstraction: Let(
                lambda
                term_res=ab.term.instantiate_templates(
                    result_domain.cast(Arrow)._.rhs,
                    templates,
                    reps.filter(
                        lambda s: s.from_symbol != ab.ident.sym
                    ).concat(result_domain.cast(Arrow)._.binder.then(
                        lambda b: Substitution.new(
                            from_symbol=ab.ident.sym,
                            to_term=b
                        ).singleton
                    ))
                ),
                id_dom=ab.ident.domain_val._or(
                    result_domain.cast(Arrow)._.lhs
                ):

                TypingsDescription.new(
                    bindings=id_dom.then(
                        lambda lhs_dom:

                        term_res.bindings.at(0).domain_val.then(
                            lambda rhs_dom: make_binding(
                                Self.make_arrow(
                                    lhs_dom,
                                    rhs_dom,
                                    result_domain.cast(Arrow)._.binder
                                )
                            )
                        )
                    ).singleton.concat(term_res.bindings),

                    equations=term_res.equations.concat(id_dom.then(
                        lambda lhs_dom:
                        ab.term.find_occurrences(ab.ident.sym).map(
                            lambda id: term_res.bindings.find(
                                lambda b: b.target == id
                            ).then(
                                lambda b: UnifyQuery.new(
                                    first=lhs_dom,
                                    second=b.domain_val
                                )
                            )
                        )
                    )),

                    new_symbols=term_res.new_symbols
                )
            ),

            lambda ar=Arrow: Let(
                lambda
                lhs_res=ar.lhs.instantiate_templates(
                    No(Term), templates, reps
                ),
                rhs_res=ar.rhs.instantiate_templates(
                    No(Term), templates, reps
                ),
                binder_res=ar.binder.then(
                    lambda b: b.instantiate_templates(
                        ar.lhs, templates, reps
                    )
                ),
                new_sym=Self.unique_fresh_symbol("arrow"):

                TypingsDescription.new(
                    bindings=make_binding(
                        Self.make_ident(new_sym)
                    ).singleton.concat(
                        lhs_res.bindings
                    ).concat(
                        rhs_res.bindings
                    ).concat(
                        binder_res.bindings
                    ),

                    equations=lhs_res.equations.concat(
                        rhs_res.equations
                    ).concat(
                        binder_res.equations
                    ),

                    new_symbols=new_sym.singleton.concat(
                        lhs_res.new_symbols
                    ).concat(
                        rhs_res.new_symbols
                    )
                )
            )
        ))

        return Self.domain_val.then(
            lambda expected_dom:

            templated_result.bindings.at(0).domain_val.then(
                lambda found_dom: Let(
                    lambda q=UnifyQuery.new(
                        first=found_dom,
                        second=expected_dom
                    ):

                    TypingsDescription.new(
                        bindings=templated_result.bindings,
                        equations=q.singleton.concat(
                            templated_result.equations
                        ),
                        new_symbols=templated_result.new_symbols
                    )
                )
            ),
            default_val=templated_result
        )

    @langkit_property(return_type=T.Term, public=True)
    def domain_val():
        return Self.domain_var.get_value._.node.cast_or_raise(Term)

    @langkit_property(return_type=T.Bool)
    def is_highest_ranked_term(fst=DependzNode.entity, snd=DependzNode.entity):
        t1 = Var(fst.cast_or_raise(Term).node)
        t2 = Var(snd.cast_or_raise(Term).node)
        return Self.equivalent(t1.self_or_higher_ranked_term(t2))

    @langkit_property(return_type=T.Term, memoized=True)
    def self_or_higher_ranked_term(other=T.Term):
        self_chain = Var(Self.domain_chain)
        other_chain = Var(other.domain_chain)
        first_common = Var(self_chain.find(
            lambda d1: other_chain.any(
                lambda d2: d1.equivalent(d2)
            )
        ))
        return first_common._or(PropertyError(Term, "Terms are incompatible"))

    @langkit_property(return_type=T.Term.array, memoized=True)
    def domain_chain():
        id = Var(Self.cast_or_raise(Identifier))
        domain = Var(id.intro._.term)
        return Self.singleton.concat(If(
            domain.cast(Identifier).then(lambda d: d.intro._.term == domain),
            domain.node.singleton,
            domain._.domain_chain
        ))


@abstract
class Identifier(Term):
    sym = AbstractProperty(type=Symbol)

    to_string = Property(Self.sym.image)

    @langkit_property(public=True, return_type=T.Bool)
    def is_introducing():
        return Self.parent.cast(Introduction).then(
            lambda i: i.ident == Self
        )

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


class Arrow(Term):
    binder = Field(type=Term)
    lhs = Field(type=Term)
    rhs = Field(type=Term)

    @langkit_property()
    def to_string():
        lhs_str = Var(Self.lhs._.to_string)
        rhs_str = Var(Self.rhs.to_string)
        return String('(').concat(Self.binder.then(
            lambda b: If(
                Self.has_constraining_binder,
                b.to_string.concat(String(':')).concat(lhs_str),
                lhs_str
            ),
            default_val=lhs_str
        ).concat(String(' -> ')).concat(rhs_str).concat(String(')')))

    @langkit_property(return_type=Term.entity)
    def param():
        return Entity.lhs.normalized_entities

    @langkit_property(return_type=Term.entity)
    def result():
        return Entity.rhs.normalized_entities

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
    Identifer : Term
    """
    ident = Field(type=SourceId)
    term = Field(type=Term)

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
            instance=Self.term.rename_all(renamings).normalize,
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
            Self.ident.intro.term.normalized_entities,
            No(Binding.array), tries
        )

    @langkit_property(public=False, return_type=T.Bool)
    def check_domains_internal(expected_domain=T.Term.entity,
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
                            instances.mapcat(
                                lambda i: i.instance.free_symbols
                            ).concat(
                                result.new_symbols
                            )
                        ): result.bindings.map(
                            lambda b: Binding.new(
                                target=b.target,
                                domain_val=
                                b.domain_val.substitute_all(substs).normalize
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
    @langkit_property(return_type=Introduction.array,
                      memoized=True)
    def all_introductions():
        return Self.children_env.get(No(T.Symbol)).filtermap(
            lambda n: n.cast(Introduction).node,
            lambda n: n.is_a(Introduction)
        )

    @langkit_property(return_type=Introduction.array,
                      memoized=True)
    def all_constructors():
        return Self.all_introductions.filter(
            lambda n: n.definition.is_null
        )

    env_spec = EnvSpec(
        add_env()
    )

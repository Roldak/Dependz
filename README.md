# Dependz
Small dependent type calculus written in Langkit

# Examples

Here is what it looks like:

```
# Definition of the naturals:
Nat : Type
Z : Nat
S : Nat -> Nat

# Definition of Vectors of fixed lengths over arbitrary types:
Vec : Type -> Nat -> Type
Nil : Vec t Z
Cons : t -> Vec t n -> Vec t (S n)
```

Note that free variables `t` and `n` in the definitions above are *implicits* and will be automatically be derived from the context when needed. For example:
```
main : Vec Nat (S Z)
main = Cons Z Nil
```
In the call to `Cons`, both `t` and `n` will be bound automatically. By the way, you can test that the following does not type check:
```
main : Vec Nat (S Z)
main = Cons Z (Cons Z Nil)
```
because the type inferred for `Cons Z (Cons Z Nil)` is `Vec Nat (S (S Z))` and it does not unify with `Vec Nat (S Z)`!

Here is an example of how you can write a prototype for your `add` function that receives two vectors of the same length `n` and also returns a vector of length `n`:
```
add : Vec Nat n -> Vec Nat n -> Vec Nat n
```

You can check that the following does not type check:
```
add (Cons (S Z) Nil) (Cons Z (Cons (S Z) Nil))
```

More interesting, it is possible to define the equality type in order to prove that two terms are equal:
```
Eq : t -> t -> Type
Refl : Eq x x
```
The intuition is that the Refl is the only constructor of `Eq`, and it takes two times the term `x`. So when you successfully construct a term `Refl a b` it must mean that `a` and `b` are equal.

Don't hesitate to have a look at the testsuite for more examples.

# Implementation details

Dependz is implemented using the Langkit framework. In particular, the lexer, parser, type checker and evaluator are all written using Langkit's DSL (with the help if some trivial extensions that can be found in `dependz/extensions`).

Type checking/inference is a done using both Langkit's equation solver (mainly for verification) and custom ad-hoc unification mechanisms.

Note that the main missing feature is term deconstruction (via pattern matching or custom constructs), although a prototype (via `elim_type` built-ins) is under development and already type checks.


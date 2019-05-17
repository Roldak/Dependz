from __future__ import absolute_import, division, print_function

from langkit.lexer import (
    Lexer, LexerToken, Literal, WithText, WithSymbol, WithTrivia, Pattern
)


class Token(LexerToken):
    Ident = WithSymbol()

    Colon = WithText()
    Arrow = WithText()
    Equal = WithText()
    ParOpen = WithText()
    ParClose = WithText()

    Comment = WithTrivia()
    Whitespace = WithTrivia()
    Newlines = WithText()


dependz_lexer = Lexer(Token)

dependz_lexer.add_rules(
    # Blanks and trivia
    (Pattern(r"[ \r\t]+"), Token.Whitespace),
    (Pattern(r"[\n]+"), Token.Newlines),
    (Pattern(r"#(.?)+"), Token.Comment),

    (Pattern('[a-zA-Z_][a-zA-Z0-9_]*'), Token.Ident),
    (Literal(':'), Token.Colon),
    (Literal('->'), Token.Arrow),
    (Literal('='), Token.Equal),
    (Literal('('), Token.ParOpen),
    (Literal(')'), Token.ParClose)
)
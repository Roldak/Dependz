#! /usr/bin/env python

import os

from langkit.libmanage import ManageScript


class Manage(ManageScript):
    def create_context(self, args):
        from langkit.compile_context import CompileCtx

        from language.lexer import dependz_lexer
        from language.grammar import dependz_grammar

        return CompileCtx(lang_name='Dependz',
                          short_name='LDL',
                          lexer=dependz_lexer,
                          grammar=dependz_grammar)


if __name__ == '__main__':
    Manage().run()


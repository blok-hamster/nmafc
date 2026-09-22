"""Turn one Python file into symbols and the references between them.

Uses `ast` from the standard library, which means this is exact, offline, and
free. That is the whole cost argument for code memory: embedding a repository
costs real money per rebuild and yields a probabilistic index, while parsing it
costs milliseconds and yields the definitions themselves.

A file that does not parse produces no symbols rather than an exception. Half a
repository is normally syntactically valid mid-edit, and an index that refuses
to build while a file is broken is an index that is unavailable exactly when it
is most wanted.
"""

from __future__ import annotations

import ast
import builtins
import hashlib

from nmafc.schemas.code import FileRecord, Symbol, SymbolKind

# Referenced names worth recording as edges. Builtins are excluded because
# `len`, `print` and `range` appear in nearly every function and an edge that
# joins everything to everything carries no information -- the same reason the
# clustering coefficient rather than raw degree was the right decay signal.
_IGNORED_REFS = frozenset(dir(builtins))


def content_hash(text: str) -> str:
    """sha256 of exactly the bytes shown, so a match means the source is the
    source that was indexed and not merely a file of the same name."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _refs(node: ast.AST, module_names: frozenset[str] = frozenset()) -> list[str]:
    """Names this definition depends on: calls, base classes, decorators.

    Attribute calls contribute the attribute (`self._router.retrieve` gives
    `retrieve`), because that is the name the definition being looked for is
    declared under. The receiver is not recorded: `self` and `_router` join a
    method to every other method on the same object, which is the everything-to-
    everything edge again.

    A bare name is recorded only when `module_names` says it is bound at module
    level in this file, or imported into it. Constants are referenced by name and
    never by call, so without this a `BINDING` symbol has no reverse edges at all
    and "what uses DEFAULT_TIMEOUT" is unanswerable -- which is half the value of
    indexing it. Recording *every* bare name instead would be the
    everything-to-everything edge this function exists to avoid: a local variable
    called `config` in forty functions would join all forty to a module named
    config. Restricting it to names the module actually binds keeps the edge
    meaningful, and costs one pass over `tree.body` to collect.

    Nested definitions are skipped, and skipped by not descending into them
    rather than by ignoring the node itself. `ast.walk` cannot express this --
    it flattens the whole subtree, so declining to record a nested `def` still
    records everything inside it, and a class ends up owning every call its
    methods make. That makes `callers("helper")` answer with both the method
    that calls it and the class the method happens to sit in, which is exactly
    the everything-to-everything edge this function exists to avoid.
    """
    out: list[str] = []
    nested = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

    def record(child: ast.AST) -> None:
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                out.append(func.id)
            elif isinstance(func, ast.Attribute):
                out.append(func.attr)
        elif isinstance(child, ast.Attribute):
            out.append(child.attr)
        elif (isinstance(child, ast.Name) and child.id in module_names
                and isinstance(child.ctx, ast.Load)):
            # Reads only. The target of an assignment is being bound, not
            # referenced, so counting it makes `DEFAULT_TIMEOUT = 30` a user of
            # itself and puts every binding in its own caller list.
            out.append(child.id)

    def descend(parent: ast.AST) -> None:
        for child in ast.iter_child_nodes(parent):
            if isinstance(child, nested):
                continue
            record(child)
            descend(child)

    descend(node)

    if isinstance(node, ast.ClassDef):
        for base in node.bases:
            if isinstance(base, ast.Name):
                out.append(base.id)
            elif isinstance(base, ast.Attribute):
                out.append(base.attr)

    decorators = getattr(node, "decorator_list", [])
    for dec in decorators:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name):
            out.append(target.id)
        elif isinstance(target, ast.Attribute):
            out.append(target.attr)

    seen: dict[str, None] = {}
    for name in out:
        if name not in _IGNORED_REFS and not name.startswith("__"):
            seen.setdefault(name, None)
    return list(seen)


def _signature(lines: list[str], node: ast.AST) -> str:
    """The definition's header, from `def`/`class` to the colon that opens it.

    Taken from the source rather than unparsed from the AST, because the point
    of a code memory is to show what is written -- including the type
    annotations and default values as the author spelled them.
    """
    start = node.lineno - 1
    collected: list[str] = []
    depth = 0
    for raw in lines[start:start + 40]:
        collected.append(raw.strip())
        depth += raw.count("(") + raw.count("[") - raw.count(")") - raw.count("]")
        if depth <= 0 and raw.rstrip().endswith(":"):
            break
    return " ".join(collected)


def _bound_names(node: ast.AST) -> list[str]:
    """Module-level names this statement binds, if it is a plain assignment.

    Only bare names. `config.timeout = 30` and `table[key] = value` mutate
    something that is defined elsewhere, so recording them as definitions would
    put two entries in the index claiming to be the same thing. Tuple unpacking
    is taken apart, because `HOT, COLD = 0, 1` really does define both.

    Dunders are skipped: `__all__` and `__version__` are packaging metadata, and
    nobody asks a code memory where `__all__` is defined.
    """
    if isinstance(node, ast.AnnAssign):
        targets = [node.target]
    elif isinstance(node, ast.Assign):
        targets = node.targets
    else:
        return []

    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.extend(el.id for el in target.elts if isinstance(el, ast.Name))
    return [n for n in names if not n.startswith("__")]


def _binding_signature(lines: list[str], node: ast.AST, name: str) -> str:
    """The assignment as written, on one line, truncated if it runs long.

    Not `_signature`, which hunts for the colon that opens a block and would run
    to its forty-line limit on an assignment that has no colon at all.
    """
    end = getattr(node, "end_lineno", node.lineno) or node.lineno
    text = " ".join(line.strip() for line in lines[node.lineno - 1:end])
    return text if len(text) <= 160 else text[:157] + "..."


def _docline(node: ast.AST) -> str:
    doc = ast.get_docstring(node)
    if not doc:
        return ""
    return doc.strip().split("\n", 1)[0].strip()


def parse_source(path: str, source: str) -> tuple[FileRecord, list[Symbol]]:
    """Every definition in one file, with its span, hash and references.

    Methods are recorded under `Class.method` and reported as METHOD, so that a
    reference to a bare `retrieve` can be resolved to every class that defines
    one and the caller can choose. Flattening them to bare names would lose the
    owner; recording only the class would lose the method.
    """
    file_hash = content_hash(source)
    record = FileRecord(path=path, content_hash=file_hash)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # A file mid-edit is normal. Returning an empty parse keeps the rest of
        # the index usable and lets the caller see, from `symbol_keys`, that
        # this file currently contributes nothing.
        return record, []

    lines = source.split("\n")
    symbols: list[Symbol] = []

    # Everything the module binds at its top level, collected before anything is
    # emitted. `_refs` needs it to decide whether a bare name is a reference to
    # something this file defines or just a local variable, and that decision
    # cannot be made one statement at a time -- a function on line 10 routinely
    # uses a constant defined on line 200.
    module_names = frozenset(
        [n for node in tree.body for n in _bound_names(node)]
        + [alias.asname or alias.name.split(".")[0]
           for node in tree.body
           if isinstance(node, (ast.Import, ast.ImportFrom))
           for alias in node.names]
    )

    def emit(node, qualname: str, kind: SymbolKind,
             signature: str | None = None) -> None:
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        body = "\n".join(lines[node.lineno - 1:end])
        symbols.append(Symbol(
            path=path,
            qualname=qualname,
            kind=kind,
            lineno=node.lineno,
            end_lineno=end,
            signature=(_signature(lines, node) if signature is None
                       else signature),
            docline=_docline(node) if signature is None else "",
            refs=_refs(node, module_names),
            content_hash=content_hash(body),
        ))

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", "") or ""
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                record.imports[bound] = (
                    f"{module}.{alias.name}" if module else alias.name
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            emit(node, node.name, SymbolKind.FUNCTION)
        elif isinstance(node, ast.ClassDef):
            emit(node, node.name, SymbolKind.CLASS)
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    emit(member, f"{node.name}.{member.name}", SymbolKind.METHOD)
        else:
            for name in _bound_names(node):
                emit(node, name, SymbolKind.BINDING,
                     signature=_binding_signature(lines, node, name))

    record.symbol_keys = [s.key for s in symbols]
    return record, symbols

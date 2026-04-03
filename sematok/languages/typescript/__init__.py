"""
TypeScript language configuration for sematok.

Covers both .ts and .tsx files using the TSX tree-sitter grammar (superset).
Patterns span the TypeScript ecosystem: React/Vue/Angular/Svelte frontends,
Node.js backends (NestJS, Fastify, Hono, Express), ORMs (Prisma, TypeORM,
Drizzle), state management, testing, and utility/FP libraries.
"""

import re

import tree_sitter_typescript as tstypescript
from tree_sitter import Language

from sematok.languages import LanguageConfig


# ---------------------------------------------------------------------------
# Safe-zone override: treat JSDoc comments as compressible
# ---------------------------------------------------------------------------

def _is_jsdoc(node, source_bytes: bytes) -> bool:
    """Treat JSDoc comments (/** ... */) as safe/compressible.

    JSDoc contains highly repetitive boilerplate: @param, @returns, @throws,
    @deprecated, @example, {@link}, {@code}, etc.  Regular block comments
    (/* ... */) and line comments (// ...) remain unsafe.

    License headers (/** @license ... */) are excluded — they are
    repo-specific text, not compressible boilerplate.
    """
    if node.type != "comment":
        return False
    text = source_bytes[node.start_byte:node.end_byte]
    if not text.startswith(b"/**"):
        return False
    # Exclude license headers (e.g. Angular's /** @license ... */)
    if b"@license" in text:
        return False
    return True


# ---------------------------------------------------------------------------
# Node types (verified with tree-sitter-typescript 0.23.2, TSX grammar)
# ---------------------------------------------------------------------------

UNSAFE_NODE_TYPES = {
    "comment",           # Both // and /* */ (single type in TS grammar)
    "string",            # 'single' and "double" quoted strings
    "template_string",   # `backtick` template literals
}

# ---------------------------------------------------------------------------
# Candidate patterns -- regex extractors for boilerplate discovery
# ---------------------------------------------------------------------------

CANDIDATE_PATTERNS = [re.compile(p) for p in [
    # --- Imports ---
    r"import\s+\{[^}]+\}\s+from\s+['\"]",
    r"import\s+type\s+\{[^}]+\}\s+from\s+['\"]",
    r"import\s+\*\s+as\s+\w+\s+from\s+['\"]",
    r"import\s+\w+\s+from\s+['\"]",
    # --- Exports ---
    r"export\s+(?:default\s+)?(?:function|class|const|let|type|interface|enum)\s",
    # --- Type annotations ---
    r":\s*(?:string|number|boolean|void|never|unknown|any|undefined|null)\b",
    r":\s*(?:Promise|Array|Record|Partial|Readonly|Required|Omit|Pick)<",
    r":\s*(?:React\.)?(?:FC|ReactNode|ReactElement|JSX\.Element)\b",
    # --- Generic constraints ---
    r"<\w+\s+extends\s+\w",
    r"<\w+(?:,\s*\w+)*>",
    # --- React hooks ---
    r"use(?:State|Effect|Callback|Memo|Ref|Context|Reducer)\s*[<(]",
    # --- Decorators (NestJS, TypeORM, Angular) ---
    r"@(?:Injectable|Component|Controller|Module|Entity|Column|Guard|Pipe)\s*\(",
    # --- Async patterns ---
    r"async\s+(?:function\s+)?\w+\s*[<(]",
    r"await\s+\w",
    # --- Error handling ---
    r"throw\s+new\s+\w+Error\(",
    r"catch\s*\(\w+(?::\s*\w+)?\)",
    # --- Arrow functions with types ---
    r"\)\s*(?::\s*\w[^=]*)?=>\s*[{(]",
    # --- Common patterns ---
    r"console\.(?:log|error|warn|info|debug)\(",
    r"(?:describe|it|test|expect|beforeEach|afterEach)\s*\(",
    # --- JSDoc tags ---
    r"@(?:param|returns?|throws|deprecated|example|see|since|override)\b",
    # --- Optional chaining / nullish ---
    r"\?\.\w+",
    r"\?\?",
    # --- Type assertions ---
    r"\bas\s+\w",
    # --- satisfies ---
    r"\bsatisfies\s+\w",
    # --- Common method chains ---
    r"\.(?:then|catch|finally)\s*\(",
    r"\.(?:map|filter|reduce|forEach|find|some|every)\s*\(",
]]

# ---------------------------------------------------------------------------
# Seed patterns -- high-frequency exact-match (pattern, category) pairs
# ---------------------------------------------------------------------------

SEED_PATTERNS = [
    # --- Imports (12) ---
    ("import { ", "import"),
    ("import type { ", "import"),
    (" } from '", "import"),
    (' } from "', "import"),
    ("import * as ", "import"),
    ("from 'react'", "import"),
    ("from 'react';", "import"),
    ("from 'next/", "import"),
    ("from '@/", "import"),
    ("from '../", "import"),
    ("from './", "import"),
    ("from 'node:", "import"),

    # --- Exports (6) ---
    ("export default ", "export"),
    ("export const ", "export"),
    ("export function ", "export"),
    ("export interface ", "export"),
    ("export type ", "export"),
    ("export async function ", "export"),

    # --- Type annotations (10) ---
    (": string", "type"),
    (": number", "type"),
    (": boolean", "type"),
    (": void", "type"),
    (": unknown", "type"),
    (": Promise<", "type"),
    (": Record<", "type"),
    (": Partial<", "type"),
    ("| undefined", "type"),
    ("| null", "type"),

    # --- React (10) ---
    ("useState<", "react"),
    ("useState(", "react"),
    ("useEffect(() => {", "react"),
    ("useCallback(", "react"),
    ("useMemo(", "react"),
    ("useRef<", "react"),
    ("useContext(", "react"),
    ("React.FC<", "react"),
    (": React.ReactNode", "react"),
    ("className={", "react"),

    # --- Async / Promise (6) ---
    ("async function ", "async"),
    ("async () => {", "async"),
    ("await ", "async"),
    ("Promise.all(", "async"),
    ("Promise.resolve(", "async"),
    (".then(", "async"),

    # --- Error handling (6) ---
    ("throw new Error(", "error"),
    ("throw new TypeError(", "error"),
    ("catch (error)", "error"),
    ("catch (error: unknown)", "error"),
    ("catch (e)", "error"),
    ("} finally {", "error"),

    # --- Console / logging (4) ---
    ("console.log(", "logging"),
    ("console.error(", "logging"),
    ("console.warn(", "logging"),
    ("console.info(", "logging"),

    # --- Testing (8) ---
    ("describe('", "test"),
    ('describe("', "test"),
    ("it('", "test"),
    ('it("', "test"),
    ("expect(", "test"),
    ("test('", "test"),
    ('test("', "test"),
    ("beforeEach(", "test"),

    # --- JSDoc (6) ---
    ("@param ", "jsdoc"),
    ("@returns ", "jsdoc"),
    ("@throws ", "jsdoc"),
    ("@deprecated", "jsdoc"),
    ("@example", "jsdoc"),
    ("{@link ", "jsdoc"),

    # --- Common expressions (7) ---
    ("return null;", "expression"),
    ("return undefined;", "expression"),
    ("=== null", "expression"),
    ("=== undefined", "expression"),
    ("!== null", "expression"),
    ("!== undefined", "expression"),
    ("?? ", "expression"),
]

# ---------------------------------------------------------------------------
# AST mining: subtree root types
# ---------------------------------------------------------------------------

SUBTREE_ROOT_TYPES = {
    "expression_statement", "lexical_declaration",
    "return_statement", "throw_statement",
    "if_statement", "for_statement", "for_in_statement",
    "while_statement", "do_statement",
    "try_statement", "switch_statement",
    "function_declaration", "arrow_function",
    "class_declaration", "interface_declaration",
    "type_alias_declaration", "enum_declaration",
    "import_statement", "export_statement",
    "call_expression", "new_expression",
    "assignment_expression", "variable_declarator",
    "jsx_element", "jsx_self_closing_element",
    "decorator",
}

# ---------------------------------------------------------------------------
# Template mining: identifier classification
# ---------------------------------------------------------------------------

FIXED_PARENT_TYPES = {
    "class_declaration", "interface_declaration",
    "enum_declaration", "type_alias_declaration",
    "function_declaration", "method_definition",
    "call_expression", "new_expression",
    "member_expression",
    "import_statement", "import_specifier",
    "export_statement",
    "decorator",
    "type_identifier", "generic_type", "type_arguments",
    "catch_clause",
    "jsx_opening_element", "jsx_closing_element",
    "jsx_self_closing_element",
}

NORMALIZE_PARENT_TYPES = {
    "variable_declarator", "assignment_expression",
    "return_statement", "throw_statement",
    "arguments", "binary_expression", "unary_expression",
    "conditional_expression",
    "array", "object", "pair",
    "arrow_function", "formal_parameters",
    "required_parameter", "optional_parameter",
    "jsx_attribute", "jsx_expression",
    "template_substitution",
}

# ---------------------------------------------------------------------------
# Structural names -- well-known identifiers that should never be normalized
# ---------------------------------------------------------------------------

STRUCTURAL_NAMES = {
    # --- Keywords / literals ---
    "this", "super", "undefined", "null", "true", "false",
    "void", "never", "unknown", "any", "typeof", "keyof",
    "infer", "readonly", "declare", "abstract", "override",
    "as", "is", "in", "of", "satisfies", "asserts",

    # --- Primitive types ---
    "string", "number", "boolean", "bigint", "symbol", "object", "Function",

    # --- Built-in objects ---
    "Array", "Object", "Map", "Set", "WeakMap", "WeakSet",
    "Promise", "Date", "RegExp",
    "Error", "TypeError", "RangeError", "ReferenceError", "SyntaxError",
    "URIError", "EvalError", "AggregateError",
    "JSON", "Math", "Number", "String", "Boolean", "Symbol", "BigInt",
    "Proxy", "Reflect",
    "ArrayBuffer", "SharedArrayBuffer", "DataView",
    "Int8Array", "Uint8Array", "Uint8ClampedArray",
    "Int16Array", "Uint16Array", "Int32Array", "Uint32Array",
    "Float32Array", "Float64Array", "BigInt64Array", "BigUint64Array",

    # --- Utility types ---
    "Partial", "Required", "Readonly", "Record",
    "Pick", "Omit", "Exclude", "Extract",
    "NonNullable", "ReturnType", "Parameters",
    "ConstructorParameters", "InstanceType", "ThisType",
    "Awaited", "Uppercase", "Lowercase", "Capitalize", "Uncapitalize",
    "NoInfer",

    # --- Collections / iterators ---
    "Iterator", "AsyncIterator", "Generator", "AsyncGenerator",
    "Iterable", "AsyncIterable", "IterableIterator",
    "PromiseLike", "PromiseSettledResult",

    # --- DOM types ---
    "HTMLElement", "HTMLDivElement", "HTMLInputElement",
    "HTMLButtonElement", "HTMLFormElement", "HTMLAnchorElement",
    "HTMLImageElement", "HTMLCanvasElement", "HTMLSelectElement",
    "HTMLTextAreaElement", "HTMLSpanElement",
    "Element", "Node", "Document", "Window",
    "Event", "MouseEvent", "KeyboardEvent", "FocusEvent",
    "TouchEvent", "PointerEvent", "InputEvent", "CustomEvent",
    "EventTarget", "NodeList", "EventListener",
    "DOMRect", "DOMMatrix", "MutationObserver", "ResizeObserver",
    "IntersectionObserver",

    # --- React types ---
    "FC", "ReactNode", "ReactElement", "JSX",
    "Component", "PureComponent", "Fragment", "Suspense", "StrictMode",
    "Ref", "MutableRefObject", "RefObject",
    "Dispatch", "SetStateAction",
    "ChangeEvent", "FormEvent", "SyntheticEvent",
    "CSSProperties", "PropsWithChildren", "ComponentProps",
    "ComponentType", "ElementType",

    # --- React hooks ---
    "useState", "useEffect", "useCallback", "useMemo",
    "useRef", "useContext", "useReducer",
    "useLayoutEffect", "useImperativeHandle",
    "useDeferredValue", "useTransition", "useId",

    # --- Node.js globals ---
    "Buffer", "process", "console",
    "require", "module", "exports",
    "__dirname", "__filename",
    "setTimeout", "setInterval", "clearTimeout", "clearInterval",
    "setImmediate", "queueMicrotask", "globalThis",

    # --- Web APIs ---
    "fetch", "Request", "Response", "Headers",
    "URL", "URLSearchParams", "FormData",
    "Blob", "File", "FileReader",
    "ReadableStream", "WritableStream", "TransformStream",
    "AbortController", "AbortSignal",
    "WebSocket", "MessagePort", "BroadcastChannel",
    "Crypto", "crypto", "TextEncoder", "TextDecoder",
    "performance", "navigator", "document", "window", "location",

    # --- Common method names ---
    "toString", "valueOf", "hasOwnProperty", "constructor", "prototype",
    "length", "name", "size",
    "push", "pop", "shift", "unshift",
    "slice", "splice", "concat", "join",
    "map", "filter", "reduce", "forEach",
    "find", "findIndex", "some", "every",
    "includes", "indexOf", "lastIndexOf",
    "keys", "values", "entries",
    "flat", "flatMap", "fill", "sort", "reverse",
    "has", "get", "set", "delete", "clear",
    "add", "next", "done", "value",
    "then", "catch", "finally",
    "resolve", "reject", "all", "allSettled", "race", "any",
    "assign", "freeze", "create", "defineProperty",
    "parse", "stringify",
    "from", "of", "isArray",
    "startsWith", "endsWith", "trim", "split", "replace",
    "match", "search", "test",
    "apply", "bind", "call",
    "log", "error", "warn", "info", "debug",
    "emit", "on", "off", "once", "removeListener",

    # --- Framework / decorator names ---
    "Injectable", "Component", "Controller", "Module",
    "Entity", "Column", "PrimaryColumn", "PrimaryGeneratedColumn",
    "Guard", "Pipe", "Middleware", "Interceptor",
    "Service", "Repository", "Resolver",
    "Query", "Mutation", "Subscription",
    "Schema", "Model", "Table", "Field", "Input",
    "Get", "Post", "Put", "Delete", "Patch",
    "Body", "Param", "Req", "Res",
    "Inject", "Optional", "UseGuards", "UseInterceptors",
    "Prop", "Watch", "Emit",

    # --- Testing ---
    "describe", "it", "test", "expect",
    "beforeAll", "afterAll", "beforeEach", "afterEach",
    "jest", "vi", "mock", "fn", "spyOn",
    "toBe", "toEqual", "toHaveBeenCalled", "toThrow",
    "toContain", "toBeTruthy", "toBeFalsy", "toBeNull",
    "toBeUndefined", "toBeDefined", "toHaveLength",
    "toMatchSnapshot", "toMatchInlineSnapshot",

    # --- Common identifiers ---
    "default", "async", "await",
    "new", "class", "interface", "type", "enum",
    "extends", "implements",
    "import", "export", "from",
    "const", "let", "var",
    "return", "throw", "yield",
    "if", "else", "switch", "case",
    "for", "while", "do", "break", "continue",
    "try", "catch", "finally",
    "static", "private", "protected", "public",
}

# ---------------------------------------------------------------------------
# Paths to skip during corpus extraction
# ---------------------------------------------------------------------------

SKIP_PATH_PATTERNS = [
    "/node_modules/",
    "/dist/",
    "/build/",
    "/out/",
    ".d.ts",
    ".min.js",
    ".min.ts",
    "/generated/",
    "/__generated__/",
    "/coverage/",
    "/.next/",
    "/.nuxt/",
    "/.svelte-kit/",
]

# ---------------------------------------------------------------------------
# Repositories: 58 training + 10 evaluation (all permissive licenses)
# ---------------------------------------------------------------------------

REPOS = [
    # --- Frontend frameworks (7) ---
    ("vuejs", "core"),                            # MIT          | Vue 3
    ("angular", "angular"),                       # MIT          | Angular
    ("sveltejs", "svelte"),                       # MIT          | Svelte 5
    ("solidjs", "solid"),                         # MIT          | SolidJS
    ("preactjs", "preact"),                       # MIT          | Lightweight React alt
    ("excalidraw", "excalidraw"),                 # MIT          | Collaborative whiteboard
    ("radix-ui", "primitives"),                   # MIT          | Headless UI primitives

    # --- Meta-frameworks (5) ---
    ("vercel", "next.js"),                        # MIT          | React SSR/SSG
    ("nuxt", "nuxt"),                             # MIT          | Vue meta-framework
    ("remix-run", "remix"),                       # MIT          | Full-stack React
    ("withastro", "astro"),                       # MIT          | Content-first SSG
    ("sveltejs", "kit"),                          # MIT          | SvelteKit

    # --- Backend frameworks (5) ---
    ("nestjs", "nest"),                           # MIT          | Enterprise Node.js
    ("honojs", "hono"),                           # MIT          | Edge/serverless framework
    ("trpc", "trpc"),                             # MIT          | End-to-end typesafe RPC
    ("fastify", "fastify"),                       # MIT          | High-perf web server
    ("socketio", "socket.io"),                    # MIT          | Real-time communication

    # --- State management (5) ---
    ("reduxjs", "redux-toolkit"),                 # MIT          | Redux official toolkit
    ("pmndrs", "zustand"),                        # MIT          | Minimal state management
    ("pmndrs", "jotai"),                          # MIT          | Atomic state management
    ("TanStack", "query"),                        # MIT          | Async state management
    ("mobxjs", "mobx"),                           # MIT          | Reactive state

    # --- Data / ORM / API (7) ---
    ("prisma", "prisma"),                         # Apache-2.0   | Next-gen TypeScript ORM
    ("drizzle-team", "drizzle-orm"),              # MIT          | Lightweight SQL ORM
    ("typeorm", "typeorm"),                       # MIT          | Decorator-based ORM
    ("kysely-org", "kysely"),                     # MIT          | Type-safe query builder
    ("apollographql", "apollo-client"),           # MIT          | GraphQL client
    ("MichalLytek", "type-graphql"),              # MIT          | GraphQL schema builder
    ("ReactiveX", "rxjs"),                        # Apache-2.0   | Reactive programming

    # --- UI component libraries (4) ---
    ("ant-design", "ant-design"),                 # MIT          | Enterprise React UI
    ("mui", "material-ui"),                       # MIT          | Material Design React
    ("storybookjs", "storybook"),                 # MIT          | Component dev environment
    ("xyflow", "xyflow"),                         # MIT          | React Flow / node-based UI

    # --- Developer tools (5) ---
    ("microsoft", "TypeScript"),                  # Apache-2.0   | TypeScript compiler
    ("typescript-eslint", "typescript-eslint"),   # MIT          | TS ESLint tooling
    ("vitejs", "vite"),                           # MIT          | Next-gen build tool
    ("vitest-dev", "vitest"),                     # MIT          | Vite-native test framework
    ("microsoft", "playwright"),                  # Apache-2.0   | E2E testing framework

    # --- Validation / schema (4) ---
    ("colinhacks", "zod"),                        # MIT          | TS-first schema validation
    ("react-hook-form", "react-hook-form"),       # MIT          | Performant React forms
    ("typestack", "class-validator"),             # MIT          | Decorator validation
    ("typestack", "class-transformer"),           # MIT          | Serialization/deserialization

    # --- Utility / FP (5) ---
    ("date-fns", "date-fns"),                     # MIT          | Date utility library
    ("Effect-TS", "effect"),                      # MIT          | Functional effects library
    ("gcanti", "fp-ts"),                          # MIT          | Functional programming
    ("gcanti", "io-ts"),                          # MIT          | Runtime type validation
    ("vercel", "swr"),                            # MIT          | Stale-while-revalidate

    # --- Auth (2) ---
    ("nextauthjs", "next-auth"),                  # MIT          | Auth for Next.js
    ("lucia-auth", "lucia"),                      # MIT          | Simple auth library

    # --- AI / ML (2) ---
    ("langchain-ai", "langchainjs"),              # MIT          | LLM framework
    ("vercel", "ai"),                             # MIT          | AI SDK

    # --- Documentation / content (2) ---
    ("facebook", "docusaurus"),                   # MIT          | React doc site generator
    ("vuejs", "vitepress"),                       # MIT          | Vue doc framework

    # --- Monorepo / build (2) ---
    ("nrwl", "nx"),                               # MIT          | Monorepo management
    ("changesets", "changesets"),                  # MIT          | Versioning/changelog

    # --- Routing / navigation (1) ---
    ("TanStack", "router"),                       # MIT          | Type-safe routing

    # --- Collaborative / real-time (1) ---
    ("tldraw", "tldraw"),                         # Apache-2.0   | Collaborative whiteboard SDK

    # --- Headless CMS / commerce (1) ---
    ("medusajs", "medusa"),                       # MIT          | Headless commerce

    # --- React Native (2) ---
    ("expo", "expo"),                             # MIT          | RN development platform
    ("react-navigation", "react-navigation"),     # MIT          | RN navigation library

    # --- Electron / desktop (2) ---
    ("microsoft", "vscode"),                      # MIT          | VS Code editor
    ("toeverything", "AFFiNE"),                   # MIT          | Knowledge base / whiteboard
]

EVAL_REPOS = [
    "vitest-dev--vitest",                         # MIT          | Testing framework
    "honojs--hono",                               # MIT          | Backend framework
    "pmndrs--zustand",                            # MIT          | State management
    "colinhacks--zod",                            # MIT          | Schema validation
    "withastro--astro",                           # MIT          | Meta-framework
    "drizzle-team--drizzle-orm",                  # MIT          | ORM
    "Effect-TS--effect",                          # MIT          | FP library
    "tldraw--tldraw",                             # Apache-2.0   | Collaborative app
    "langchain-ai--langchainjs",                  # MIT          | AI/ML
    "ant-design--ant-design",                     # MIT          | UI components
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_config() -> LanguageConfig:
    """Return the TypeScript language configuration."""
    return LanguageConfig(
        name="typescript",
        file_extension=".ts",
        tree_sitter_language=Language(tstypescript.language_tsx()),
        source_extensions=[".ts", ".tsx"],

        unsafe_node_types=UNSAFE_NODE_TYPES,
        is_safe_override=_is_jsdoc,

        candidate_patterns=CANDIDATE_PATTERNS,
        seed_patterns=SEED_PATTERNS,

        subtree_root_types=SUBTREE_ROOT_TYPES,
        fixed_parent_types=FIXED_PARENT_TYPES,
        normalize_parent_types=NORMALIZE_PARENT_TYPES,
        structural_names=STRUCTURAL_NAMES,

        ident_pattern=r"([a-zA-Z_$][\w$]*)",

        skip_path_patterns=SKIP_PATH_PATTERNS,

        repos=REPOS,
        eval_repos=EVAL_REPOS,
    )
